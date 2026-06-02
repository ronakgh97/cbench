use crate::load::StaticMemPool;
use anyhow::Context;
use blas_rs::lvl3::gemm;
use blas_rs::utils::Noise;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

const BASE_SCORE: usize = 1280;
pub const SAMPLE_SIZE: usize = 2156;

pub async fn run_benchmark(runs: usize, warmups: usize, max_thread: usize) -> anyhow::Result<()> {
    println!(
        "Warmup runs: {} Benchmark runs: {} Threads: {}\n",
        warmups, runs, max_thread
    );
    println!("Running BLAS bench...");

    let mem_pool = Arc::new(StaticMemPool::<[f32; SAMPLE_SIZE * SAMPLE_SIZE], 64>::init());
    let mut noise = Noise::init();
    let pool_clone = mem_pool.clone();
    tokio::spawn(async move {
        ctrl_c().await;
        println!("Stopping...");
        drop(pool_clone);
        std::process::exit(0);
    });

    // warmup phase (single-thread)
    {
        let mut block_a = mem_pool.try_alloc_zeroed().context("Pool exhausted")?;
        let mut block_b = mem_pool.try_alloc_zeroed().context("Pool exhausted")?;
        noise.fill_f32(&mut *block_a);
        noise.fill_f32(&mut *block_b);

        for _ in 0..warmups {
            let mut block_c = mem_pool.try_alloc_zeroed().context("Pool exhausted")?;
            gemm(
                SAMPLE_SIZE,
                SAMPLE_SIZE,
                SAMPLE_SIZE,
                1.0,
                &*block_a,
                SAMPLE_SIZE,
                &*block_b,
                SAMPLE_SIZE,
                0.0,
                &mut *block_c,
                SAMPLE_SIZE,
                false,
                false,
            );
            black_box(&*block_c);
        }
    }

    let mut score = vec![(0.0f64, 0.0f64); runs];

    #[allow(clippy::needless_range_loop)]
    for i in 0..runs {
        let mut block_a = mem_pool.try_alloc_zeroed().context("Pool exhausted")?;
        let mut block_b = mem_pool.try_alloc_zeroed().context("Pool exhausted")?;
        noise.fill_f32(&mut *block_a);
        noise.fill_f32(&mut *block_b);

        let start = Instant::now();
        if max_thread == 1 {
            let mut block_c = mem_pool.try_alloc_zeroed().context("Pool exhausted")?;
            gemm(
                SAMPLE_SIZE,
                SAMPLE_SIZE,
                SAMPLE_SIZE,
                1.0,
                &*block_a,
                SAMPLE_SIZE,
                &*block_b,
                SAMPLE_SIZE,
                0.0,
                &mut *block_c,
                SAMPLE_SIZE,
                false,
                false,
            );
            black_box(&*block_c);
        } else {
            std::thread::scope(|s| {
                let a_ref: &[f32] = block_a.get();
                let b_ref: &[f32] = block_b.get();
                for _ in 0..max_thread {
                    let mut block = mem_pool
                        .try_alloc_zeroed()
                        .expect("Pool exhausted during thread allocation");
                    s.spawn(move || {
                        gemm(
                            SAMPLE_SIZE,
                            SAMPLE_SIZE,
                            SAMPLE_SIZE,
                            1.0,
                            a_ref,
                            SAMPLE_SIZE,
                            b_ref,
                            SAMPLE_SIZE,
                            0.0,
                            block.get_mut(),
                            SAMPLE_SIZE,
                            false,
                            false,
                        );
                        black_box(block.get());
                    });
                }
            });
        }
        let elapsed = start.elapsed().as_secs_f64();

        let flops_per_mul = 2.0 * (SAMPLE_SIZE as f64).powi(3) - (SAMPLE_SIZE as f64).powi(2);
        let total_flops = flops_per_mul * (max_thread as f64);
        let gflops = total_flops / elapsed / 1e9;

        score[i] = (elapsed, gflops);

        println!(
            "Run {}: Time = {:.3?}s GFLOPS = {:.2}",
            i + 1,
            elapsed,
            gflops
        );
    }

    // TODO: run crypto here

    println!("-------------------------------------");
    let avg_gflops = score[..runs].iter().map(|s| s.1).sum::<f64>() / (runs as f64);
    let total_time = score[..runs].iter().map(|s| s.0).sum::<f64>();
    let cpu_score = BASE_SCORE + (avg_gflops + 1000.0) as usize;
    println!("Estimated CPU Score: {}", cpu_score);
    println!("Average GFLOPS/core: {:.2}", avg_gflops / max_thread as f64);
    println!("Total time: {}s", total_time as f32);

    println!("Find your CPU here: https://boinc.bakerlab.org/rosetta/cpu_list.php");

    Ok(())
}

async fn ctrl_c() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed Ctrl+C handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
