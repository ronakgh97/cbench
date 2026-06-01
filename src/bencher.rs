use crate::load::StaticMemPool;
use anyhow::Context;
use blas_rs::lvl3::gemm;
use blas_rs::utils::gen_fill;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

pub const SAMPLE_SIZE: usize = 2156;

pub async fn run_benchmark(runs: usize, warmups: usize, max_thread: usize) -> anyhow::Result<()> {
    println!(
        "Warmup runs: {}, Benchmark runs: {}, Threads: {}",
        warmups, runs, max_thread
    );

    let mem_pool = Arc::new(StaticMemPool::<[f32; SAMPLE_SIZE * SAMPLE_SIZE], 16>::init());

    let pool_clone = mem_pool.clone();
    tokio::spawn(async move {
        ctrl_c().await;
        println!("Stopping...");
        drop(pool_clone);
        std::process::exit(0);
    });

    let mat_len = SAMPLE_SIZE * SAMPLE_SIZE;

    // warmup phase (single-thread)
    {
        let mut block_a = mem_pool.try_alloc_zeroed().context("Pool exhausted")?;
        let mut block_b = mem_pool.try_alloc_zeroed().context("Pool exhausted")?;
        gen_fill(&mut *block_a);
        gen_fill(&mut *block_b);

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

    // generate new matrices
    let mut matrix_vec = Vec::with_capacity(runs);
    for _ in 0..runs {
        let mut block_a = mem_pool.try_alloc_zeroed().context("Pool exhausted")?;
        let mut block_b = mem_pool.try_alloc_zeroed().context("Pool exhausted")?;
        gen_fill(&mut *block_a);
        gen_fill(&mut *block_b);
        matrix_vec.push((block_a, block_b));
    }

    let mut score: Vec<(f64, f64)> = vec![(0.0, 0.0); runs];

    for (i, matrics) in matrix_vec.iter().enumerate().take(runs) {
        let start = Instant::now();
        if max_thread == 1 {
            let mut block_c = mem_pool.try_alloc_zeroed().context("Pool exhausted")?;
            gemm(
                SAMPLE_SIZE,
                SAMPLE_SIZE,
                SAMPLE_SIZE,
                1.0,
                &*matrics.0,
                SAMPLE_SIZE,
                &*matrics.1,
                SAMPLE_SIZE,
                0.0,
                &mut *block_c,
                SAMPLE_SIZE,
                false,
                false,
            );
            black_box(&*block_c);
        } else {
            let mut scratch: Vec<Vec<f32>> =
                (0..max_thread).map(|_| vec![0.0f32; mat_len]).collect();
            std::thread::scope(|s| {
                for res in &mut scratch {
                    s.spawn(|| {
                        gemm(
                            SAMPLE_SIZE,
                            SAMPLE_SIZE,
                            SAMPLE_SIZE,
                            1.0,
                            &*matrics.0,
                            SAMPLE_SIZE,
                            &*matrics.1,
                            SAMPLE_SIZE,
                            0.0,
                            res,
                            SAMPLE_SIZE,
                            false,
                            false,
                        );
                        black_box(&*res);
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
            "Run {}: Time = {:.3?}s, GFLOPS = {:.2}",
            i + 1,
            elapsed,
            gflops
        );
    }

    println!("-------------------------------------");
    let avg_gflops = score.iter().map(|s| s.1).sum::<f64>() / (score.len() as f64);
    let total_time = score.iter().map(|s| s.0).sum::<f64>();
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
