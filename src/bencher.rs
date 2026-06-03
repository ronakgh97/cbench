use crate::load::{NONCE_LEN, StaticMemPool, TAG_LEN, decrypt_buf, encrypt_buf};
use anyhow::Context;
use blas_rs::lvl3::gemm;
use blas_rs::utils::Noise;
use sha3::{Digest, Sha3_512};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

const BASE_SCORE: usize = 1280;
pub const MATRIX_SIZE: usize = 2048;
pub const SAMPLE_SIZE: usize = MATRIX_SIZE * MATRIX_SIZE;
pub const MAX_RUN: usize = 18;
pub const MAX_WARMUP: usize = 8;
pub const POOL_CAPACITY: usize = MAX_RUN * 3; // each bench run need 3 blocks

pub async fn run_benchmark(warmups: usize, runs: usize, max_thread: usize) -> anyhow::Result<()> {
    let mut noise = Noise::init();
    println!(
        "Warmup runs: {} Benchmark runs: {} Threads: {}\n",
        warmups, runs, max_thread
    );

    println!("Running BLAS bench...");
    let mem_pool = Arc::new(StaticMemPool::<[f32; SAMPLE_SIZE], { POOL_CAPACITY }>::init());
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
                MATRIX_SIZE,
                MATRIX_SIZE,
                MATRIX_SIZE,
                1.0,
                &*block_a,
                MATRIX_SIZE,
                &*block_b,
                MATRIX_SIZE,
                0.0,
                &mut *block_c,
                MATRIX_SIZE,
                false,
                false,
            );
            black_box(&*block_c);
        }
    }

    let mut blas_score = vec![(0.0f64, 0.0f64); runs];

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
                MATRIX_SIZE,
                MATRIX_SIZE,
                MATRIX_SIZE,
                1.0,
                &*block_a,
                MATRIX_SIZE,
                &*block_b,
                MATRIX_SIZE,
                0.0,
                &mut *block_c,
                MATRIX_SIZE,
                false,
                false,
            );
            black_box(&*block_c);
        } else {
            std::thread::scope(|s| {
                let block_a: &[f32] = block_a.get();
                let block_b: &[f32] = block_b.get();
                for _ in 0..max_thread {
                    let mut block_c = mem_pool
                        .try_alloc_zeroed()
                        .expect("Pool exhausted during thread allocation");
                    s.spawn(move || {
                        gemm(
                            MATRIX_SIZE,
                            MATRIX_SIZE,
                            MATRIX_SIZE,
                            1.0,
                            block_a,
                            MATRIX_SIZE,
                            block_b,
                            MATRIX_SIZE,
                            0.0,
                            block_c.get_mut(),
                            MATRIX_SIZE,
                            false,
                            false,
                        );
                        black_box(block_c.get());
                    });
                }
            });
        }
        let elapsed = start.elapsed().as_secs_f64();

        let flops_per_mul = 2.0 * (MATRIX_SIZE as f64).powi(3) - (MATRIX_SIZE as f64).powi(2);
        let total_flops = flops_per_mul * (max_thread as f64);
        let gflops = total_flops / elapsed / 1e9;

        blas_score[i] = (elapsed, gflops);

        println!(
            "Run {}: Time = {:.3?}s GFLOPS = {:.2}",
            i + 1,
            elapsed,
            gflops
        );
    }

    drop(mem_pool); // free the pool before crypto bench

    let mut crypto_score = vec![(0.0f64, 0.0f64); runs];
    println!("\nRunning Crypto bench...");
    let mem_pool = Arc::new(StaticMemPool::<[u8; SAMPLE_SIZE], { POOL_CAPACITY * 3 }>::init());
    let pool_clone = mem_pool.clone();
    tokio::spawn(async move {
        ctrl_c().await;
        println!("Stopping...");
        drop(pool_clone);
        std::process::exit(0);
    });

    let mut key = [0u8; 32];
    noise.fill_bytes(&mut key);

    let plan_text_len = SAMPLE_SIZE - (NONCE_LEN + TAG_LEN);
    #[allow(clippy::needless_range_loop)]
    for i in 0..runs {
        let mut block_a = mem_pool.try_alloc_zeroed().context("Pool exhausted")?;
        let mut block_b = mem_pool.try_alloc_zeroed().context("Pool exhausted")?;
        let mut block_c = mem_pool.try_alloc_zeroed().context("Pool exhausted")?;
        noise.fill_bytes(&mut *block_a);
        noise.fill_bytes(&mut *block_b);
        noise.fill_bytes(&mut *block_c);
        let start_time = Instant::now();
        use core::arch::x86_64::{__rdtscp, _mm_lfence, _rdtsc};

        let start = unsafe {
            _mm_lfence();
            _rdtsc()
        };

        if max_thread == 1 {
            let plan_text: &mut [u8] = &mut block_a.get_mut()[..plan_text_len];
            let cipher_text: &mut [u8] = block_b.get_mut();
            black_box(&plan_text);
            black_box(&cipher_text);
            black_box(&block_c);

            crypto_stress(plan_text, cipher_text, key, &mut *block_c)
                .context("Crypto stress failed")?;
        } else {
            std::thread::scope(|s| {
                for _ in 0..max_thread {
                    let mut thread_block_a = mem_pool
                        .try_alloc_zeroed()
                        .expect("Pool exhausted during thread allocation");
                    let mut thread_block_b = mem_pool
                        .try_alloc_zeroed()
                        .expect("Pool exhausted during thread allocation");
                    let mut thread_block_c = mem_pool
                        .try_alloc_zeroed()
                        .expect("Pool exhausted during thread allocation");
                    noise.fill_bytes(&mut *thread_block_a);
                    noise.fill_bytes(&mut *thread_block_b);
                    noise.fill_bytes(&mut *thread_block_c);

                    s.spawn(move || {
                        let plan_text: &mut [u8] = &mut thread_block_a.get_mut()[..plan_text_len];
                        let cipher_text: &mut [u8] = thread_block_b.get_mut();
                        black_box(&plan_text);
                        black_box(&cipher_text);
                        black_box(&thread_block_c);

                        crypto_stress(plan_text, cipher_text, key, &mut *thread_block_c)
                            .expect("Crypto stress failed");
                    });
                }
            })
        }

        let mut aux = 0;
        let end = unsafe {
            let e = __rdtscp(&mut aux);
            _mm_lfence();
            e
        };
        let cycles = (end - start) as f64;
        let elapsed = start_time.elapsed().as_secs_f64();
        crypto_score[i] = (elapsed, cycles);

        println!("Run {}: Time = {:.3?}s Cycles = {}", i + 1, elapsed, cycles);
    }

    println!("---------------------------------------");
    let avg_gflops = blas_score[..runs].iter().map(|s| s.1).sum::<f64>() / (runs as f64);
    let total_blas_time = blas_score[..runs].iter().map(|s| s.0).sum::<f64>();

    let avg_crypto_cycles = crypto_score[..runs].iter().map(|s| s.1).sum::<f64>() / (runs as f64);
    let total_crypto_time = crypto_score[..runs].iter().map(|s| s.0).sum::<f64>();

    let cpu_score = BASE_SCORE + (avg_gflops + avg_crypto_cycles) as usize;
    println!("Estimated CPU Score: {}", cpu_score);
    println!("Average SCORE/Core: {:.2}", (cpu_score / max_thread) as f32);
    println!("Total time: {:.2}s", total_blas_time + total_crypto_time);

    println!("Find your CPU here: https://boinc.bakerlab.org/rosetta/cpu_list.php");

    Ok(())
}

fn crypto_stress(
    plan_text: &mut [u8],
    cipher_text: &mut [u8],
    key: [u8; 32],
    hash_buf: &mut [u8],
) -> anyhow::Result<()> {
    encrypt_buf(plan_text, cipher_text, &key)?;
    decrypt_buf(cipher_text, plan_text, &key)?;
    Sha3_512::digest(&hash_buf);
    encrypt_buf(plan_text, cipher_text, &key)?;
    decrypt_buf(cipher_text, plan_text, &key)?;
    Sha3_512::digest(&hash_buf);
    encrypt_buf(plan_text, cipher_text, &key)?;
    decrypt_buf(cipher_text, plan_text, &key)?;
    Sha3_512::digest(&hash_buf);
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
