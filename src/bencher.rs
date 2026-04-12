use crate::load::{generate_matrix, matrix_matrix_mul_into};
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

// TODO: Random generation are pain and should consider inside bench loop, Need better scoring
// TODO: Currently this runs matmul on given threads, and compute `flops` * thread_num,
//  but I think we should parallelize matmul computation internally and fix the flops calculation....not run them in each threads?
//  so it can count as single matmul kernel efficiency, not System thread scheduling efficiency, which is not what we want to measure.

const SAMPLE_SIZE: usize = 2048;

pub fn run_benchmark(runs: usize, warmups: Option<usize>, max_thread: usize) -> anyhow::Result<()> {
    let warmup_runs = warmups.unwrap_or(2);

    println!(
        "Warmup runs: {}, Benchmark runs: {}, Threads: {}",
        warmup_runs, runs, max_thread
    );

    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(max_thread)
        .build()?;

    let mat_len = SAMPLE_SIZE * SAMPLE_SIZE;
    let mut buf_result = vec![0.0f64; mat_len];
    let mut buf_b_t = vec![0.0f64; mat_len];

    // Warmup phase (single-thread)
    {
        let matrix_a = generate_matrix(SAMPLE_SIZE, &thread_pool);
        let matrix_b = generate_matrix(SAMPLE_SIZE, &thread_pool);

        for _ in 0..warmup_runs {
            matrix_matrix_mul_into(
                &matrix_a,
                &matrix_b,
                SAMPLE_SIZE,
                SAMPLE_SIZE,
                SAMPLE_SIZE,
                SAMPLE_SIZE,
                &mut buf_b_t,
                &mut buf_result,
            );
            black_box(&buf_result);
        }
    }

    // Generate new matrics
    let mut matrix_vec = Vec::with_capacity(runs);
    for _ in 0..runs {
        if max_thread == 1 {
            let matrix_a = generate_matrix(SAMPLE_SIZE, &thread_pool);
            let matrix_b = generate_matrix(SAMPLE_SIZE, &thread_pool);
            matrix_vec.push((matrix_a, matrix_b));
        } else {
            let (matrix_a, matrix_b) = thread_pool.install(|| {
                rayon::join(
                    || generate_matrix(SAMPLE_SIZE, &thread_pool),
                    || generate_matrix(SAMPLE_SIZE, &thread_pool),
                )
            });
            matrix_vec.push((matrix_a, matrix_b));
        }
    }

    let mut score: Vec<(f64, f64)> = vec![(0.0, 0.0); runs];

    for (i, matrics) in matrix_vec.iter().enumerate().take(runs) {
        let start = Instant::now();
        if max_thread == 1 {
            matrix_matrix_mul_into(
                &matrics.0,
                &matrics.1,
                SAMPLE_SIZE,
                SAMPLE_SIZE,
                SAMPLE_SIZE,
                SAMPLE_SIZE,
                &mut buf_b_t,
                &mut buf_result,
            );
            black_box(&buf_result);
        } else {
            thread_pool.install(|| {
                (0..max_thread).into_par_iter().for_each(|_| {
                    let mut res = vec![0.0f64; mat_len];
                    let mut bt = vec![0.0f64; mat_len];
                    matrix_matrix_mul_into(
                        &matrics.0,
                        &matrics.1,
                        SAMPLE_SIZE,
                        SAMPLE_SIZE,
                        SAMPLE_SIZE,
                        SAMPLE_SIZE,
                        &mut bt,
                        &mut res,
                    );
                    black_box(res);
                });
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
    println!("Average GFLOPS score: {:.2}", avg_gflops);
    println!("Total time: {}min", (total_time / 60.0) as f32);

    println!("Find your CPU here: https://boinc.bakerlab.org/rosetta/cpu_list.php");

    Ok(())
}
