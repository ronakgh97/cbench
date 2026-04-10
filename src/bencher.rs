use crate::load::{generate_vectors, matrix_matrix_mul};
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

// TODO: Random generation are pain and should consider inside bench loop, Need better scoring
// TODO: Currently this runs matmul on given threads, and compute `flops` * thread_num,
//  but I think we should parallelize matmul computation internally and fix the flops calculation....not run them in each threads?
//  so it can count as single matmul kernel efficiency, not System thread scheduling efficiency, which is not what we want to measure.

const SAMPLE_SIZE: usize = 2048;

pub fn run_benchmark(runs: usize, warmups: Option<usize>, max_thread: usize) -> anyhow::Result<()> {
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(max_thread)
        .build()?;

    let warmup_runs = warmups.unwrap_or(2);

    println!("Running benchmarks...");
    println!(
        "Warmup runs: {}, Benchmark runs: {}, Threads: {}",
        warmup_runs, runs, max_thread
    );

    // Warmup phase
    {
        let matrix_a = generate_vectors(SAMPLE_SIZE, SAMPLE_SIZE, &thread_pool);
        let matrix_b = generate_vectors(SAMPLE_SIZE, SAMPLE_SIZE, &thread_pool);

        let flat_a: Vec<f64> = matrix_a.into_iter().flatten().collect();
        let flat_b: Vec<f64> = matrix_b.into_iter().flatten().collect();

        for _ in 0..warmup_runs {
            let _ = matrix_matrix_mul(&flat_a, &flat_b, SAMPLE_SIZE);
        }
    }

    // Generate new matrics
    let mut matrix_vec = Vec::with_capacity(runs);

    for _ in 0..runs {
        let matrix_a = generate_vectors(SAMPLE_SIZE, SAMPLE_SIZE, &thread_pool);
        let matrix_b = generate_vectors(SAMPLE_SIZE, SAMPLE_SIZE, &thread_pool);
        let flat_a: Vec<f64> = matrix_a.into_iter().flatten().collect();
        let flat_b: Vec<f64> = matrix_b.into_iter().flatten().collect();

        matrix_vec.push((flat_a, flat_b));
    }

    let mut score = Vec::with_capacity(runs);

    for (i, matrics) in matrix_vec.iter().enumerate().take(runs) {
        let start = Instant::now();
        if max_thread == 1 {
            let _ = black_box(matrix_matrix_mul(&matrics.0, &matrics.1, SAMPLE_SIZE));
        } else {
            thread_pool.install(|| {
                (0..max_thread).into_par_iter().for_each(|_| {
                    let _ = black_box(matrix_matrix_mul(&matrics.0, &matrics.1, SAMPLE_SIZE));
                });
            });
        }
        let elapsed = start.elapsed().as_secs_f64();

        let flops_per_mul = 2.0 * (SAMPLE_SIZE as f64).powi(3);
        let total_flops = flops_per_mul * (max_thread as f64);
        let gflops = total_flops / elapsed / 1e9;

        score.push((elapsed, gflops));

        println!(
            "Run {}: Time = {:.3?}, GFLOPS = {:.2}",
            i + 1,
            elapsed,
            gflops
        );
    }

    println!("-------------------------------------");
    let avg_gflops = score.iter().map(|s| s.1).sum::<f64>() / (score.len() as f64);
    let total_time = score.iter().map(|s| s.0).sum::<f64>();
    println!("Average GFLOPS score: {:.2}", avg_gflops);
    println!("Total time: {}min", total_time / 60.0);

    println!("Checkout this page: https://boinc.bakerlab.org/rosetta/cpu_list.php");

    Ok(())
}
