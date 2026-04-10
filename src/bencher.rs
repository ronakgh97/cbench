use crate::load::{generate_vectors, matrix_matrix_mul};
use std::hint::black_box;
use std::time::Instant;

const SAMPLE_SIZE: usize = 2048;

pub fn run_benchmark(runs: usize, warmups: Option<usize>, max_thread: usize) -> anyhow::Result<()> {
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(max_thread)
        .build()?;

    let warmup_runs = warmups.unwrap_or(2);

    println!("Running benchmark...\n");
    println!(
        "Warmup runs: {}, Benchmark runs: {}, Max threads: {}",
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

    println!("{:>8} {:>12}", "Run", "Median GFLOPs");
    println!("{:>8} {:>12}", "---", "------------");

    Ok(())
}
