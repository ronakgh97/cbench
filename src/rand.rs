use rayon::ThreadPool;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;

/// Generates a random matrix of given dimensions with values in range `[-1.0, 1.0]`
#[inline]
pub fn gen_mat(rows: usize, col: usize, pool: &ThreadPool) -> Vec<f64> {
    if col == 0 || rows == 0 {
        panic!("Matrix dimensions must be greater than zero");
    }

    let mut matrix = vec![0.0f64; col * rows];

    pool.install(|| {
        matrix.par_iter_mut().for_each(|x| {
            *x = fastrand::f64() * 2.0 - 1.0;
        });
    });

    matrix
}

/// Generates a random vector of given dimension with values in range `[-1.0, 1.0]`
#[inline]
pub fn gen_vec(num: usize, dim: usize, pool: Option<&ThreadPool>) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0f64; dim]; num];

    if let Some(pool) = pool {
        pool.install(|| {
            result.par_iter_mut().for_each(|v| {
                for x in v {
                    *x = fastrand::f64() * 2.0 - 1.0;
                }
            });
        });

        result
    } else {
        for v in &mut result {
            for x in v {
                *x = fastrand::f64() * 2.0 - 1.0;
            }
        }
        result
    }
}

/// Fills an existing f32 buffer with random values in range `[-1.0, 1.0]`
#[inline]
pub fn gen_fill(buf: &mut [f32], pool: &ThreadPool) {
    pool.install(|| {
        buf.par_iter_mut().for_each(|x| {
            *x = fastrand::f32() * 2.0 - 1.0;
        });
    });
}

/// Generates a random byte vector of given size, useful for testing with binary data operations
#[inline]
pub fn get_bytes(size: u32) -> Vec<u8> {
    (0..size).map(|_| fastrand::u8(..)).collect()
}

#[test]
#[ignore]
fn test_thread_gen() {
    use crate::rand::gen_mat;
    use rayon::ThreadPoolBuilder;
    use std::hint::black_box;
    use std::time::Instant;

    let dim = 2048;

    let thread_1 = std::thread::available_parallelism()
        .unwrap_or(std::num::NonZeroUsize::new(1).unwrap())
        .get();
    let pool_1 = ThreadPoolBuilder::new()
        .num_threads(thread_1)
        .build()
        .unwrap();

    let start_1 = Instant::now();
    let exec_1 = gen_mat(dim, dim, &pool_1);
    let elapsed_1 = start_1.elapsed();

    black_box(exec_1); // Prevent compiler from optimizing away the result
    drop(pool_1); // Explicitly drop the thread pool to free resources before the next test

    let thread_2 = 1;
    let pool_2 = ThreadPoolBuilder::new()
        .num_threads(thread_2)
        .build()
        .unwrap();

    let start_2 = Instant::now();
    let exec_2 = gen_mat(dim, dim, &pool_2);
    let elapsed_2 = start_2.elapsed();

    black_box(exec_2);
    drop(pool_2);

    println!("Generated {}x{} matrix", dim, dim);
    println!("Time with {} threads: {:?}", thread_1, elapsed_1);
    println!("Time with {} thread: {:?}", thread_2, elapsed_2);

    assert!(elapsed_1 < elapsed_2);
}
