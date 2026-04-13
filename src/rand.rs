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

/// Fills an existing buffer with random values in range `[-1.0, 1.0]`
#[inline]
pub fn gen_fill(buf: &mut [f64], pool: &ThreadPool) {
    pool.install(|| {
        buf.par_iter_mut().for_each(|x| {
            *x = fastrand::f64() * 2.0 - 1.0;
        });
    });
}

/// Generates a random byte vector of given size, useful for testing with binary data operations
#[inline]
pub fn get_bytes(size: u32) -> Vec<u8> {
    (0..size).map(|_| fastrand::u8(..)).collect()
}
