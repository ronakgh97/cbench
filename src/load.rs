use rayon::ThreadPool;
use rayon::prelude::*;
use wide::f64x8;

#[inline(always)]
/// Dot-product with SIMD, does have simd fallback thanks to [`wide`](https://docs.rs/wide/latest/wide/)
/// Returns value in `[-inf, inf]`
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        // Yes, I'm doing this *unnecessary* check, because I like symmetric, so stfu
        panic!("Vectors must not be empty");
    }
    if a.len() != b.len() {
        panic!(
            "Vector dimensions must match, a: {}, b: {}",
            a.len(),
            b.len()
        );
    }
    let chunks = a.len() / 8;
    let mut sum = f64x8::ZERO;

    for i in 0..chunks {
        let offset = i * 8;
        let va = f64x8::from(&a[offset..offset + 8]);
        let vb = f64x8::from(&b[offset..offset + 8]);
        sum += va * vb;
    }

    let arr = sum.to_array();
    let mut total: f64 = arr.iter().sum();

    // Handle remainder
    let remainder_start = chunks * 8;
    for i in remainder_start..a.len() {
        total += a[i] * b[i];
    }

    total
}

/// Multiple matrix with vec, returning the resulting vector. [BLAS 2]
/// The matrix is expected to be in row-major order and the dimensions must match,
#[inline(always)]
#[allow(dead_code)]
pub fn matrix_vec_mul(matrix: &[f64], vector: &[f64], dim: usize) -> Vec<f64> {
    if dim == 0 {
        panic!("Dimension must be greater than zero");
    }
    if vector.len() != dim || matrix.len() != dim * dim {
        panic!(
            "Dimension mismatch: matrix has {} rows, vector has {} elements, expected dimension {}",
            matrix.len() / dim,
            vector.len(),
            dim
        );
    }

    let mut vec = vec![0.0f64; dim];
    for (i, out) in vec.iter_mut().enumerate() {
        let row = &matrix[i * dim..(i + 1) * dim];
        *out = dot_product(row, vector);
    }

    vec
}

/// Multiply two matrices (dim x dim), returning the resulting matrix. [BLAS 3]
/// Both matrices are expected to be in row-major order.
#[inline(always)]
pub fn matrix_matrix_mul(matrix_a: &[f64], matrix_b: &[f64], dim: usize) -> Vec<f64> {
    if dim == 0 {
        panic!("Dimension must be greater than zero");
    }
    if matrix_a.len() != dim * dim || matrix_b.len() != dim * dim {
        panic!(
            "Dimension mismatch: expected {} elements, a has {}, b has {}",
            dim * dim,
            matrix_a.len(),
            matrix_b.len()
        );
    }

    let mut result = vec![0.0f64; dim * dim];

    // Transpose matrix B to easy access pattern for dot product
    let mut b_t = vec![0.0f64; dim * dim];
    unsafe {
        b_t.set_len(dim * dim);
    }
    for col in 0..dim {
        for row in 0..dim {
            b_t[col * dim + row] = matrix_b[row * dim + col];
        }
    }

    for row in 0..dim {
        let a_row = &matrix_a[row * dim..(row + 1) * dim];
        for col in 0..dim {
            let b_col = &b_t[col * dim..(col + 1) * dim];
            result[row * dim + col] = dot_product(a_row, b_col);
        }
    }

    result
}

#[inline(always)]
#[allow(dead_code)]
/// Transpose a square matrix (dim x dim) represented as a flat vector. Returns the transposed matrix in row-major order.
pub fn transpose_matrix(matrix: &[f64]) -> Vec<f64> {
    let dim = (matrix.len() as f64).sqrt() as usize;
    if dim * dim != matrix.len() {
        panic!("Input must be a square matrix");
    }

    let mut transposed = vec![0.0f64; matrix.len()];
    for row in 0..dim {
        for col in 0..dim {
            transposed[col * dim + row] = matrix[row * dim + col];
        }
    }
    transposed
}

/// Generates a flat random square matrix of given dimension with values in range `[-1.0, 1.0]`
#[inline]
pub fn generate_matrix(dim: usize, pool: &ThreadPool) -> Vec<f64> {
    if dim == 0 {
        panic!("Dimension must be greater than zero");
    }

    let mut matrix = vec![0.0f64; dim * dim];

    pool.install(|| {
        matrix.par_iter_mut().for_each(|x| {
            *x = fastrand::f64() * 2.0 - 1.0;
        });
    });

    matrix
}

/// Generates a random vector of given dimension with values in range `[-1.0, 1.0]`
#[inline]
#[allow(dead_code)]
pub fn generate_vectors(vector_num: usize, dimensions: usize, pool: &ThreadPool) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0f64; dimensions]; vector_num];

    pool.install(|| {
        result.par_iter_mut().for_each(|v| {
            for x in v {
                *x = fastrand::f64() * 2.0 - 1.0;
            }
        });
    });

    result
}

/// Generates a random byte vector of given size, useful for testing with binary data or metadata.
#[inline]
#[allow(dead_code)]
pub fn get_bytes(size: u32) -> Vec<u8> {
    (0..size).map(|_| fastrand::u8(..)).collect()
}

#[test]
fn test_thread_gen() {
    use std::hint::black_box;
    use std::time::Instant;
    let num_vectors = 1024 * 1024;
    let dimensions = 128;

    let thread_1 = std::thread::available_parallelism()
        .unwrap_or(std::num::NonZeroUsize::new(1).unwrap())
        .get();
    let pool_1 = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_1)
        .build()
        .unwrap();

    let start_1 = Instant::now();
    let exec_1 = generate_vectors(num_vectors, dimensions, &pool_1);
    let elapsed_1 = start_1.elapsed();
    black_box(exec_1); // Prevent compiler from optimizing away the result

    drop(pool_1); // Explicitly drop the thread pool to free resources before the next test

    let thread_2 = 1;
    let pool_2 = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_2)
        .build()
        .unwrap();

    let start_2 = Instant::now();
    let exec_2 = generate_vectors(num_vectors, dimensions, &pool_2);
    let elapsed_2 = start_2.elapsed();
    black_box(exec_2);

    drop(pool_2);

    println!(
        "Generated {} vectors of dimension {} in {:?} ({} threads) vs {:?} ({} thread)",
        num_vectors, dimensions, elapsed_1, thread_1, elapsed_2, thread_2
    );

    assert!(elapsed_1 < elapsed_2);
}
