use rayon::ThreadPool;
use rayon::prelude::*;
use wide::{f32x16, f64x8};

#[inline(always)]
/// Dot-product with SIMD, has simd fallback thanks to [`wide`](https://docs.rs/wide/latest/wide/)
/// Returns value in `[-inf, inf]`
pub fn dot_product_x8(a: &[f64], b: &[f64]) -> f64 {
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

    let mut total_sum: f64 = sum.to_array().iter().sum();

    // Handle remainder
    let remainder_start = chunks * 8;
    for i in remainder_start..a.len() {
        total_sum += a[i] * b[i];
    }

    total_sum
}

#[inline(always)]
/// Dot-product with AVX-512, has fallback to [x8](dot_product_x8) if not supported
pub fn dot_product_x16(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        panic!("Vectors must not be empty");
    }
    if a.len() != b.len() {
        panic!(
            "Vector dimensions must match, a: {}, b: {}",
            a.len(),
            b.len()
        );
    }

    if is_x86_feature_detected!("avx512f") {
        let chunks = a.len() / 16;
        let mut sum = f32x16::new([0.0; 16]);

        for i in 0..chunks {
            let offset = i * 16;

            // The wide crate doesn't support f64x16,
            // so we have to load as f32 and then convert to f64
            let mut ta = [0.0f32; 16];
            let mut tb = [0.0f32; 16];

            // Load 16 elements from each vector and convert to f32
            for n in 0..16 {
                ta[n] = a[offset + n] as f32;
                tb[n] = b[offset + n] as f32;
            }

            let va = f32x16::from(ta);
            let vb = f32x16::from(tb);
            sum += va * vb;
        }
        let mut tot_sum = sum.to_array().iter().sum::<f32>() as f64;
        let remainder_start = chunks * 16;
        for i in remainder_start..a.len() {
            tot_sum += a[i] * b[i];
        }
        tot_sum
    } else {
        dot_product_x8(a, b)
    }
}

/// Multiple matrix with vec, returning the resulting vector. [BLAS 2]
/// The matrix is expected to be in row-major order and the dimensions must match,
#[inline]
#[allow(dead_code)]
pub fn matrix_vec_mul(matrix: &[f64], vector: &[f64], dim: usize, use_x16: bool) -> Vec<f64> {
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

    if use_x16 {
        for (i, out) in vec.iter_mut().enumerate() {
            let row = &matrix[i * dim..(i + 1) * dim];
            *out = dot_product_x16(row, vector);
        }
        return vec;
    }

    for (i, out) in vec.iter_mut().enumerate() {
        let row = &matrix[i * dim..(i + 1) * dim];
        *out = dot_product_x8(row, vector);
    }

    vec
}

/// Multiply two matrices using 4 simd registers at a time, fallbacks if less, returning the resulting matrix
/// The matrices are expected to be in row-major order and the dimensions must match,
#[inline(always)]
pub fn matrix_matrix_mul(
    matrix_a: &[f64],
    matrix_b: &[f64],
    rows_a: usize,
    cols_a: usize,
    rows_b: usize,
    cols_b: usize,
) -> Vec<f64> {
    if cols_a == 0 || rows_a == 0 || cols_b == 0 || rows_b == 0 {
        panic!("Dimensions must be greater than zero");
    }

    if cols_a != rows_b {
        panic!(
            "cols_a & rows_b must match for multiplication: a has {} cols, b has {} rows",
            cols_a, rows_b
        );
    }

    if matrix_a.len() != rows_a * cols_a || matrix_b.len() != rows_b * cols_b {
        panic!(
            "Dimension mismatch: matrix_a has {} elements, expected {}, matrix_b has {} elements, expected {}",
            matrix_a.len(),
            rows_a * cols_a,
            matrix_b.len(),
            rows_b * cols_b
        );
    }

    let mut result = vec![0.0f64; rows_a * cols_b];

    // Transpose matrix B to easy access pattern for dot product
    let b_t = transpose_matrix(rows_b, cols_b, matrix_b);

    let to_f64: fn(f64x8: f64x8) -> f64 = |v: f64x8| v.to_array().iter().sum();

    for row in 0..rows_a {
        let a_row = &matrix_a[row * cols_a..(row + 1) * cols_a];

        // Check how many full 4-column blocks we can process it
        let col_limit = cols_b - (cols_b % 4);

        // Step forward by 4 columns at a time
        for col in (0..col_limit).step_by(4) {
            let offset = 8;

            // Process 4 columns at a time
            let mut sum0 = f64x8::ZERO;
            let mut sum1 = f64x8::ZERO;
            let mut sum2 = f64x8::ZERO;
            let mut sum3 = f64x8::ZERO;

            let mut t = 0; // <-- Track the offset
            // Loop till remainder
            while t + offset <= cols_a {
                // Pull 8 elements from the current row of A
                let va = f64x8::from(&a_row[t..t + offset]);

                // Pull 8 elements from the columns of B (which are rows in b_t)
                let b0 = f64x8::from(&b_t[col * cols_a + t..][..offset]);
                let b1 = f64x8::from(&b_t[(col + 1) * cols_a + t..][..offset]);
                let b2 = f64x8::from(&b_t[(col + 2) * cols_a + t..][..offset]);
                let b3 = f64x8::from(&b_t[(col + 3) * cols_a + t..][..offset]);

                sum0 += va * b0;
                sum1 += va * b1;
                sum2 += va * b2;
                sum3 += va * b3;

                t += offset;
            }

            let base = row * cols_b + col;
            result[base] = to_f64(sum0);
            result[base + 1] = to_f64(sum1);
            result[base + 2] = to_f64(sum2);
            result[base + 3] = to_f64(sum3);

            // Handle leftover, using the tracker to continue from where we left off
            for lf in t..cols_a {
                let a_val = a_row[lf];
                result[base] += a_val * b_t[col * cols_a + lf];
                result[base + 1] += a_val * b_t[(col + 1) * cols_a + lf];
                result[base + 2] += a_val * b_t[(col + 2) * cols_a + lf];
                result[base + 3] += a_val * b_t[(col + 3) * cols_a + lf];
            }
        }

        // Fallback for remaining columns that don't fit into a 4-column block
        for col in col_limit..cols_b {
            let b_col = &b_t[col * cols_a..(col + 1) * cols_a];
            result[row * cols_b + col] = dot_product_x8(a_row, b_col);
        }
    }

    result
}

#[inline(always)]
/// Transpose a matrix (cols x rows) represented as a flat vector.
/// Returns the transposed matrix in row-major order.
pub fn transpose_matrix(rows: usize, cols: usize, matrix: &[f64]) -> Vec<f64> {
    if rows == 0 || cols == 0 {
        panic!("Dimension must be greater than zero");
    }

    if rows * cols != matrix.len() {
        panic!(
            "Dimension mismatch: expected {} elements, got {}",
            rows * cols,
            matrix.len()
        );
    }

    let mut transposed = vec![0.0f64; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            transposed[col * rows + row] = matrix[row * cols + col];
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
pub fn generate_vectors(
    vector_num: usize,
    dimensions: usize,
    pool: Option<&ThreadPool>,
) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0f64; dimensions]; vector_num];

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

/// Generates a random byte vector of given size, useful for testing with binary data operations
#[inline]
#[allow(dead_code)]
pub fn get_bytes(size: u32) -> Vec<u8> {
    (0..size).map(|_| fastrand::u8(..)).collect()
}

#[test]
fn test_simd() -> anyhow::Result<()> {
    use rayon::ThreadPoolBuilder;
    use std::hint::black_box;
    use std::time::Instant;

    let dim = 2048;
    let runs = 4096;

    let mut res_x8: f64 = 0.0;
    let mut res_x16: f64 = 0.0;
    let thread_pool = ThreadPoolBuilder::new().num_threads(12).build()?;

    let matrix_a = generate_matrix(dim, &thread_pool);
    let matrix_b = generate_matrix(dim, &thread_pool);

    let start_x8 = Instant::now();
    for _ in 0..runs {
        res_x8 = black_box(dot_product_x8(&matrix_a, &matrix_b));
    }
    let elapsed_x8 = start_x8.elapsed().as_secs_f64();

    let start_x16 = Instant::now();
    for _ in 0..runs {
        res_x16 = black_box(dot_product_x16(&matrix_a, &matrix_b));
    }
    let elapsed_x16 = start_x16.elapsed().as_secs_f64();

    let flops_per_dot = 2.0 * dim as f64;
    let flops_x8 = flops_per_dot / elapsed_x8;
    let flops_x16 = flops_per_dot / elapsed_x16;

    println!(
        "Time for {} runs of dot_product_x8: {:.3} seconds, FLOPS: {:.3}",
        runs, elapsed_x8, flops_x8
    );

    let is_avx512 = is_x86_feature_detected!("avx512f");

    println!(
        "Time for {} runs of dot_product_x16({}): {:.3} seconds, FLOPS: {:.3}",
        runs, is_avx512, elapsed_x16, flops_x16
    );

    assert_eq!(res_x8.round(), res_x16.round());

    Ok(())
}

#[test]
fn test_matmul() -> anyhow::Result<()> {
    use rayon::ThreadPoolBuilder;
    use std::hint::black_box;
    use std::time::Instant;

    let dim = 2048;
    let runs = 6;

    let thread_pool = ThreadPoolBuilder::new().num_threads(12).build()?;

    let matrix_a = generate_matrix(dim, &thread_pool);
    let matrix_b = generate_matrix(dim, &thread_pool);

    let start = Instant::now();
    for _ in 0..runs {
        black_box(matrix_matrix_mul(&matrix_a, &matrix_b, dim, dim, dim, dim));
    }
    let duration = start.elapsed().as_secs_f64();

    let flops_per_mul = (2.0 * (dim as f64).powi(3) - (dim as f64).powi(2)) * runs as f64;
    let gflops = flops_per_mul / duration / 1e9;

    println!(
        "Time for {} runs of {}x{} matmul: {:.3} seconds, GFLOPS: {:.3}",
        runs, dim, dim, duration, gflops
    );

    assert_eq!(matrix_a.len(), dim * dim);
    assert_eq!(matrix_b.len(), dim * dim);

    Ok(())
}

#[test]
fn test_thread_gen() {
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
    let exec_1 = generate_matrix(dim, &pool_1);
    let elapsed_1 = start_1.elapsed();

    black_box(exec_1); // Prevent compiler from optimizing away the result
    drop(pool_1); // Explicitly drop the thread pool to free resources before the next test

    let thread_2 = 1;
    let pool_2 = ThreadPoolBuilder::new()
        .num_threads(thread_2)
        .build()
        .unwrap();

    let start_2 = Instant::now();
    let exec_2 = generate_matrix(dim, &pool_2);
    let elapsed_2 = start_2.elapsed();

    black_box(exec_2);
    drop(pool_2);

    println!("Generated {}x{} matrix", dim, dim);
    println!("Time with {} threads: {:?}", thread_1, elapsed_1);
    println!("Time with {} thread: {:?}", thread_2, elapsed_2);

    assert!(elapsed_1 < elapsed_2);
}
