use std::ptr::read_unaligned;
use std::slice::from_raw_parts;
use wide::{f32x16, f64x8};

#[inline(always)]
/// Dot-product with SIMD, has simd fallback thanks to [`wide`](https://docs.rs/wide/latest/wide/)
/// Returns value in `(-inf, inf)`
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

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        unsafe {
            let offset = i * 8;

            let va_ptr = a_ptr.add(offset);
            let ba_ptr = b_ptr.add(offset);

            let va = f64x8::from(from_raw_parts(va_ptr, 8));
            let vb = f64x8::from(from_raw_parts(ba_ptr, 8));
            sum = va.mul_add(vb, sum);
        }
    }

    let mut total_sum = from_f64x8(sum);

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
            sum = va.mul_add(vb, sum);
        }

        let mut total = 0.0f64;

        for &n in &sum.to_array() {
            total += n as f64;
        }

        let remainder_start = chunks * 16;
        for i in remainder_start..a.len() {
            total += a[i] * b[i];
        }
        total
    } else {
        dot_product_x8(a, b)
    }
}

/// Matrix-vector multiplication, multiplies a matrix (rows x cols) with a vector of dimension `dim`, returning the resulting vector of size `rows`.
/// The matrix is expected to be in row-major order and the dimensions must match, otherwise it will panic.
#[inline(always)]
pub fn gemv(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    vector: &[f64],
    dim: usize,
    use_x16: bool,
) -> Vec<f64> {
    // Result of matrix-vector multiplication is a vector of length `rows`.
    let mut out = vec![0.0f64; rows];

    gemv_into(matrix, rows, cols, vector, dim, &mut out, use_x16);
    out
}

/// Matrix-vector multiplication, similar to [`gemv`](gemv), but the result is stored in the provided `res` slice, which must be pre-allocated to the correct size (rows).
/// The matrix is expected to be in row-major order and the dimensions must match, otherwise it will panic.
/// The `use_x16` flag is for AVX-512 compatibility
#[inline(always)]
pub fn gemv_into(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    vector: &[f64],
    dim: usize,
    res: &mut [f64],
    use_x16: bool,
) {
    if dim == 0 {
        panic!("Dimension must be greater than zero");
    }

    if cols != dim {
        panic!(
            "Matrix columns must match provided dimension, got cols: {}, dimension: {}",
            cols, dim
        );
    }

    if vector.len() != dim {
        panic!(
            "Vector length must equal dimension, got vector length: {}, dimension: {}",
            vector.len(),
            dim
        );
    }

    if res.len() != rows {
        panic!(
            "Result buffer length must equal number of rows: {} != {}",
            res.len(),
            rows
        );
    }

    let is_x16 = use_x16 && is_x86_feature_detected!("avx512f");

    // Process 4 rows at a time and use 8-wide across the columns.
    let width = 8;
    let row_limit = rows - (rows % 4);

    for i in (0..row_limit).step_by(4) {
        let strt0 = cols * i;
        let strt1 = cols * (i + 1);
        let strt2 = cols * (i + 2);
        let strt3 = cols * (i + 3);

        // let row0 = &matrix[i * cols..i * cols + cols];
        // let row1 = &matrix[(i + 1) * cols..(i + 1) * cols + cols];
        // let row2 = &matrix[(i + 2) * cols..(i + 2) * cols + cols];
        // let row3 = &matrix[(i + 3) * cols..(i + 3) * cols + cols];

        let mut sum0 = f64x8::ZERO;
        let mut sum1 = f64x8::ZERO;
        let mut sum2 = f64x8::ZERO;
        let mut sum3 = f64x8::ZERO;

        let mut k = 0;
        while k + width <= cols {
            let v = f64x8::from(&vector[k..k + width]);

            unsafe {
                let p0 = matrix.as_ptr().add(strt0 + k);
                let p1 = matrix.as_ptr().add(strt1 + k);
                let p2 = matrix.as_ptr().add(strt2 + k);
                let p3 = matrix.as_ptr().add(strt3 + k);

                let a0 = f64x8::from(read_unaligned(p0 as *const [f64; 8]));
                let a1 = f64x8::from(read_unaligned(p1 as *const [f64; 8]));
                let a2 = f64x8::from(read_unaligned(p2 as *const [f64; 8]));
                let a3 = f64x8::from(read_unaligned(p3 as *const [f64; 8]));

                sum0 = a0.mul_add(v, sum0);
                sum1 = a1.mul_add(v, sum1);
                sum2 = a2.mul_add(v, sum2);
                sum3 = a3.mul_add(v, sum3);
            }
            k += width;
        }

        let mut r0 = from_f64x8(sum0);
        let mut r1 = from_f64x8(sum1);
        let mut r2 = from_f64x8(sum2);
        let mut r3 = from_f64x8(sum3);

        // Handle tail
        #[allow(clippy::needless_range_loop)]
        for n in k..cols {
            let v = vector[n];
            unsafe {
                r0 += *matrix.as_ptr().add(strt0 + n) * v;
                r1 += *matrix.as_ptr().add(strt1 + n) * v;
                r2 += *matrix.as_ptr().add(strt2 + n) * v;
                r3 += *matrix.as_ptr().add(strt3 + n) * v;
            }
        }

        res[i] = r0;
        res[i + 1] = r1;
        res[i + 2] = r2;
        res[i + 3] = r3;
    }

    // Handle leftover rows that don't fit into a 4-row block
    for i in row_limit..rows {
        let row = &matrix[i * cols..(i + 1) * cols];

        res[i] = if is_x16 {
            dot_product_x16(row, vector)
        } else {
            dot_product_x8(row, vector)
        };
    }
}

#[inline(always)]
/// Multiply two matrices using 4 simd registers at a time, fallbacks if less, returning the resulting matrix
/// The matrices are expected to be in row-major order and the dimensions must match
pub fn gemm(
    matrix_a: &[f64],
    matrix_b: &[f64],
    rows_a: usize,
    cols_a: usize,
    rows_b: usize,
    cols_b: usize,
) -> Vec<f64> {
    let mut result = vec![0.0f64; rows_a * cols_b];
    let mut b_t = vec![0.0f64; rows_b * cols_b];
    gemm_into(
        matrix_a,
        matrix_b,
        rows_a,
        cols_a,
        rows_b,
        cols_b,
        &mut b_t,
        &mut result,
    );
    result
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
/// In-place matrix multiplication, similar to [`matmul`](gemm), but the result is stored in the provided `result` slice, which must be pre-allocated to the correct size (rows_a * cols_b).
/// The matrix is expected to be in row-major order and the dimensions must match, otherwise it will panic.
pub fn gemm_into(
    matrix_a: &[f64],
    matrix_b: &[f64],
    rows_a: usize,
    cols_a: usize,
    rows_b: usize,
    cols_b: usize,
    b_t: &mut [f64],
    result: &mut [f64],
) {
    if cols_a != rows_b {
        panic!(
            "Inner dimensions must match for multiplication, got cols_a: {}, rows_b: {}",
            cols_a, rows_b
        );
    }

    let size_a = rows_a * cols_a;
    let size_b = rows_b * cols_b;
    let size_res = rows_a * cols_b;
    if matrix_a.len() != size_a
        || matrix_b.len() != size_b
        || result.len() != size_res
        || b_t.len() != size_b
    {
        panic!(
            "Size mismatch: matrix_a {}, matrix_b {}, result {}, b_t {}, expected a {}, b {}, result {}, b_t {}",
            matrix_a.len(),
            matrix_b.len(),
            result.len(),
            b_t.len(),
            size_a,
            size_b,
            size_res,
            size_b,
        );
    }

    if result.len() != size_res {
        panic!(
            "Result buffer size mismatch: expected {}, got {}",
            size_res,
            result.len()
        );
    }

    transpose_mat_into(rows_b, cols_b, matrix_b, b_t);

    let b_ptr = b_t.as_ptr();

    for row in 0..rows_a {
        let a_row = &matrix_a[row * cols_a..(row + 1) * cols_a];

        // Check how many full 4-column blocks we can process it
        let col_limit = cols_b - (cols_b % 4);

        // Step forward by 4 columns at a time
        for col in (0..col_limit).step_by(4) {
            let offset = 8;
            let mut sum0 = f64x8::ZERO;
            let mut sum1 = f64x8::ZERO;
            let mut sum2 = f64x8::ZERO;
            let mut sum3 = f64x8::ZERO;

            let b0_strt = col * cols_a;
            let b1_strt = (col + 1) * cols_a;
            let b2_strt = (col + 2) * cols_a;
            let b3_strt = (col + 3) * cols_a;

            let a_ptr = a_row.as_ptr();

            let mut t = 0;
            while t + offset <= cols_a {
                unsafe {
                    // Pull 8 elements from the current row of A
                    let va_ptr = a_ptr.add(t);
                    let va = f64x8::from(read_unaligned(va_ptr as *const [f64; 8]));

                    let b0_ptr = b_ptr.add(b0_strt + t);
                    let b1_ptr = b_ptr.add(b1_strt + t);
                    let b2_ptr = b_ptr.add(b2_strt + t);
                    let b3_ptr = b_ptr.add(b3_strt + t);

                    // Pull 8 elements from the columns of B (which are rows in b_t)
                    let b0 = f64x8::from(read_unaligned(b0_ptr as *const [f64; 8]));
                    let b1 = f64x8::from(read_unaligned(b1_ptr as *const [f64; 8]));
                    let b2 = f64x8::from(read_unaligned(b2_ptr as *const [f64; 8]));
                    let b3 = f64x8::from(read_unaligned(b3_ptr as *const [f64; 8]));

                    sum0 = va.mul_add(b0, sum0);
                    sum1 = va.mul_add(b1, sum1);
                    sum2 = va.mul_add(b2, sum2);
                    sum3 = va.mul_add(b3, sum3);
                }
                t += offset;
            }

            let base = row * cols_b + col;
            result[base] = from_f64x8(sum0);
            result[base + 1] = from_f64x8(sum1);
            result[base + 2] = from_f64x8(sum2);
            result[base + 3] = from_f64x8(sum3);

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
}

#[inline(always)]
/// Transpose a matrix (cols x rows) represented as a flat vector.
/// Returns the transposed matrix in row-major order.
pub fn transpose_mat(rows: usize, cols: usize, matrix: &[f64]) -> Vec<f64> {
    let mut transposed = vec![0.0f64; rows * cols];
    transpose_mat_into(rows, cols, matrix, &mut transposed);
    transposed
}

#[inline(always)]
/// In-place transpose of a matrix, the input and output slices must be the same size and the matrix is expected to be in row-major order.
/// The output will also be in row-major order but with rows and columns swapped.
pub fn transpose_mat_into(rows: usize, cols: usize, matrix: &[f64], output: &mut [f64]) {
    let len = rows * cols;
    if len != matrix.len() || len != output.len() {
        panic!(
            "size mismatch: rows={} cols={} input={} output={}",
            rows,
            cols,
            matrix.len(),
            output.len()
        );
    }
    for col in 0..cols {
        let start = col * rows;
        for row in 0..rows {
            output[start + row] = matrix[row * cols + col];
        }
    }
}

#[inline(always)]
fn from_f64x8(v: f64x8) -> f64 {
    let a = v.to_array();
    a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7]
}

#[test]
fn test_simd() -> anyhow::Result<()> {
    use crate::rand::gen_mat;
    use rayon::ThreadPoolBuilder;
    use std::hint::black_box;
    use std::time::Instant;

    let dim = 2048;
    let runs = 4096;

    let mut res_x8: f64 = 0.0;
    let mut res_x16: f64 = 0.0;
    let thread_pool = ThreadPoolBuilder::new().num_threads(12).build()?;

    let matrix_a = gen_mat(dim, dim, &thread_pool);
    let matrix_b = gen_mat(dim, dim, &thread_pool);

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
fn test_gemv() -> anyhow::Result<()> {
    use crate::rand::gen_mat;
    use crate::rand::gen_vec;
    use rayon::ThreadPoolBuilder;
    use std::hint::black_box;
    use std::time::Instant;

    let dim = 2048;
    let runs = 16;

    let thread_pool = ThreadPoolBuilder::new().num_threads(12).build()?;

    let matrix = gen_mat(dim, dim, &thread_pool);
    let vec = gen_vec(1, dim, Some(&thread_pool));

    let start = Instant::now();
    for _ in 0..runs {
        black_box(gemv(&matrix, dim, dim, &vec[0], dim, true));
    }
    let duration = start.elapsed().as_secs_f64();

    let flops_per_mul = 2.0 * dim.pow(2) as f64;
    let gflops = flops_per_mul / duration / 1e9;

    println!(
        "Time for {} runs of {}x{} gemv: {:.3} seconds, GFLOPS: {:.3}",
        runs, dim, dim, duration, gflops
    );

    let matrix = gen_mat(dim, dim, &thread_pool);
    let vec = gen_vec(1, dim, Some(&thread_pool));

    let mut out = vec![0.0f64; dim];

    let start = Instant::now();
    for _ in 0..runs {
        gemv_into(&matrix, dim, dim, &vec[0], dim, &mut out, true);
        black_box(());
    }
    let duration = start.elapsed().as_secs_f64();

    let flops_per_mul = 2.0 * dim.pow(2) as f64;
    let gflops = flops_per_mul / duration / 1e9;

    println!(
        "Time for {} runs of {}x{} gemv(in-place): {:.3} seconds, GFLOPS: {:.3}",
        runs, dim, dim, duration, gflops
    );

    assert_eq!(out.len(), dim);

    Ok(())
}

#[test]
fn test_gemm() -> anyhow::Result<()> {
    use crate::rand::gen_mat;
    use rayon::ThreadPoolBuilder;
    use std::hint::black_box;
    use std::time::Instant;

    let dim = 2048;
    let runs = 8;

    let thread_pool = ThreadPoolBuilder::new().num_threads(12).build()?;

    let matrix_a = gen_mat(dim, dim, &thread_pool);
    let matrix_b = gen_mat(dim, dim, &thread_pool);

    let start = Instant::now();
    for _ in 0..runs {
        black_box(gemm(&matrix_a, &matrix_b, dim, dim, dim, dim));
    }
    let duration = start.elapsed().as_secs_f64();

    let flops_per_mul = (2.0 * (dim as f64).powi(3) - (dim as f64).powi(2)) * runs as f64;
    let gflops = flops_per_mul / duration / 1e9;

    println!(
        "Time for {} runs of {}x{} gemm: {:.3} seconds, GFLOPS: {:.3}",
        runs, dim, dim, duration, gflops
    );

    let mut buf_result = vec![0.0f64; dim * dim];
    let mut buf_b_t = vec![0.0f64; dim * dim];

    let start = Instant::now();
    for _ in 0..runs {
        gemm_into(
            &matrix_a,
            &matrix_b,
            dim,
            dim,
            dim,
            dim,
            &mut buf_b_t,
            &mut buf_result,
        );
        black_box(());
    }
    let duration = start.elapsed().as_secs_f64();

    let flops_per_mul = (2.0 * (dim as f64).powi(3) - (dim as f64).powi(2)) * runs as f64;
    let gflops = flops_per_mul / duration / 1e9;

    println!(
        "Time for {} runs of {}x{} gemm(in-place): {:.3} seconds, GFLOPS: {:.3}",
        runs, dim, dim, duration, gflops
    );

    assert_eq!(matrix_a.len(), dim * dim);
    assert_eq!(matrix_b.len(), dim * dim);

    Ok(())
}

#[test]
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
