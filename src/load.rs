use std::ptr::read_unaligned;
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

            let va = f64x8::from(read_unaligned(va_ptr as *const [f64; 8]));
            let vb = f64x8::from(read_unaligned(ba_ptr as *const [f64; 8]));
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

/// Matrix-vector multiplication using 6 simd register at time, multiplies a matrix (rows x cols) with a vector of dimension `dim`, returning the resulting vector of size `rows`.
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

/// Matrix-vector multiplication, but the result is stored in the provided `res` slice, which must be pre-allocated to the correct size (rows).
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

    res.fill(0.0f64);

    let is_x16 = use_x16 && is_x86_feature_detected!("avx512f");

    // Process 4 rows at a time and use 8-wide across the columns.
    let width = 8;
    let row_limit = rows - (rows % 6);

    for i in (0..row_limit).step_by(6) {
        let strt0 = cols * i;
        let strt1 = cols * (i + 1);
        let strt2 = cols * (i + 2);
        let strt3 = cols * (i + 3);
        let strt4 = cols * (i + 4);
        let strt5 = cols * (i + 5);

        let mut sum0 = f64x8::ZERO;
        let mut sum1 = f64x8::ZERO;
        let mut sum2 = f64x8::ZERO;
        let mut sum3 = f64x8::ZERO;
        let mut sum4 = f64x8::ZERO;
        let mut sum5 = f64x8::ZERO;

        let mut k = 0;
        while k + width <= cols {
            {
                let idx0 = strt0.checked_add(k).and_then(|v| v.checked_add(48));
                unsafe {
                    // _mm_prefetch instruction is safe to execute on any pointer.
                    if let Some(idx0) = idx0 {
                        core::arch::x86_64::_mm_prefetch(
                            matrix.as_ptr().add(idx0) as *const i8,
                            core::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
            }

            let v =
                unsafe { f64x8::from(read_unaligned(vector.as_ptr().add(k) as *const [f64; 8])) };

            unsafe {
                let p0 = matrix.as_ptr().add(strt0 + k);
                let p1 = matrix.as_ptr().add(strt1 + k);
                let p2 = matrix.as_ptr().add(strt2 + k);
                let p3 = matrix.as_ptr().add(strt3 + k);
                let p4 = matrix.as_ptr().add(strt4 + k);
                let p5 = matrix.as_ptr().add(strt5 + k);

                let a0 = f64x8::from(read_unaligned(p0 as *const [f64; 8]));
                let a1 = f64x8::from(read_unaligned(p1 as *const [f64; 8]));
                let a2 = f64x8::from(read_unaligned(p2 as *const [f64; 8]));
                let a3 = f64x8::from(read_unaligned(p3 as *const [f64; 8]));
                let a4 = f64x8::from(read_unaligned(p4 as *const [f64; 8]));
                let a5 = f64x8::from(read_unaligned(p5 as *const [f64; 8]));

                sum0 = a0.mul_add(v, sum0);
                sum1 = a1.mul_add(v, sum1);
                sum2 = a2.mul_add(v, sum2);
                sum3 = a3.mul_add(v, sum3);
                sum4 = a4.mul_add(v, sum4);
                sum5 = a5.mul_add(v, sum5);
            }
            k += width;
        }

        let mut r0 = from_f64x8(sum0);
        let mut r1 = from_f64x8(sum1);
        let mut r2 = from_f64x8(sum2);
        let mut r3 = from_f64x8(sum3);
        let mut r4 = from_f64x8(sum4);
        let mut r5 = from_f64x8(sum5);

        // Handle tail
        #[allow(clippy::needless_range_loop)]
        for n in k..cols {
            let v = vector[n];
            unsafe {
                r0 += *matrix.as_ptr().add(strt0 + n) * v;
                r1 += *matrix.as_ptr().add(strt1 + n) * v;
                r2 += *matrix.as_ptr().add(strt2 + n) * v;
                r3 += *matrix.as_ptr().add(strt3 + n) * v;
                r4 += *matrix.as_ptr().add(strt4 + n) * v;
                r5 += *matrix.as_ptr().add(strt5 + n) * v;
            }
        }

        res[i] = r0;
        res[i + 1] = r1;
        res[i + 2] = r2;
        res[i + 3] = r3;
        res[i + 4] = r4;
        res[i + 5] = r5;
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
/// In-place matrix multiplication, but the result is stored in the provided `result` slice, which must be pre-allocated and zeroed to the correct size (rows_a * cols_b).
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
    result.fill(0.0f64);

    // Number of rows of A to process in one block
    let block_a = 256;
    // Number of columns of B to process in one block
    let block_b = 128;
    // Number of elements in the inner dimension to process in one block,
    // This should be tuned and large enough to amortize the overhead but small enough to fit in cache
    let block_c = 64;

    let b_ptr = b_t.as_ptr();

    for i in (0..rows_a).step_by(block_a) {
        for j in (0..cols_b).step_by(block_b) {
            for k in (0..cols_a).step_by(block_c) {
                // Clamp to matrix dimensions
                let i_max = (i + block_a).min(rows_a);
                let j_max = (j + block_b).min(cols_b);
                let k_max = (k + block_c).min(cols_a);

                //TODO: Try to fetch the next block of A and B into L1 cache, this is a bit tricky because we have to calculate the correct offsets

                // Process the block of A rows against the block of B columns
                for row in i..i_max {
                    // Get the current row of A, we will use it across the block of B columns
                    let a_row = &matrix_a[row * cols_a..(row + 1) * cols_a];
                    let a_ptr = a_row.as_ptr();

                    // Check how many full 4-column blocks we can process it
                    // (j_max - j) is the number of columns in this block,
                    // we want to round it down to the nearest multiple of 4
                    let col_limit = j_max - ((j_max - j) % 4);

                    // Step forward by 4 columns at a time
                    for col in (j..col_limit).step_by(4) {
                        let offset = 8;
                        let mut sum0 = f64x8::ZERO;
                        let mut sum1 = f64x8::ZERO;
                        let mut sum2 = f64x8::ZERO;
                        let mut sum3 = f64x8::ZERO;

                        let b0_strt = col * cols_a;
                        let b1_strt = (col + 1) * cols_a;
                        let b2_strt = (col + 2) * cols_a;
                        let b3_strt = (col + 3) * cols_a;

                        let mut t = k;
                        while t + offset <= k_max {
                            // t + 8 <= k_max <= cols_a, so we are within bounds of a_row and b_t
                            // We use read_unaligned since pointers might not be aligned to 64 bytes.
                            unsafe {
                                // Pull 8 elements from the current row of A
                                // Slice a_row has length cols_a, so a_ptr.add(t) is valid.
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
                        let mut r0 = from_f64x8(sum0);
                        let mut r1 = from_f64x8(sum1);
                        let mut r2 = from_f64x8(sum2);
                        let mut r3 = from_f64x8(sum3);

                        // Handle leftovers
                        for lf in t..k_max {
                            // lf < k_max <= cols_a, which is checked at start
                            unsafe {
                                let a_val = *a_ptr.add(lf);
                                r0 += a_val * *b_ptr.add(b0_strt + lf);
                                r1 += a_val * *b_ptr.add(b1_strt + lf);
                                r2 += a_val * *b_ptr.add(b2_strt + lf);
                                r3 += a_val * *b_ptr.add(b3_strt + lf);
                            }
                        }

                        // Accumulate back to result buffer only once (avoids double memory access overhead)
                        unsafe {
                            let res_ptr = result.as_mut_ptr().add(base);
                            *res_ptr += r0;
                            *res_ptr.add(1) += r1;
                            *res_ptr.add(2) += r2;
                            *res_ptr.add(3) += r3;
                        }
                    }

                    // Handle remaining columns that don't fit into a 4-column block
                    for col in col_limit..j_max {
                        let wide = 8;
                        let mut sum = f64x8::ZERO;
                        let b_base = col * cols_a;

                        let mut t = k;

                        while t + wide <= k_max {
                            unsafe {
                                let va_ptr = a_ptr.add(t);
                                let vb_ptr = b_ptr.add(b_base + t);

                                let va = f64x8::from(read_unaligned(va_ptr as *const [f64; 8]));
                                let vb = f64x8::from(read_unaligned(vb_ptr as *const [f64; 8]));

                                sum = va.mul_add(vb, sum);
                            }
                            t += wide;
                        }

                        let mut sum_f64 = from_f64x8(sum);

                        // Handle tail for this column
                        for t in t..k_max {
                            sum_f64 += a_row[t] * b_t[b_base + t];
                        }

                        result[row * cols_b + col] += sum_f64;
                    }
                }
            }
        }
    }

    // NOTE: Previous version
    // for row in 0..rows_a {
    //     let a_row = &matrix_a[row * cols_a..(row + 1) * cols_a];
    //     let a_ptr = a_row.as_ptr();
    //
    //     // Check how many full 4-column blocks we can process it
    //     let col_limit = cols_b - (cols_b % 4);
    //
    //     // Step forward by 4 columns at a time
    //     for col in (0..col_limit).step_by(4) {
    //         let offset = 8;
    //         let mut sum0 = f64x8::ZERO;
    //         let mut sum1 = f64x8::ZERO;
    //         let mut sum2 = f64x8::ZERO;
    //         let mut sum3 = f64x8::ZERO;
    //
    //         let b0_strt = col * cols_a;
    //         let b1_strt = (col + 1) * cols_a;
    //         let b2_strt = (col + 2) * cols_a;
    //         let b3_strt = (col + 3) * cols_a;
    //
    //         let mut t = 0;
    //         while t + offset <= cols_a {
    //             unsafe {
    //                 // Pull 8 elements from the current row of A
    //                 let va_ptr = a_ptr.add(t);
    //                 let va = f64x8::from(read_unaligned(va_ptr as *const [f64; 8]));
    //
    //                 let b0_ptr = b_ptr.add(b0_strt + t);
    //                 let b1_ptr = b_ptr.add(b1_strt + t);
    //                 let b2_ptr = b_ptr.add(b2_strt + t);
    //                 let b3_ptr = b_ptr.add(b3_strt + t);
    //
    //                 // Pull 8 elements from the columns of B (which are rows in b_t)
    //                 let b0 = f64x8::from(read_unaligned(b0_ptr as *const [f64; 8]));
    //                 let b1 = f64x8::from(read_unaligned(b1_ptr as *const [f64; 8]));
    //                 let b2 = f64x8::from(read_unaligned(b2_ptr as *const [f64; 8]));
    //                 let b3 = f64x8::from(read_unaligned(b3_ptr as *const [f64; 8]));
    //
    //                 sum0 = va.mul_add(b0, sum0);
    //                 sum1 = va.mul_add(b1, sum1);
    //                 sum2 = va.mul_add(b2, sum2);
    //                 sum3 = va.mul_add(b3, sum3);
    //             }
    //             t += offset;
    //         }
    //
    //         let base = row * cols_b + col;
    //         result[base] = from_f64x8(sum0);
    //         result[base + 1] = from_f64x8(sum1);
    //         result[base + 2] = from_f64x8(sum2);
    //         result[base + 3] = from_f64x8(sum3);
    //
    //         // Handle leftover, using the tracker to continue from where we left off
    //         for lf in t..cols_a {
    //             let a_val = a_row[lf];
    //             result[base] += a_val * b_t[b0_strt + lf];
    //             result[base + 1] += a_val * b_t[b1_strt + lf];
    //             result[base + 2] += a_val * b_t[b2_strt + lf];
    //             result[base + 3] += a_val * b_t[b3_strt + lf];
    //         }
    //     }
    //
    //     // Fallback for remaining columns that don't fit into a 4-column block
    //     for col in col_limit..cols_b {
    //         let b_col = &b_t[col * cols_a..(col + 1) * cols_a];
    //         result[row * cols_b + col] = dot_product_x8(a_row, b_col);
    //     }
    // }
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
        let in_ptr = matrix.as_ptr();
        let out_ptr = output.as_mut_ptr();
        for row in 0..rows {
            unsafe {
                *out_ptr.add(start + row) = *in_ptr.add(row * cols + col);
            }
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

    let dim = 1024;
    let runs = 4096;

    let mut res_x8: f64 = 0.0;
    let mut res_x16: f64 = 0.0;
    let thread_pool = ThreadPoolBuilder::new().num_threads(12).build()?;

    let flat_a = gen_mat(dim, dim, &thread_pool);
    let flat_b = gen_mat(dim, dim, &thread_pool);

    let start_x8 = Instant::now();
    for _ in 0..runs {
        res_x8 = black_box(dot_product_x8(&flat_a, &flat_b));
    }
    let elapsed_x8 = start_x8.elapsed().as_secs_f64();

    let start_x16 = Instant::now();
    for _ in 0..runs {
        res_x16 = black_box(dot_product_x16(&flat_a, &flat_b));
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

    let rows = 4096;
    let cols = 2048;
    let runs = 6;

    let thread_pool = ThreadPoolBuilder::new().num_threads(12).build()?;

    let matrix = gen_mat(rows, cols, &thread_pool);
    let vec = gen_vec(1, cols, Some(&thread_pool));

    let start = Instant::now();
    for _ in 0..runs {
        black_box(gemv(&matrix, rows, cols, &vec[0], cols, false));
    }
    let duration = start.elapsed().as_secs_f64();

    let flops_per_mul = 2.0 * rows as f64 * cols as f64;
    let gflops = flops_per_mul / duration / 1e9;

    println!(
        "Time for {} runs of {}x{} gemv: {:.3} seconds, GFLOPS: {:.3}",
        runs, rows, cols, duration, gflops
    );

    let matrix = gen_mat(rows, cols, &thread_pool);
    let vec = gen_vec(1, cols, Some(&thread_pool));

    let mut out = vec![0.0f64; rows];

    let start = Instant::now();
    for _ in 0..runs {
        gemv_into(&matrix, rows, cols, &vec[0], cols, &mut out, false);
        black_box(());
    }
    let duration = start.elapsed().as_secs_f64();

    let flops_per_mul = 2.0 * rows as f64 * cols as f64 * runs as f64;
    let gflops = flops_per_mul / duration / 1e9;

    println!(
        "Time for {} runs of {}x{} gemv(in-place): {:.3} seconds, GFLOPS: {:.3}",
        runs, rows, cols, duration, gflops
    );

    assert_eq!(out.len(), rows);

    Ok(())
}

#[test]
fn test_gemm() -> anyhow::Result<()> {
    use crate::rand::gen_mat;
    use rayon::ThreadPoolBuilder;
    use std::hint::black_box;
    use std::time::Instant;

    let rows = 4096;
    let cols = 2048;
    let runs = 6;

    let thread_pool = ThreadPoolBuilder::new().num_threads(12).build()?;

    let matrix_a = gen_mat(rows, cols, &thread_pool);
    let matrix_b = gen_mat(rows, cols, &thread_pool);

    let start = Instant::now();
    for _ in 0..runs {
        black_box(gemm(&matrix_a, &matrix_b, rows, cols, cols, rows));
    }
    let duration = start.elapsed().as_secs_f64();

    let flops_per_mul = (2.0 * (rows * cols) as f64) * cols as f64;
    let gflops = flops_per_mul / duration / 1e9;

    println!(
        "Time for {} runs of {}x{} gemm: {:.3} seconds, GFLOPS: {:.3}",
        runs, rows, cols, duration, gflops
    );

    let mut buf_result = vec![0.0f64; rows * rows];
    let mut buf_b_t = vec![0.0f64; cols * rows];

    let start = Instant::now();
    for _ in 0..runs {
        gemm_into(
            &matrix_a,
            &matrix_b,
            rows,
            cols,
            cols,
            rows,
            &mut buf_b_t,
            &mut buf_result,
        );
        black_box(());
    }
    let duration = start.elapsed().as_secs_f64();

    let flops_per_mul = (2.0 * (rows * cols) as f64) * cols as f64 * runs as f64;
    let gflops = flops_per_mul / duration / 1e9;

    println!(
        "Time for {} runs of {}x{} gemm(in-place): {:.3} seconds, GFLOPS: {:.3}",
        runs, rows, cols, duration, gflops
    );

    assert_eq!(matrix_a.len(), rows * cols);
    assert_eq!(matrix_b.len(), rows * cols);

    Ok(())
}
