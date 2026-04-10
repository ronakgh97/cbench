use rayon::prelude::*;
use wide::f64x8;

#[inline]
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

/// Multiplies a matrix (flattened) with a vector, returning the resulting vector.
/// The matrix is expected to be in row-major order and the dimensions must match,
#[inline]
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

/// Generates a random vector of given dimension with values in range `[-1.0, 1.0]`
#[inline]
pub fn generate_vectors(vector_num: usize, dimensions: usize, thread_num: usize) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0f64; dimensions]; vector_num];

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_num)
        .build()
        .unwrap();

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
fn test_parallel() {
    use std::hint::black_box;
    use std::time::Instant;
    let num_vectors = 1024 * 1024;
    let dimensions = 512;

    let start_1 = Instant::now();
    let exec_1 = generate_vectors(num_vectors, dimensions, 24);
    let elapsed_1 = start_1.elapsed();
    black_box(exec_1); // Prevent compiler from optimizing away the result

    let start_2 = Instant::now();
    let exec_2 = generate_vectors(num_vectors, dimensions, 1);
    let elapsed_2 = start_2.elapsed();
    black_box(exec_2);

    println!(
        "Generated {} vectors of dimension {} in {:?} (24 threads) vs {:?} (1 threads)",
        num_vectors, dimensions, elapsed_1, elapsed_2
    );
    assert!(elapsed_1 < elapsed_2);
}
