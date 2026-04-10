cbench is a microbenchmark for testing the performance of CPU via maths operations [
BLAS](https://www.netlib.org/blas/) and scores in terms of FLOPS or GFLOPS (Giga
Floating Point Operations Per Second), for `matmul` its approx to ~ `2 * N^3` ops/time

ref

- https://docs.rs/rayon/latest/rayon/index.html
- https://docs.rs/rayon-core/1.13.0/rayon_core/
- https://doc.rust-lang.org/std/simd/
- https://www.netlib.org/blas/
- https://www.netlib.org/linpack/ (literal FORTRAN 💀💀💀)
- https://setiathome.berkeley.edu/cpu_list.php
- https://en.wikipedia.org/wiki/Floating_point_operations_per_second
- https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms