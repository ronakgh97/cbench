cbench is a microbenchmark tool for CPU & GPU via maths operations [BLAS](https://www.netlib.org/blas/) and scores in
terms of FLOPS or GFLOPS (Giga Floating Point Operations Per Second), for `matmul` its approx to ~ `2 * N^3` ops/time

```shell
cbench run -r 5 -w 2 -m 24
Running benchmarks...
Warmup runs: 2, Benchmark runs: 5, Threads: 24
Run 1: Time = 41.693, GFLOPS = 9.89
Run 2: Time = 41.749, GFLOPS = 9.88
Run 3: Time = 41.897, GFLOPS = 9.84
Run 4: Time = 42.179, GFLOPS = 9.78
Run 5: Time = 41.745, GFLOPS = 9.88
-------------------------------------
Average GFLOPS score: 9.85
Total time: 3.4876907916666666min
Checkout this page: https://boinc.bakerlab.org/rosetta/cpu_list.php
```

> NEED HELP WITH BLAS, GETTING MEMORY EXPLOSIONS 💥

ref

- https://docs.rs/rayon/latest/rayon/index.html
- https://docs.rs/rayon-core/1.13.0/rayon_core/
- https://doc.rust-lang.org/std/simd/
- https://www.netlib.org/blas/
- https://www.netlib.org/linpack/ (literal FORTRAN 💀💀💀)
- https://setiathome.berkeley.edu/cpu_list.php
- https://en.wikipedia.org/wiki/Floating_point_operations_per_second
- https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms