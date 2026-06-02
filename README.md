**cbench** is a microbenchmark tool for x86 CPU,
performs a series of random maths computation (blas/crypto/pi) and scores in terms of GFLOPS & meaningful metrics

```shell
cbench run --runs 8 --warmups 3 --max-threads 18
Warmup runs: 3 Benchmark runs: 8 Threads: 18

Running BLAS bench...
Run 1: Time = 1.293s GFLOPS = 279.03
Run 2: Time = 1.275s GFLOPS = 282.85
Run 3: Time = 1.287s GFLOPS = 280.33
Run 4: Time = 1.254s GFLOPS = 287.75
Run 5: Time = 1.233s GFLOPS = 292.44
Run 6: Time = 1.258s GFLOPS = 286.76
Run 7: Time = 1.295s GFLOPS = 278.54
Run 8: Time = 1.261s GFLOPS = 286.09
-------------------------------------
Estimated CPU Score: 2564
Average GFLOPS/core: 15.79
Total time: 10.155198s
Find your CPU here: https://boinc.bakerlab.org/rosetta/cpu_list.php
```

ref

- https://docs.rs/rayon/latest/rayon/index.html
- https://docs.rs/rayon-core/1.13.0/rayon_core/
- https://docs.rs/wgpu/latest/wgpu/index.html
- https://doc.rust-lang.org/std/simd/
- https://www.netlib.org/blas/
- https://www.netlib.org/benchmark/whetstone.c
- https://setiathome.berkeley.edu/cpu_list.php
- https://en.wikipedia.org/wiki/Floating_point_operations_per_second
- https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms