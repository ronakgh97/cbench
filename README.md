**cbench** is a microbenchmark tool for x86 CPU,
performs a series of random maths computation (blas/crypto/pi) and scores in terms of GFLOPS & meaningful metrics

```shell
cbench run --runs 8 --warmups 3 --max-threads 12
Warmup runs: 3 Benchmark runs: 8 Threads: 12
Run 1: Time = 0.854s GFLOPS = 281.63
Run 2: Time = 0.908s GFLOPS = 264.81
Run 3: Time = 0.927s GFLOPS = 259.32
Run 4: Time = 0.837s GFLOPS = 287.31
Run 5: Time = 0.872s GFLOPS = 275.81
Run 6: Time = 0.890s GFLOPS = 270.33
Run 7: Time = 0.925s GFLOPS = 260.02                                                                                                                                   
Run 8: Time = 0.863s GFLOPS = 278.53
-------------------------------------
Average GFLOPS/core: 22.68
Total time: 7.0757966s
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