**cbench** is a microbenchmark tool for x86 CPU,
performs a series of random maths computation (blas/crypto/pi) and scores in terms of GFLOPS & meaningful metrics

```shell
cbench run --runs 8 --warmups 3 --max-threads 18
Warmup runs: 3 Benchmark runs: 8 Threads: 18

Running BLAS bench...
Run 1: Time = 0.406s GFLOPS = 761.14
Run 2: Time = 0.403s GFLOPS = 766.86
Run 3: Time = 0.397s GFLOPS = 778.96
Run 4: Time = 0.402s GFLOPS = 768.82
Run 5: Time = 0.397s GFLOPS = 779.28
Run 6: Time = 0.398s GFLOPS = 776.11
Run 7: Time = 0.391s GFLOPS = 790.79
Run 8: Time = 0.405s GFLOPS = 763.08

Running Crypto bench...
Run 1: Time = 0.282s Cycles = 681473512
Run 2: Time = 0.283s Cycles = 685411479
Run 3: Time = 0.284s Cycles = 687970278
Run 4: Time = 0.280s Cycles = 678521233
Run 5: Time = 0.283s Cycles = 683580753
Run 6: Time = 0.283s Cycles = 685580881
Run 7: Time = 0.276s Cycles = 668620334
Run 8: Time = 0.282s Cycles = 683344140
---------------------------------------
Estimated CPU Score: 2053
Average SCORE/Core: 114.00
BLAS: 773.13 GFLOPS
Crypto: 0.13 GB/s
Total time: 5.45s
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