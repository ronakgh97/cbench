
**cbench** is a microbenchmark tool for x86 CPU,
performs a series of random maths computation (blas/crypto/pi) and scores in terms of GFLOPS & 'meaningful' metrics

```shell
cbench run --runs 12 --warmups 3 --max-threads 24
Warmup runs: 3 Benchmark runs: 12 Threads: 24

Running BLAS bench...
Run 1: Time = 0.518s GFLOPS = 795.62
Run 2: Time = 0.525s GFLOPS = 785.45
Run 3: Time = 0.539s GFLOPS = 764.43
Run 4: Time = 0.527s GFLOPS = 782.11
Run 5: Time = 0.529s GFLOPS = 778.82
Run 6: Time = 0.566s GFLOPS = 727.86
Run 7: Time = 0.667s GFLOPS = 617.94
Run 8: Time = 0.552s GFLOPS = 747.26
Run 9: Time = 0.567s GFLOPS = 726.60
Run 10: Time = 0.567s GFLOPS = 726.88
Run 11: Time = 0.553s GFLOPS = 745.74
Run 12: Time = 0.548s GFLOPS = 752.52

Running Crypto bench...
Run 1: Time = 0.366s Cycles = 885025351
Run 2: Time = 0.366s Cycles = 886463388
Run 3: Time = 0.363s Cycles = 878682205
Run 4: Time = 0.361s Cycles = 873655316
Run 5: Time = 0.356s Cycles = 862357462
Run 6: Time = 0.361s Cycles = 872472569
Run 7: Time = 0.348s Cycles = 841397145
Run 8: Time = 0.363s Cycles = 878682106
Run 9: Time = 0.361s Cycles = 873925947
Run 10: Time = 0.353s Cycles = 853336201
Run 11: Time = 0.361s Cycles = 872361526
Run 12: Time = 0.355s Cycles = 857629033
---------------------------------------
Estimated CPU Score: 2130
Average SCORE/Core: 88.00
BLAS: 745.94 GFLOPS
Crypto: 105.01 MB/s
Total time: 10.97s
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