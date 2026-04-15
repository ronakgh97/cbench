**cbench** is a microbenchmark tool for CPU & GPU, performs a series of maths
computation [BLAS](https://www.netlib.org/blas/) and scores in terms of FLOPS or GFLOPS

```shell
cbench run --runs 8 --warmups 3 --max-threads 12
Warmup runs: 3, Benchmark runs: 8, Threads: 12
Run 1: Time = 1.142s, GFLOPS = 208.26
Run 2: Time = 1.165s, GFLOPS = 204.12
Run 3: Time = 1.133s, GFLOPS = 209.85
Run 4: Time = 1.120s, GFLOPS = 212.35
Run 5: Time = 1.148s, GFLOPS = 207.06
Run 6: Time = 1.183s, GFLOPS = 201.02
Run 7: Time = 1.130s, GFLOPS = 210.50
Run 8: Time = 1.138s, GFLOPS = 208.95
-------------------------------------
Average GFLOPS/core: 17.31
Total time: 9.159081s
Find your CPU here: https://boinc.bakerlab.org/rosetta/cpu_list.php
```

ref

- https://docs.rs/rayon/latest/rayon/index.html
- https://docs.rs/rayon-core/1.13.0/rayon_core/
- https://doc.rust-lang.org/std/simd/
- https://www.netlib.org/blas/
- https://www.netlib.org/benchmark/whetstone.c
- https://setiathome.berkeley.edu/cpu_list.php
- https://en.wikipedia.org/wiki/Floating_point_operations_per_second
- https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms