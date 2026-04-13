**cbench** is a microbenchmark tool for CPU & GPU, performs a series of maths
computation [BLAS](https://www.netlib.org/blas/) and scores in terms of FLOPS or GFLOPS

```shell
cbench run --runs 5 --warmups 2 --max-threads 12
Warmup runs: 2, Benchmark runs: 5, Threads: 12
Run 1: Time = 1.331s, GFLOPS = 154.87
Run 2: Time = 1.369s, GFLOPS = 150.53
Run 3: Time = 1.291s, GFLOPS = 159.62
Run 4: Time = 1.365s, GFLOPS = 150.98
Run 5: Time = 1.379s, GFLOPS = 149.50
-------------------------------------
Average GFLOPS score: 153.10
Total time: 6.7351074s
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