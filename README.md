**cbench** is a microbenchmark tool for CPU & GPU using maths operations [BLAS](https://www.netlib.org/blas/) and scores
in terms of FLOPS or GFLOPS, for `matmul` its approx to ~ `2 * N^3` ops/time

```shell
cbench run --runs 5 --warmups 2 --max-threads 12
Warmup runs: 2, Benchmark runs: 5, Threads: 12
Run 1: Time = 23.205s, GFLOPS = 8.88
Run 2: Time = 21.855s, GFLOPS = 9.43
Run 3: Time = 21.272s, GFLOPS = 9.69
Run 4: Time = 21.025s, GFLOPS = 9.81
Run 5: Time = 21.001s, GFLOPS = 9.82
-------------------------------------
Average GFLOPS score: 9.53
Total time: 1.8059822min
Find your CPU here: https://boinc.bakerlab.org/rosetta/cpu_list.php
```

> NEED HELP WITH [BLAS](./src/load.rs), getting memory Bottlenecks

ref

- https://docs.rs/rayon/latest/rayon/index.html
- https://docs.rs/rayon-core/1.13.0/rayon_core/
- https://doc.rust-lang.org/std/simd/
- https://www.netlib.org/blas/
- https://www.netlib.org/benchmark/whetstone.c
- https://setiathome.berkeley.edu/cpu_list.php
- https://en.wikipedia.org/wiki/Floating_point_operations_per_second
- https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms