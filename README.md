
**cbench** is a microbenchmark tool for x86 CPU,
performs a series of random maths computation (blas/crypto/pi) and scores in terms of GFLOPS & 'meaningful' metrics

```shell
cbench run --runs 12 --warmups 3 --max-threads 24
Warmup runs: 3 Benchmark runs: 12 Threads: 24

Running BLAS bench...
Run 1: Time = 0.488s GFLOPS = 844.04
Run 2: Time = 0.503s GFLOPS = 820.12
Run 3: Time = 0.515s GFLOPS = 800.43
Run 4: Time = 0.492s GFLOPS = 837.29
Run 5: Time = 0.499s GFLOPS = 826.51
Run 6: Time = 0.488s GFLOPS = 844.37
Run 7: Time = 0.497s GFLOPS = 829.50
Run 8: Time = 0.504s GFLOPS = 817.40
Run 9: Time = 0.503s GFLOPS = 819.99
Run 10: Time = 0.508s GFLOPS = 812.21
Run 11: Time = 0.507s GFLOPS = 812.73
Run 12: Time = 0.505s GFLOPS = 816.47

Running Crypto bench...
Run 1: Time = 0.331s Cycles = 801163363
Run 2: Time = 0.342s Cycles = 828547666
Run 3: Time = 0.342s Cycles = 828459561
Run 4: Time = 0.338s Cycles = 817078197
Run 5: Time = 0.340s Cycles = 821889399
Run 6: Time = 0.331s Cycles = 799835009
Run 7: Time = 0.326s Cycles = 789520575
Run 8: Time = 0.328s Cycles = 794557962
Run 9: Time = 0.323s Cycles = 782067216
Run 10: Time = 0.328s Cycles = 794023725
Run 11: Time = 0.347s Cycles = 839391206
Run 12: Time = 0.344s Cycles = 831175626
---------------------------------------
Estimated CPU Score: 2216
Average SCORE/Core: 92.00
BLAS: 823.42 GFLOPS
Crypto: 112.65 MB/s
Total time: 10.03s
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