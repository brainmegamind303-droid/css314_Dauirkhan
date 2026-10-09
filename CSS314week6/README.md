# CUDA Lab 02: Advanced Geometries & Stencils

**Student ID:** 230103386
**Allocated GPU Node:** Tesla T4
**CUDA Compute Capability:** 7.5
**Official Verification Token:** 24911256D599314E66B1

## Task 1: Warp Divergence Benchmark (N = 2^20, 1,000 iterations/element, 10-trial average)

| Kernel | Avg kernel time (ms) | Slowdown vs. Kernel A |
|---|---|---|
| A: Uniform path | 2.9554 ms | 1.00x |
| B: Interleaved divergence (idx % 2) | 9.4545 ms | 3.20x |
| C: Warp-aligned branching (idx // 32) | 3.8811 ms | 1.31x |

**Analysis:** Kernel B is 3.20x slower than Kernel A because adjacent threads (`idx % 2 == 0`) within the exact same 32-thread warp take different execution branches[cite: 2]. As a result, the hardware SIMT unit serializes both paths, executing them sequentially while masking off non-active threads[cite: 2]. In Kernel C (`idx // 32`), branching is aligned to 32-thread warps, meaning all threads within a single warp follow the same execution path without intra-warp divergence, leading to execution performance much closer to the uniform baseline (only 1.31x)[cite: 2].

## Task 2: 1D Stencil
Output: `TASK 2 PASSED: MAX DELTA = 5.960464477539063e-08`

## Task 3: Grid-Stride
256 threads x 64 blocks = 16,384 threads handle 16,777,216 elements; each thread processes ~1,024 elements, stepping by the total grid size[cite: 3, 4].
Output: `TASK 3 PASSED: all 16,777,216 elements == 4.25`

## Task 4: Sobel-X
2048x2048 image, 16x16 blocks, grid = ceil(2048/16) x ceil(2048/16) = 128 x 128 blocks[cite: 4, 5].
Output: `TASK 4 PASSED: MAX DELTA = 4.76837158203125e-07`