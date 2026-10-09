"""Task 1: Warp divergence microbenchmark."""
import time
import numpy as np
from numba import cuda

N = 2 ** 20
ITERS = 1000
TPB = 256

A = np.float32(1.0001)
B = np.float32(0.0001)


@cuda.jit(device=True, inline=True)
def path1(v):
    # 1,000 multiply-accumulate operations
    for _ in range(ITERS):
        v = v * A + B
    return v


@cuda.jit(device=True, inline=True)
def path2(v):
    # 1,000 distinct subtract-divide operations
    for _ in range(ITERS):
        v = (v - B) / A
    return v


@cuda.jit
def kernel_uniform(y, n):
    idx = cuda.grid(1)
    if idx < n:
        v = y[idx]
        for _ in range(ITERS):
            v = v * A + B
        y[idx] = v


@cuda.jit
def kernel_divergent(y, n):
    idx = cuda.grid(1)
    if idx < n:
        v = y[idx]
        if idx % 2 == 0:
            v = path1(v)
        else:
            v = path2(v)
        y[idx] = v


@cuda.jit
def kernel_warp_aligned(y, n):
    idx = cuda.grid(1)
    if idx < n:
        v = y[idx]
        warp_id = idx // 32
        if warp_id % 2 == 0:
            v = path1(v)
        else:
            v = path2(v)
        y[idx] = v


def bench(kernel, d_y, blocks, trials=10):
    kernel[blocks, TPB](d_y, N)          # warm-up (also triggers JIT compile)
    cuda.synchronize()
    times = []
    for _ in range(trials):
        cuda.synchronize()
        t0 = time.perf_counter()
        kernel[blocks, TPB](d_y, N)
        cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return 1000.0 * float(np.mean(times))   # ms


def main():
    h_y = np.ones(N, dtype=np.float32)
    d_y = cuda.to_device(h_y)               # transfer excluded from timing
    blocks = (N + TPB - 1) // TPB

    results = {}
    for name, k in [("Kernel A (Uniform)", kernel_uniform),
                    ("Kernel B (Interleaved Divergence)", kernel_divergent),
                    ("Kernel C (Warp-Aligned)", kernel_warp_aligned)]:
        d_y.copy_to_device(h_y)             # reset data before each kernel
        results[name] = bench(k, d_y, blocks)

    base = results["Kernel A (Uniform)"]
    print(f"{'Kernel':40s} {'Avg ms':>10s} {'vs A':>8s}")
    for name, ms in results.items():
        print(f"{name:40s} {ms:10.4f} {ms / base:7.2f}x")


if __name__ == "__main__":
    main()
