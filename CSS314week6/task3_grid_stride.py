"""Task 3: Grid-stride loop vector scaling."""
import numpy as np
from numba import cuda


@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor


def run_grid_stride(h_arr, factor):
    h_arr = np.ascontiguousarray(h_arr, dtype=np.float32)
    N = h_arr.size
    d_arr = cuda.to_device(h_arr)
    threads_per_block = 256
    blocks_per_grid = 64          # 16,384 threads total, far fewer than N
    grid_stride_scale_kernel[blocks_per_grid, threads_per_block](
        d_arr, np.float32(factor), N)
    return d_arr.copy_to_host()


if __name__ == "__main__":
    N = 2 ** 24
    factor = 4.25
    h = np.ones(N, dtype=np.float32)
    res = run_grid_stride(h, factor)
    assert res.size == N and np.all(res == np.float32(factor))
    print(f"TASK 3 PASSED: all {N:,} elements == {factor}")
