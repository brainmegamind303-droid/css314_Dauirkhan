"""
Lab 4: Memory Hierarchy, Cache Coherency, and False Sharing -- Python / Numba

Run:
    python3 lab4_false_sharing.py

Produces:
    lab4_scaling.csv   (Task 4.1/4.2: unpadded vs padded vs thread-local, P in {1,2,4,8,16})
    lab4_scaling.png   (scaling plot of all three variants)

Note: run with `perf stat -e L1-dcache-load-misses,L1-dcache-store-misses
python3 lab4_false_sharing.py` on Linux for Task 4.4's cache-miss counts.
"""

import csv
import time
import numpy as np
from numba import njit, prange

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ITERATIONS = 100_000_000


# --------------------------------------------------------------------------
# 1. Unpadded: adjacent int64 counters share a 64-byte cache line
#    (8 bytes * 8 counters = 64 bytes -> every core's writes invalidate
#    every other core's cached copy of the line)
# --------------------------------------------------------------------------
@njit(parallel=True)
def false_sharing_test(num_threads: int, iters: int):
    counters = np.zeros(num_threads, dtype=np.int64)
    for tid in prange(num_threads):
        for _ in range(iters):
            counters[tid] += 1
    return counters


# --------------------------------------------------------------------------
# 2. Padded: stride of 8 int64 elements = 64 bytes between counters, so each
#    thread's counter occupies its own exclusive cache line.
# --------------------------------------------------------------------------
@njit(parallel=True)
def padded_sharing_test(num_threads: int, iters: int):
    stride = 8
    counters = np.zeros(num_threads * stride, dtype=np.int64)
    for tid in prange(num_threads):
        idx = tid * stride
        for _ in range(iters):
            counters[idx] += 1
    return counters


# --------------------------------------------------------------------------
# 3. Thread-local accumulator: increment a register-resident local scalar,
#    write to the shared array exactly once at the end of the loop.
# --------------------------------------------------------------------------
@njit(parallel=True)
def local_accumulator_test(num_threads: int, iters: int):
    counters = np.zeros(num_threads, dtype=np.int64)
    for tid in prange(num_threads):
        local = 0
        for _ in range(iters):
            local += 1
        counters[tid] = local
    return counters


def task_4_1_4_2_4_3(p_values=(1, 2, 4, 8, 16), iters=ITERATIONS,
                      csv_path="lab4_scaling.csv", plot_path="lab4_scaling.png"):
    print("[Task 4.1/4.2/4.3] Benchmarking unpadded vs padded vs thread-local...")
    # warm up JIT
    false_sharing_test(2, 1000)
    padded_sharing_test(2, 1000)
    local_accumulator_test(2, 1000)

    rows = []
    unpadded_times, padded_times, local_times = [], [], []
    for p in p_values:
        t0 = time.perf_counter(); false_sharing_test(p, iters); t1 = time.perf_counter()
        t2 = time.perf_counter(); padded_sharing_test(p, iters); t3 = time.perf_counter()
        t4 = time.perf_counter(); local_accumulator_test(p, iters); t5 = time.perf_counter()
        unpadded_times.append(t1 - t0)
        padded_times.append(t3 - t2)
        local_times.append(t5 - t4)
        rows.append((p, t1 - t0, t3 - t2, t5 - t4))
        print(f"  P={p:3d}  unpadded={t1-t0:.3f}s  padded={t3-t2:.3f}s  local={t5-t4:.3f}s")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threads_P", "unpadded_seconds", "padded_seconds", "thread_local_seconds"])
        w.writerows(rows)

    plt.figure(figsize=(7, 5))
    plt.plot(p_values, unpadded_times, marker="o", label="Unpadded (False Sharing)")
    plt.plot(p_values, padded_times, marker="s", label="Padded (64B stride)")
    plt.plot(p_values, local_times, marker="^", label="Thread-Local Accumulator")
    plt.xlabel("Thread Count (P)")
    plt.ylabel("Execution Time (s)")
    plt.title("Lab 4: False Sharing Impact on Scaling")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[Task 4.1/4.2/4.3] Saved -> {csv_path}, {plot_path}\n")


if __name__ == "__main__":
    task_4_1_4_2_4_3()
