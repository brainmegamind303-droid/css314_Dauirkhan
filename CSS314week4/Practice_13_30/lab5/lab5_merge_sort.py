"""
Lab 5: Recursive Task-Based Parallelism (Parallel Merge Sort) -- Python

Run:
    python3 lab5_merge_sort.py

Produces:
    lab5_cutoff_sweep.csv   (Task 5.2: execution time vs cutoff K)
    lab5_cutoff_sweep.png   (plotted against log(K))
    lab5_work_span.txt      (Task 5.3: measured T1 work, notes for T_infinity)
"""

import csv
import time
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sequential_merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = sequential_merge_sort(arr[:mid])
    right = sequential_merge_sort(arr[mid:])
    return merge(left, right)


def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def parallel_merge_sort_entry(arr, cutoff=50000, depth=0, max_depth=3):
    """
    IMPORTANT DEVIATION FROM THE MANUAL'S STARTER CODE:
    The manual's original template creates a brand-new ProcessPoolExecutor at
    *every* recursive call. Unlike a real OpenMP task (a few bytes queued
    onto a work-stealing deque) or a Java ForkJoinTask, spawning a Python OS
    process is extremely expensive (tens of milliseconds). Sweeping the
    cutoff K down to 1 or 10 on a large N recreates thousands of process
    pools and can hang or exhaust the OS process table.

    A practical implementation caps *parallel* recursion depth (max_depth) --
    beyond that depth it falls back to plain sequential recursion -- while
    `cutoff` still governs the switch to the fast base-case sort. This
    mirrors how a real task scheduler avoids task-creation overhead
    dominating runtime for fine-grained work, and keeps the cutoff sweep
    (Task 5.2) meaningful without an OpenMP/ForkJoin-grade lightweight task
    runtime. Document this deviation and its rationale in your lab report.
    """
    if len(arr) <= cutoff:
        return sequential_merge_sort(arr)
    mid = len(arr) // 2
    if depth >= max_depth:
        left = parallel_merge_sort_entry(arr[:mid], cutoff, depth + 1, max_depth)
        right = parallel_merge_sort_entry(arr[mid:], cutoff, depth + 1, max_depth)
        return merge(left, right)
    with ProcessPoolExecutor(max_workers=2) as executor:
        f_left = executor.submit(parallel_merge_sort_entry, arr[:mid], cutoff, depth + 1, max_depth)
        f_right = executor.submit(parallel_merge_sort_entry, arr[mid:], cutoff, depth + 1, max_depth)
        left = f_left.result()
        right = f_right.result()
    return merge(left, right)


# --------------------------------------------------------------------------
# Task 5.1: correctness verification
# --------------------------------------------------------------------------
def task_5_1(n=200_000, cutoff=10_000):
    print("[Task 5.1] Verifying correctness...")
    data = np.random.randint(0, 10_000_000, size=n).tolist()
    result = parallel_merge_sort_entry(data, cutoff=cutoff)
    assert result == sorted(data), "Sort output does not match ground truth!"
    print("  PASSED: output is correctly sorted.\n")


# --------------------------------------------------------------------------
# Task 5.2: cutoff threshold sweep
# --------------------------------------------------------------------------
def task_5_2(n=5_000_000, cutoffs=(1, 10, 100, 1_000, 10_000, 50_000, 100_000),
             csv_path="lab5_cutoff_sweep.csv", plot_path="lab5_cutoff_sweep.png"):
    print("[Task 5.2] Cutoff threshold sweep...")
    data = np.random.randint(0, 10_000_000, size=n).tolist()
    rows = []
    for k in cutoffs:
        copy = list(data)
        t0 = time.perf_counter()
        parallel_merge_sort_entry(copy, cutoff=k)
        t1 = time.perf_counter()
        rows.append((k, t1 - t0))
        print(f"  cutoff K={k:7d}  time={t1 - t0:.4f}s")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cutoff_K", "time_seconds"])
        w.writerows(rows)

    ks = [r[0] for r in rows]
    ts = [r[1] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(ks, ts, marker="o")
    plt.xscale("log")
    plt.xlabel("Sequential Cutoff Threshold K (log scale)")
    plt.ylabel("Execution Time (s)")
    plt.title("Lab 5 - Task 5.2: Merge Sort Time vs. Cutoff K")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[Task 5.2] Saved -> {csv_path}, {plot_path}\n")


# --------------------------------------------------------------------------
# Task 5.3: Work (T1) measurement -- Span (T_infinity) is derived analytically.
# For merge sort: T1 = O(N log N); T_infinity = O(N) because each level's
# merge phase is a strictly sequential O(N) scan that dominates the critical
# path, even though the splits themselves form an O(log N)-deep tree.
# --------------------------------------------------------------------------
def task_5_3(n=5_000_000, out_path="lab5_work_span.txt"):
    print("[Task 5.3] Measuring Work (T1)...")
    data = np.random.randint(0, 10_000_000, size=n).tolist()
    t0 = time.perf_counter()
    sequential_merge_sort(data)
    t1 = time.perf_counter()
    work_t1 = t1 - t0

    with open(out_path, "w") as f:
        f.write("Lab 5 Task 5.3: Work-Span Analysis\n")
        f.write("===================================\n\n")
        f.write(f"N = {n}\n")
        f.write(f"Measured Work T1 (sequential merge sort, single core) = {work_t1:.4f} s\n\n")
        f.write("Span (T_infinity) estimate:\n")
        f.write("  For this recursive merge sort, the recursive *splits* form a tree of\n")
        f.write("  depth O(log N), but each level's merge step is a strictly sequential\n")
        f.write("  O(N) scan (see merge()). Because the merge cannot itself be\n")
        f.write("  parallelized in this implementation, the critical path length is\n")
        f.write("  dominated by the top-level merge: T_infinity = O(N).\n\n")
        f.write("  Theoretical parallelism P_theoretical = T1 / T_infinity\n")
        f.write("                                        = O(N log N) / O(N) = O(log N)\n")
        f.write(f"                                        ~= log2({n}) = {math.log2(n):.2f}\n\n")
        f.write("  This means additional cores beyond ~log2(N) processors yield\n")
        f.write("  diminishing returns for this specific implementation, unless the\n")
        f.write("  merge step itself is parallelized (e.g. parallel merge via binary\n")
        f.write("  search partitioning of the two sorted halves).\n")
    print(f"  Work T1 = {work_t1:.4f}s; see {out_path} for full derivation\n")


if __name__ == "__main__":
    task_5_1()
    task_5_2()
    task_5_3()
