"""
Lab 2: Numerical Integration (Pi Approximation) & Parallel Reductions (Python / Numba)

Run:
    python3 lab2_pi_reduction.py

Produces:
    lab2_race_condition.csv     (Task 2.1)
    lab2_critical_overhead.csv  (Task 2.2)
    lab2_scaling.csv            (Task 2.3 / 2.4)
    lab2_speedup.png            (Task 2.4 speedup vs. ideal)
"""

import csv
import time
import threading
import numpy as np
from numba import njit, prange

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_STEPS = 100_000_000   # reduce for quicker local iteration while debugging


# --------------------------------------------------------------------------
# Serial baseline
# --------------------------------------------------------------------------
@njit
def calc_pi_serial(num_steps: int) -> float:
    step = 1.0 / num_steps
    total_sum = 0.0
    for i in range(num_steps):
        x = (i + 0.5) * step
        total_sum += 4.0 / (1.0 + x * x)
    return total_sum * step


# --------------------------------------------------------------------------
# Parallel tree reduction (Numba infers reduction(+:total_sum) from prange)
# --------------------------------------------------------------------------
@njit(parallel=True)
def calc_pi_reduction(num_steps: int) -> float:
    step = 1.0 / num_steps
    total_sum = 0.0
    for i in prange(num_steps):
        x = (i + 0.5) * step
        total_sum += 4.0 / (1.0 + x * x)
    return total_sum * step


# --------------------------------------------------------------------------
# Task 2.1: Naive unsynchronized race condition (pure Python threads --
# demonstrates the *concept*; Numba's nogil release + GIL semantics differ
# from a true OpenMP data race, so this uses plain threading.Thread on a
# Python-level accumulator to reproduce lost updates.)
# --------------------------------------------------------------------------
def _race_worker(shared, start, end, step):
    for i in range(start, end):
        x = (i + 0.5) * step
        shared[0] += 4.0 / (1.0 + x * x)   # unprotected read-modify-write


def run_naive_race(num_steps: int, threads: int) -> float:
    step = 1.0 / num_steps
    shared = [0.0]
    chunk = num_steps // threads
    pool = []
    for t in range(threads):
        start = t * chunk
        end = num_steps if t == threads - 1 else start + chunk
        th = threading.Thread(target=_race_worker, args=(shared, start, end, step))
        pool.append(th)
        th.start()
    for th in pool:
        th.join()
    return shared[0] * step


def task_2_1(p_values=(1, 2, 4, 8), num_steps=2_000_000, csv_path="lab2_race_condition.csv"):
    print("[Task 2.1] Race condition quantification...")
    rows = []
    for p in p_values:
        pi_val = run_naive_race(num_steps, p)
        err = abs(pi_val - np.pi)
        rows.append((p, pi_val, err))
        print(f"  P={p:2d}  pi={pi_val:.10f}  error={err:.3e}")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threads_P", "computed_pi", "absolute_error"])
        w.writerows(rows)
    print(f"[Task 2.1] Saved -> {csv_path}\n")


# --------------------------------------------------------------------------
# Task 2.2: Critical section overhead (lock around every accumulation)
# --------------------------------------------------------------------------
def _critical_worker(shared, lock, start, end, step):
    for i in range(start, end):
        x = (i + 0.5) * step
        term = 4.0 / (1.0 + x * x)
        with lock:
            shared[0] += term


def run_critical_section(num_steps: int, threads: int) -> float:
    step = 1.0 / num_steps
    shared = [0.0]
    lock = threading.Lock()
    chunk = num_steps // threads
    pool = []
    for t in range(threads):
        start = t * chunk
        end = num_steps if t == threads - 1 else start + chunk
        th = threading.Thread(target=_critical_worker, args=(shared, lock, start, end, step))
        pool.append(th)
        th.start()
    for th in pool:
        th.join()
    return shared[0] * step


def task_2_2(num_steps=1_000_000, threads=4, csv_path="lab2_critical_overhead.csv"):
    print("[Task 2.2] Critical section overhead...")
    t0 = time.perf_counter()
    _ = calc_pi_serial(num_steps)          # warm up + serial baseline timing
    t1 = time.perf_counter()
    serial_time = t1 - t0

    t2 = time.perf_counter()
    pi_val = run_critical_section(num_steps, threads)
    t3 = time.perf_counter()
    critical_time = t3 - t2

    overhead_pct = (critical_time - serial_time) / serial_time * 100
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "time_seconds", "pi_value"])
        w.writerow(["serial_baseline", serial_time, None])
        w.writerow(["critical_section", critical_time, pi_val])
        w.writerow(["overhead_percent", overhead_pct, None])
    print(f"  serial={serial_time:.4f}s  critical={critical_time:.4f}s  overhead={overhead_pct:.1f}%")
    print(f"[Task 2.2] Saved -> {csv_path}\n")


# --------------------------------------------------------------------------
# Task 2.3 / 2.4: Strong scaling of the Numba parallel reduction + speedup/efficiency
# NOTE: Numba's `parallel=True` uses NUMBA_NUM_THREADS threads set at process
# start (env var). To sweep P you must re-launch the process per P value, or
# call numba.set_num_threads(p) before each timed call (works in recent Numba).
# --------------------------------------------------------------------------
def task_2_3_2_4(p_values=(1, 2, 4, 8, 16), trials=5, num_steps=N_STEPS,
                  csv_path="lab2_scaling.csv", plot_path="lab2_speedup.png"):
    import numba
    print("[Task 2.3/2.4] Strong scaling sweep of parallel reduction...")
    _ = calc_pi_reduction(1000)  # warm-up JIT compile

    results = {}
    for p in p_values:
        numba.set_num_threads(p)
        times = []
        for _ in range(trials):
            t0 = time.perf_counter()
            calc_pi_reduction(num_steps)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg = sum(times) / len(times)
        results[p] = avg
        print(f"  P={p:3d}  mean time = {avg:.4f}s")

    t1 = results[p_values[0]]  # baseline should be P=1
    rows = []
    for p in p_values:
        speedup = t1 / results[p]
        efficiency = speedup / p
        rows.append((p, results[p], speedup, efficiency))

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threads_P", "mean_time_seconds", "speedup_S(P)", "efficiency_E(P)"])
        w.writerows(rows)

    ps = [r[0] for r in rows]
    speedups = [r[2] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(ps, speedups, marker="o", label="Measured Speedup")
    plt.plot(ps, ps, linestyle="--", label="Ideal Linear Speedup")
    plt.xlabel("Thread Count (P)")
    plt.ylabel("Speedup S(P)")
    plt.title("Lab 2 - Task 2.4: Speedup vs. Ideal Linear Speedup")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[Task 2.3/2.4] Saved -> {csv_path}, {plot_path}\n")


if __name__ == "__main__":
    task_2_1()
    task_2_2()
    task_2_3_2_4()
