"""
Lab 1: The Fork-Join Model, Team Creation, and Thread Scoping (Python / Numba edition)

Run:
    python3 lab1_forkjoin.py

Produces:
    lab1_nondeterminism.txt   (10 runs of raw stdout for Task 1.1)
    lab1_oversubscription.csv (Task 1.2 timing sweep)
    lab1_oversubscription.png (plot of team-instantiation time vs P)
    lab1_saturation.csv       (Task 1.3 CPU-bound workload timings)
"""

import threading
import time
import math
import subprocess
import sys
import csv
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Core fork-join team emulation (from the manual's starter template)
# --------------------------------------------------------------------------
def worker_task(thread_id: int, team_size: int):
    native_tid = threading.get_native_id()
    role = "Master" if thread_id == 0 else "Worker"
    time.sleep(0.001 * (thread_id % 3))
    print(f"[{role}] Logical Rank: {thread_id} of {team_size} | Native OS TID: {native_tid}")


def run_team(num_threads: int):
    print(f"--- Forking a team of {num_threads} threads ---")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_task, tid, num_threads) for tid in range(num_threads)]
        for f in futures:
            f.result()
    print("--- Joined thread team. Execution returned to serial master ---\n")


# --------------------------------------------------------------------------
# Task 1.1: Non-determinism verification -- run 10x, save raw output
# --------------------------------------------------------------------------
def task_1_1(num_runs=10, team_size=4, out_path="lab1_nondeterminism.txt"):
    print(f"[Task 1.1] Capturing {num_runs} independent runs for non-determinism analysis...")
    with open(out_path, "w") as f:
        for run_idx in range(num_runs):
            proc = subprocess.run(
                [sys.executable, "-c",
                 "from lab1_forkjoin import run_team; run_team(4)"],
                cwd=".", capture_output=True, text=True
            )
            f.write(f"=== Run {run_idx + 1} ===\n{proc.stdout}\n")
    print(f"[Task 1.1] Saved raw stdout for {num_runs} runs -> {out_path}\n")


# --------------------------------------------------------------------------
# Task 1.2: Thread oversubscription sweep
# --------------------------------------------------------------------------
def task_1_2(p_values=(1, 2, 4, 8, 16, 32, 64), trials=5, csv_path="lab1_oversubscription.csv"):
    print("[Task 1.2] Running oversubscription sweep...")
    rows = []
    for p in p_values:
        times = []
        for _ in range(trials):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=p) as executor:
                futures = [executor.submit(lambda tid=tid: threading.get_native_id())
                           for tid in range(p)]
                for f in futures:
                    f.result()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg = sum(times) / len(times)
        rows.append((p, avg))
        print(f"  P={p:3d}  mean instantiation+join time = {avg*1000:.3f} ms")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_threads_P", "mean_time_seconds"])
        writer.writerows(rows)

    ps = [r[0] for r in rows]
    ts = [r[1] * 1000 for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(ps, ts, marker="o")
    plt.xlabel("Thread Team Size (P)")
    plt.ylabel("Mean Fork+Join Time (ms)")
    plt.title("Lab 1 - Task 1.2: Thread Team Instantiation Time vs P")
    plt.xscale("log", base=2)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("lab1_oversubscription.png", dpi=150)
    plt.close()
    print(f"[Task 1.2] Saved -> {csv_path}, lab1_oversubscription.png\n")


# --------------------------------------------------------------------------
# Task 1.3: CPU saturation -- artificial workload per thread
# --------------------------------------------------------------------------
def cpu_bound_work(n_sqrt=10_000_000):
    total = 0.0
    for i in range(1, n_sqrt):
        total += math.sqrt(i)
    return total


def task_1_3(p_values=(1, 2, 4, 8), n_sqrt=2_000_000, csv_path="lab1_saturation.csv"):
    # NOTE: reduce n_sqrt for a faster run; raise to 10_000_000 per the manual
    # once you've confirmed the harness works. While this runs, watch
    # htop / Task Manager / Activity Monitor and record core utilization.
    print("[Task 1.3] Running CPU saturation sweep (watch your system monitor now)...")
    rows = []
    for p in p_values:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=p) as executor:
            futures = [executor.submit(cpu_bound_work, n_sqrt) for _ in range(p)]
            for f in futures:
                f.result()
        t1 = time.perf_counter()
        rows.append((p, t1 - t0))
        print(f"  P={p:3d}  total time = {t1 - t0:.3f} s")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_threads_P", "wall_clock_seconds"])
        writer.writerows(rows)
    print(f"[Task 1.3] Saved -> {csv_path}\n")


if __name__ == "__main__":
    run_team(4)
    task_1_1()
    task_1_2()
    task_1_3()
