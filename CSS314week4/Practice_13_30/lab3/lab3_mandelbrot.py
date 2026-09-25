"""
Lab 3: Work-Sharing & Loop Scheduling Policies (Mandelbrot Fractal) -- Python / Numba

Run:
    python3 lab3_mandelbrot.py

Produces:
    lab3_sweep.csv          (Task 3.2: 4x4 matrix, P in {2,4,8,16} x C in {1,16,64,256})
    lab3_heatmap.png        (Task 3.3)
    lab3_imbalance.csv      (Task 3.4: per-thread iteration counts + imbalance metric)
"""

import csv
import time
import threading
import numpy as np
from numba import njit, prange

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH, HEIGHT = 1920, 1080
MAX_ITER = 1000


@njit
def compute_pixel(px, py, width, height, max_iter):
    x0 = (px - width / 2.0) * 4.0 / width
    y0 = (py - height / 2.0) * 4.0 / height
    x, y = 0.0, 0.0
    iteration = 0
    while x * x + y * y <= 4.0 and iteration < max_iter:
        xtemp = x * x - y * y + x0
        y = 2.0 * x * y + y0
        x = xtemp
        iteration += 1
    return iteration


# --------------------------------------------------------------------------
# 1. Static scheduling (Numba's default prange chunking)
# --------------------------------------------------------------------------
@njit(parallel=True)
def render_mandelbrot_static(width, height, max_iter):
    img = np.zeros((height, width), dtype=np.int32)
    for y in prange(height):
        for x in range(width):
            img[y, x] = compute_pixel(x, y, width, height, max_iter)
    return img


# --------------------------------------------------------------------------
# 2. Dynamic scheduling -- explicit shared work-queue with a configurable
#    chunk size, built with plain threading (mirrors the Java AtomicInteger
#    dispenser pattern in the manual, since Numba's prange has no native
#    schedule(dynamic, chunk) knob).
# --------------------------------------------------------------------------
def render_mandelbrot_dynamic(width, height, max_iter, num_threads, chunk_size):
    img = np.zeros((height, width), dtype=np.int32)
    next_row = [0]
    lock = threading.Lock()
    per_thread_iters = [0] * num_threads   # for Task 3.4 imbalance measurement

    def worker(tid):
        local_iters = 0
        while True:
            with lock:
                start = next_row[0]
                if start >= height:
                    break
                end = min(start + chunk_size, height)
                next_row[0] = end
            for y in range(start, end):
                for x in range(width):
                    iters = compute_pixel(x, y, width, height, max_iter)
                    img[y, x] = iters
                    local_iters += iters
        per_thread_iters[tid] = local_iters

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return img, per_thread_iters


# --------------------------------------------------------------------------
# Task 3.2: 4x4 parameter sweep (threads x chunk size)
# --------------------------------------------------------------------------
def task_3_2(p_values=(2, 4, 8, 16), chunk_values=(1, 16, 64, 256), trials=3,
             csv_path="lab3_sweep.csv", heatmap_path="lab3_heatmap.png"):
    print("[Task 3.2/3.3] Running 4x4 dynamic-scheduling sweep...")
    _ = render_mandelbrot_static(100, 100, 100)  # warm up JIT for compute_pixel
    grid = np.zeros((len(p_values), len(chunk_values)))
    rows = []
    for pi_, p in enumerate(p_values):
        for ci, c in enumerate(chunk_values):
            times = []
            for _ in range(trials):
                t0 = time.perf_counter()
                render_mandelbrot_dynamic(WIDTH, HEIGHT, MAX_ITER, p, c)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            avg = sum(times) / len(times)
            grid[pi_, ci] = avg
            rows.append((p, c, avg))
            print(f"  P={p:3d}  chunk={c:3d}  mean time = {avg:.3f}s")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threads_P", "chunk_size_C", "mean_time_seconds"])
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(chunk_values)))
    ax.set_xticklabels(chunk_values)
    ax.set_yticks(range(len(p_values)))
    ax.set_yticklabels(p_values)
    ax.set_xlabel("Chunk Size (C)")
    ax.set_ylabel("Thread Count (P)")
    ax.set_title("Lab 3 - Task 3.3: Mandelbrot Render Time (s) Heatmap")
    for pi_ in range(len(p_values)):
        for ci in range(len(chunk_values)):
            ax.text(ci, pi_, f"{grid[pi_, ci]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(im, label="Time (s)")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"[Task 3.2/3.3] Saved -> {csv_path}, {heatmap_path}\n")


# --------------------------------------------------------------------------
# Task 3.4: Load imbalance metric using the dynamic scheduler's per-thread counters
# --------------------------------------------------------------------------
def task_3_4(num_threads=4, chunk_size=16, csv_path="lab3_imbalance.csv"):
    print("[Task 3.4] Measuring per-thread load imbalance...")
    _, per_thread_iters = render_mandelbrot_dynamic(WIDTH, HEIGHT, MAX_ITER, num_threads, chunk_size)
    max_w, min_w, avg_w = max(per_thread_iters), min(per_thread_iters), sum(per_thread_iters) / len(per_thread_iters)
    imbalance = (max_w - min_w) / avg_w
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["thread_id", "total_iterations_computed"])
        for tid, work in enumerate(per_thread_iters):
            w.writerow([tid, work])
        w.writerow([])
        w.writerow(["Max", max_w])
        w.writerow(["Min", min_w])
        w.writerow(["Avg", avg_w])
        w.writerow(["Imbalance_(Max-Min)/Avg", imbalance])
    print(f"  per-thread work = {per_thread_iters}")
    print(f"  imbalance = {imbalance:.4f}")
    print(f"[Task 3.4] Saved -> {csv_path}\n")


if __name__ == "__main__":
    task_3_2()
    task_3_4()
