import time
import random
import multiprocessing as mp

TOTAL_POINTS = 100_000_000
def worker_reduction(points_per_process):
    local_hits = 0
    for _ in range(points_per_process):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            local_hits += 1
    return local_hits

def main():
    print("=== PART 3: OpenMP-Style Reduction ===")
    thread_counts = [1, 2, 4, 8, 16, 32]
    baseline_time = 0.0

    print(f"{'Threads(T)':<10} | {'Runtime(ms)':<12} | {'Speedup':<15} | {'Efficiency':<12}")
    print("-" * 58)

    for num_threads in thread_counts:
        start_time = time.time()
        points_per_process = TOTAL_POINTS // num_threads

        # multiprocessing
        with mp.Pool(processes=num_threads) as pool:
            results = pool.map(worker_reduction, [points_per_process] * num_threads)
        duration = (time.time() - start_time) * 1000
        if num_threads == 1:
            baseline_time = duration

        speedup = baseline_time / duration
        efficiency = (speedup / num_threads) * 100

        print(f"{num_threads:<10} | {duration:<12.0f} | {speedup:<15.2f}x | {efficiency:<11.1f}%")
if __name__ == "__main__":
    main()