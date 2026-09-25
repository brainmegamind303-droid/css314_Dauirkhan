import time
import os
import platform
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def collatz_steps(n: int) -> int:
    steps = 0
    while n > 1:
        if (n & 1) == 0:
            n >>= 1
        else:
            n = 3 * n + 1
        steps += 1
    return steps

def process_range(start_end):
    start, end = start_end
    local_max = 0
    local_sum = 0
    local_hits = 0
    
    for i in range(start, end + 1):
        steps = collatz_steps(i)
        if steps > local_max:
            local_max = steps
        local_sum = (local_sum + steps) % 1000000007
        if steps > 100:
            local_hits += 1
            
    return local_max, local_sum, local_hits


def worker_false_sharing(args):
    start, end, thread_id, shared_array = args
    for i in range(start, end + 1):
        if collatz_steps(i) > 100:
            shared_array[thread_id] += 1


if __name__ == '__main__':
    LAST_4_DIGITS = 1234  
    N = 10_000_000 + (LAST_4_DIGITS * 1_000)

    print("=== Python Parallel Computing Lab: Amdahl Reality Gap ===")
    print(f"Target N = {N}\n")

    thread_counts = [1, 2, 4, 8, 16]
    num_runs = 3

    print("--- PHASE 3: Multi-Process Scaling ---")
    print("Workers | Run 1 (Cold) | Run 2        | Run 3        | Avg Time (s) | Speedup")
    
    t_seq_avg = 0.0

    for num_workers in thread_counts:
        times = []
        for run in range(num_runs):
            chunk_size = N // num_workers
            ranges = []
            for w in range(num_workers):
                r_start = w * chunk_size + 1
                r_end = N if w == num_workers - 1 else (w + 1) * chunk_size
                ranges.append((r_start, r_end))

            start_time = time.perf_counter()

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(process_range, ranges))

            global_max = max(r[0] for r in results)
            global_sum = sum(r[1] for r in results) % 1000000007

            end_time = time.perf_counter()
            times.append(end_time - start_time)

        avg_time = (times[1] + times[2]) / 2.0
        if num_workers == 1:
            t_seq_avg = avg_time

        speedup = t_seq_avg / avg_time

        print(f"{num_workers:7d} | {times[0]:12.6f} | {times[1]:12.6f} | {times[2]:12.6f} | {avg_time:12.6f} | {speedup:7.2f}x")

    print("\n--- PHASE 4 (Exp A): False Sharing Penalty (Inter-process Shared Memory) ---")
    max_workers = os.cpu_count() or 4
    
    chunk_size = N // max_workers
    ranges = []
    for w in range(max_workers):
        r_start = w * chunk_size + 1
        r_end = N if w == max_workers - 1 else (w + 1) * chunk_size
        ranges.append((r_start, r_end))

    # Var 1
    shared_arr = mp.Array('i', max_workers)
    tasks_naive = [(ranges[w][0], ranges[w][1], w, shared_arr) for w in range(max_workers)]

    start_time = time.perf_counter()
    processes = []
    for t in tasks_naive:
        p = mp.Process(target=worker_false_sharing, args=(t,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    t_naive = time.perf_counter() - start_time
    total_naive = sum(shared_arr)

  #var 2
    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_range, ranges))
    t_mitigated = time.perf_counter() - start_time
    total_reduction = sum(r[2] for r in results)

    print(f"Naive Shared Memory (False Sharing) : {t_naive:.6f} sec (Hits: {total_naive})")
    print(f"Reduction / Local Variables (Clean) : {t_mitigated:.6f} sec (Hits: {total_reduction})")
    print(f"Speedup Penalty Ratio               : {t_naive / t_mitigated:.2f}x")

    print("\n--- PHASE 4 (Exp B): Workload Imbalance & Chunk Size Evaluation ---")
    print(f"{'Chunking / Strategy':<30} | Execution Time (s)")
    print("-" * 52)

    chunk_sizes = [
        ("Static Large (N / Workers)", N // max_workers),
        ("Chunk Size 10,000", 10_000),
        ("Chunk Size 1,000", 1_000),
        ("Chunk Size 100 (Fine-grained)", 100),
    ]

    for label, c_size in chunk_sizes:
        start_time = time.perf_counter()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:

            results = list(executor.map(collatz_steps, range(1, N + 1), chunksize=c_size))
            
        exec_time = time.perf_counter() - start_time
        print(f"{label:<30} | {exec_time:.6f} s")