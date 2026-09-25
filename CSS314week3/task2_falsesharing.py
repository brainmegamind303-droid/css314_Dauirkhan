import multiprocessing as mp
import time

ITERATIONS = 5_000_000

def worker_adjacent(shared_arr, index):
    for _ in range(ITERATIONS):
        shared_arr[index] += 1

def worker_padded(shared_arr, index):
    padded_idx = index * 16 
    for _ in range(ITERATIONS):
        shared_arr[padded_idx] += 1

def run_test(target_fn, size):
    arr = mp.Array('q', size)
    processes = [mp.Process(target=target_fn, args=(arr, i)) for i in range(4)]
    
    start = time.perf_counter()
    for p in processes: p.start()
    for p in processes: p.join()
    return time.perf_counter() - start

if __name__ == "__main__":
    trials_adj = [run_test(worker_adjacent, size=4) for _ in range(3)]
    trials_pad = [run_test(worker_padded, size=64) for _ in range(3)]
    
    med_adj = sorted(trials_adj)[1]
    med_pad = sorted(trials_pad)[1]
    
    print(f"Adjacent Indices (False Sharing): {med_adj:.4f}s")
    print(f"Padded Indices (Cache-Aligned): {med_pad:.4f}s")
    print(f"Slowdown Factor: {med_adj / med_pad:.2f}x")