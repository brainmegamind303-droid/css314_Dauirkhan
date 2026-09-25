import threading
import time
TOTAL_OPS = 2_000_000
NUM_THREADS = 4
class UnsafeCounter:
    def __init__(self): self.val = 0
    def inc(self): self.val += 1

class LockedCounter:
    def __init__(self):
        self.val = 0
        self.lock = threading.Lock()
    def inc(self):
        with self.lock:
            self.val += 1

class LocklessCounter:
    def __init__(self): self.val = 0
    def run_bench(self):
        ops_per_thread = TOTAL_OPS // NUM_THREADS
        results = [0] * NUM_THREADS
        
        def work(thread_id):
            local_acc = 0
            for _ in range(ops_per_thread):
                local_acc += 1
            results[thread_id] = local_acc

        threads = [threading.Thread(target=work, args=(i,)) for i in range(NUM_THREADS)]
        start = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join()
        
        # Reduction stage
        self.val = sum(results)
        return self.val, time.perf_counter() - start

def bench(counter_class):
    c = counter_class()
    ops_per_thread = TOTAL_OPS // NUM_THREADS
    def work():
        for _ in range(ops_per_thread):
            c.inc()
    threads = [threading.Thread(target=work) for _ in range(NUM_THREADS)]
    start = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    return c.val, time.perf_counter() - start

if __name__ == "__main__":
    val_u, t_u = bench(UnsafeCounter)
    val_l, t_l = bench(LockedCounter)
    val_ll, t_ll = LocklessCounter().run_bench()

    print(f"Unsafe: Value = {val_u:,} / {TOTAL_OPS:,} | Time: {t_u:.4f}s")
    print(f"Locked: Value = {val_l:,} / {TOTAL_OPS:,} | Time: {t_l:.4f}s")
    print(f"Lockless: Value = {val_ll:,} / {TOTAL_OPS:,} | Time: {t_ll:.4f}s")
    print(f"Contention Cost Multiplier: {t_l / t_u:.2f}x")
    print(f"Lockless Speedup over Locked: {t_l / t_ll:.2f}x")