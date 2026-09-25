import time
import random
from threading import Thread, Lock

TOTAL_POINTS = 50_000_000
total_hits = 0
lock = Lock()
def worker_sync(points_per_thread):
    global total_hits
    for _ in range(points_per_thread):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            with lock: 
                total_hits += 1
def main():
    global total_hits
    print("=PART 2: The Synchronization Trap=")

    start_time = time.time()
    single_hits = 0
    for _ in range(TOTAL_POINTS):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            single_hits += 1
    single_time = (time.time() - start_time) * 1000
    pi_single = 4.0 * single_hits / TOTAL_POINTS
    print(f"Single Thread: Time = {single_time:.0f} ms, Pi = {pi_single:.5f}")

    total_hits = 0
    start_time = time.time()
    points_per_thread = TOTAL_POINTS // 4
    threads = [
        Thread(target=worker_sync, args=(points_per_thread,)) 
        for _ in range(4)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    sync_time = (time.time() - start_time) * 1000
    pi_sync = 4.0 * total_hits / TOTAL_POINTS
    print(f"Synchronized 4 Threads: Time = {sync_time:.0f} ms, Pi = {pi_sync:.5f}")


if __name__ == "__main__":
    main()