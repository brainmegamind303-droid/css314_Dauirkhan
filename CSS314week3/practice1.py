import random
from threading import Thread

TOTAL_POINTS = 50_000_000
total_hits = 0  

def worker(points_per_thread):
    global total_hits
    hits = 0
    for _ in range(points_per_thread):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            hits += 1
    total_hits += hits


def main():
    global total_hits
    print("PART 1:The Phantom Bug")
    
    for run in range(1, 6):
        total_hits = 0
        points_per_thread = TOTAL_POINTS // 4
        threads = [
            Thread(target=worker, args=(points_per_thread,)) 
            for _ in range(4)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        pi = 4.0 * total_hits / TOTAL_POINTS
        print(f"Run {run}: Pi = {pi:.5f} (Hits: {total_hits})")

if __name__ == "__main__":
    main()