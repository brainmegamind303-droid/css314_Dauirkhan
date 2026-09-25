# OpenMP Lab Practice Manual — Python (Numba) Source Code

Each `labN/` folder contains one self-contained script implementing that
lab's required benchmarks (Python / Numba track).

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install numpy numba matplotlib
export OMP_NUM_THREADS=<your physical core count>
export NUMBA_NUM_THREADS=<your physical core count>
```

## Run each lab

```
cd lab1 && python3 lab1_forkjoin.py
cd ../lab2 && python3 lab2_pi_reduction.py
cd ../lab3 && python3 lab3_mandelbrot.py
cd ../lab4 && python3 lab4_false_sharing.py
cd ../lab5 && python3 lab5_merge_sort.py
```

Each script writes its own CSV data + PNG plots into its folder — these are
exactly the files you need for Section III of the lab report. Sweep ranges,
N, and iteration counts are set to the manual's defaults but are called out
in each file's docstring if you need to scale them down while debugging.

## Notes / deviations from the manual's raw starter code

- **Lab 2, Task 2.1/2.2** use plain `threading.Thread` on a Python-level
  accumulator rather than Numba `@njit` functions, because Numba's `nogil`
  parallel regions don't reproduce a classic unprotected-read-modify-write
  race the way raw Python threads (racing under, then released from, the
  GIL) do. This keeps the race-condition demonstration honest.
- **Lab 2, Task 2.3/2.4** sweeps thread count via `numba.set_num_threads(p)`
  before each timed call rather than relaunching the process per P value.
- **Lab 3** implements `schedule(dynamic, chunk)` with an explicit
  `threading` + shared-counter work queue (mirroring the Java
  `AtomicInteger` pattern in the manual), since Numba's `prange` has no
  built-in dynamic/guided scheduling knob.
- **Lab 5** caps *parallel* (process-pool) recursion depth at `max_depth=3`
  — see the docstring in `lab5_merge_sort.py` for why the manual's original
  "new `ProcessPoolExecutor` per recursive call" design does not scale down
  to small cutoff values (K=1, K=10) without this cap.

## Hardware disclosure

Run these on your own machine, not a shared/virtualized grading container —
thread-scaling results are only meaningful with genuine multi-core hardware.
Record your CPU model, physical core count, logical thread count, and
L1/L2/L3 cache sizes for Section I of the report (e.g. on Linux:
`lscpu`; on macOS: `sysctl -a | grep machdep.cpu`; on Windows:
Task Manager > Performance, or `wmic cpu get *`).
