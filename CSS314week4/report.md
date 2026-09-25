# High-Performance & Parallel Computing

## Laboratory Report: Shared-Memory Concurrency & OpenMP Paradigms

### Python (Numba) Edition — Labs 1–5

|  |  |
| --- | --- |
| **Student Name** | \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |
| **Course / Section** | \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |
| **Instructor** | Sufyan bin Uzayr |
| **Date Submitted** | \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |

---

## Section I: System & Hardware Specifications

Fill in this table from your own machine before submitting. On Linux, run `lscpu` and `free -h`; on macOS, `sysctl -a | grep machdep.cpu` and `sysctl hw.memsize`; on Windows, check Task Manager > Performance or run `wmic cpu get *`. Benchmark data in Section III is only meaningful when paired with the exact hardware it was measured on — do not copy specs from a different machine than the one you actually ran the benchmarks on.

| Field | Value |
| --- | --- |
| CPU Model | *e.g. AMD Ryzen 9 7950X / Intel Core i7-13700K / Apple M2 Pro* |
| Physical Core Count | *fill in* |
| Logical Hardware Thread Count (incl. SMT/Hyper-Threading) | *fill in* |
| L1 Cache Size (per core) | *fill in, e.g. 32 KB data + 32 KB instr.* |
| L2 Cache Size (per core) | *fill in* |
| L3 Cache Size (shared) | *fill in* |
| System RAM | *fill in* |
| Operating System & Version | *fill in* |
| Python Version | `python3 --version` |
| NumPy Version | `python3 -c "import numpy; print(numpy.__version__)"` |
| Numba Version | `python3 -c "import numba; print(numba.__version__)"` |

> **Note:** benchmarks for this report were developed and functionally verified in a single-core (1 logical CPU) sandbox during preparation, purely to confirm the code runs correctly and produces sensible output. That environment cannot produce valid multi-core scaling data — all thread-count sweeps and speedup/efficiency figures in Section III must be regenerated on your own multi-core hardware.

---

## Section II: Experimental Methodology

### Timing Mechanism

All wall-clock measurements use Python's `time.perf_counter()`, a monotonic, high-resolution performance counter unaffected by system clock adjustments — the same mechanism the manual specifies for cross-language comparison against Java's `System.nanoTime()`. Each timed region is bracketed by a `t0 = time.perf_counter()` / `t1 = time.perf_counter()` pair immediately around the code under test, excluding any setup (array allocation, data generation) that is not itself part of the benchmarked operation.

### JIT Warm-Up

Every Numba `@njit` / `@njit(parallel=True)` function is called once on a small, throwaway input before any timed run. Numba compiles a function to native machine code on first invocation (or first invocation with a new argument type signature); without this warm-up call, the first "timed" run would include one-time JIT compilation latency (often hundreds of milliseconds to seconds) that has nothing to do with the algorithm's actual runtime performance, and would badly distort speedup and scaling comparisons.

### Trials and Averaging

Each timing configuration (a given thread count P, chunk size C, or cutoff K) is run multiple times and the mean is reported, per the manual's task instructions: 5 trials for Lab 1's oversubscription sweep and Lab 2's scaling sweep, 3 trials for Lab 3's 4×4 scheduling matrix, and single representative runs for Lab 4's already-long (100,000,000-iteration) false-sharing benchmarks and Lab 5's cutoff sweep, where per-configuration runtime is already on the order of seconds and thread/queue contention noise is comparatively small relative to total runtime. If your own results show high run-to-run variance, increase the trial count and report both mean and standard deviation.

### Thread Count Configuration

Numba's `parallel=True` backend reads its thread pool size from the `NUMBA_NUM_THREADS` environment variable at process start, but can also be changed at runtime via `numba.set_num_threads(p)` before a given timed call (used in Lab 2's scaling sweep and recommended for Labs 3–4 if you extend them similarly). Before running any sweep, set `OMP_NUM_THREADS` and `NUMBA_NUM_THREADS` to your machine's physical core count, and never request more threads in a sweep than your machine's logical thread count — doing so intentionally exercises the oversubscription behavior studied in Lab 1, Task 1.2, and should be labeled as such rather than mixed into "normal" scaling data.

### Noted Deviations from the Manual's Starter Code

- **Lab 2, Tasks 2.1/2.2:** implemented with plain `threading.Thread` on a Python-level accumulator rather than an `@njit` function, because Numba's `nogil` parallel regions do not reproduce a classic unprotected load/add/store race the way genuinely GIL-interleaved Python threads do; this keeps the race-condition and critical-section demonstrations honest to what the questions are actually asking about.
- **Lab 3:** `schedule(dynamic, chunk)` is implemented as an explicit shared row-index counter guarded by a `threading.Lock` (mirroring the manual's Java `AtomicInteger` work-dispenser pattern), since Numba's `prange` offers only a fixed internal chunking strategy and no dynamic/guided scheduling knob.
- **Lab 5:** parallel (process-pool-based) recursion is capped at a fixed depth (`max_depth=3` by default), falling back to ordinary sequential recursion below that depth. The manual's original template creates a new `ProcessPoolExecutor` at *every* single recursive call; because spawning an OS process costs tens of milliseconds (vs. microseconds or less for a genuine OpenMP task or Java `ForkJoinTask`), sweeping the cutoff K down to small values without this cap creates thousands of process pools and can hang the benchmark or exhaust OS process limits. This was confirmed directly while preparing this report and is documented in the `lab5_merge_sort.py` source file.

---

## Section III: Empirical Results & Visualizations

Run each lab's script (see the accompanying source code / README) on your own multi-core machine. Each script writes its own CSV file(s) and PNG plot(s) directly into its lab folder. Paste your measured values into the tables below, and embed each corresponding PNG as a figure immediately beneath its table (e.g. `![Task 1.2](lab1/lab1_oversubscription.png)`).

### Lab 1: The Fork-Join Model, Team Creation, and Thread Scoping

#### Task 1.1 — Non-Determinism (10 runs)

*Raw stdout for 10 runs is saved to lab1_nondeterminism.txt. Summarize the variation you observed (e.g., how often the print order changed) in a sentence or two below the table rather than pasting all 10 raw runs.*

| Run # | Thread ID print order observed |
| --- | --- |
| 1 |  |
| 2 |  |
| 3 |  |
| 4 |  |
| 5 |  |

> *\[ Insert a short excerpt of lab1_nondeterminism.txt, or a summary paragraph here \]*

#### Task 1.2 — Thread Oversubscription Sweep

| Thread Team Size (P) | Mean Fork+Join Time (ms) |
| --- | --- |
| 1 |  |
| 2 |  |
| 4 |  |
| 8 |  |
| 16 |  |
| 32 |  |
| 64 |  |

> *\[ Insert lab1_oversubscription.png here \]*

#### Task 1.3 — CPU Core Saturation

| Thread Count (P) | Wall-Clock Time (s) | Observed CPU Utilization |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 4 |  |  |
| 8 |  |  |

> *\[ Insert a screenshot of your system monitor during the P = physical-core-count run here \]*

### Lab 2: Numerical Integration (Pi Approximation) & Parallel Reductions

#### Task 2.1 — Race Condition Quantification

| Threads (P) | Computed Pi | Absolute Error vs. true π |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 4 |  |  |
| 8 |  |  |

> *\[ Insert lab2_race_condition.csv rendered as a table or bar chart of error vs. P here \]*

#### Task 2.2 — Critical Section Overhead

| Variant | Time (s) | Overhead vs. Serial Baseline |
| --- | --- | --- |
| Serial baseline |  | — |
| Critical section (N = 1,000,000) |  |  |

> *\[ Insert a short note on the % overhead computed from lab2_critical_overhead.csv here \]*

#### Task 2.3 / 2.4 — Strong Scaling, Speedup S(P), Efficiency E(P)

| P | Mean Time (s) | Speedup S(P) | Efficiency E(P) |
| --- | --- | --- | --- |
| 1 |  | 1.00 | 1.00 |
| 2 |  |  |  |
| 4 |  |  |  |
| 8 |  |  |  |
| 16 |  |  |  |

> *\[ Insert lab2_speedup.png here \]*

### Lab 3: Work-Sharing & Loop Scheduling Policies (Mandelbrot Fractal)

#### Task 3.2 / 3.3 — 4×4 Scheduling Matrix (Threads × Chunk Size)

| P \\ C | 1 | 16 | 64 | 256 |
| --- | --- | --- | --- | --- |
| 2 |  |  |  |  |
| 4 |  |  |  |  |
| 8 |  |  |  |  |
| 16 |  |  |  |  |

> *\[ Insert lab3_heatmap.png here \]*

#### Task 3.4 — Load Imbalance

| Thread ID | Total Iterations Computed |
| --- | --- |
| 0 |  |
| 1 |  |
| 2 |  |
| 3 |  |
| Imbalance = (Max−Min)/Avg |  |

> *\[ Insert a short paragraph interpreting lab3_imbalance.csv here \]*

### Lab 4: Memory Hierarchy, Cache Coherency, and False Sharing

#### Task 4.1 / 4.2 / 4.3 — Unpadded vs. Padded vs. Thread-Local Scaling

| P | Unpadded (s) | Padded (s) | Thread-Local (s) |
| --- | --- | --- | --- |
| 1 |  |  |  |
| 2 |  |  |  |
| 4 |  |  |  |
| 8 |  |  |  |
| 16 |  |  |  |

> *\[ Insert lab4_scaling.png here \]*

#### Task 4.4 — Hardware Performance Counters (Linux only)

*Run: perf stat -e L1-dcache-load-misses,L1-dcache-store-misses python3 lab4_false_sharing.py*

| Variant | L1-dcache-load-misses | L1-dcache-store-misses |
| --- | --- | --- |
| Unpadded |  |  |
| Padded |  |  |

> *\[ Insert raw perf stat output (or a note that perf was unavailable on your platform) here \]*

### Lab 5: Recursive Task-Based Parallelism (Parallel Merge Sort)

#### Task 5.1 — Correctness Verification

| Check | Result |
| --- | --- |
| Sorted output matches Python's sorted() | Pass / Fail |

> *\[ Insert console output confirming the assertion passed here \]*

#### Task 5.2 — Cutoff Threshold Sweep (N = 5,000,000)

| Cutoff K | Execution Time (s) |
| --- | --- |
| 1 |  |
| 10 |  |
| 100 |  |
| 1,000 |  |
| 10,000 |  |
| 50,000 |  |
| 100,000 |  |

> *\[ Insert lab5_cutoff_sweep.png here \]*

#### Task 5.3 — Work-Span Analysis

| Quantity | Value |
| --- | --- |
| Measured Work T1 (sequential, s) |  |
| Derived Span T_infinity | O(N) |
| P_theoretical = T1 / T_infinity | ≈ log2(N) |

> *\[ Insert lab5_work_span.txt (full derivation) here \]*

---

## Section IV: Analytical & Discussion Responses

Review each answer below and personalize it with references to your own measured data from Section III where relevant (e.g., "this matches what I observed in Task 1.2, where...") — this is both stronger analytical writing and important for the honor code's individual-work requirement.

### Lab 1: The Fork-Join Model, Team Creation, and Thread Scoping

#### Question 1.1 (Scheduling)

Thread IDs print in non-sequential order across consecutive runs because the operating system's kernel scheduler, not the program, decides when each ready thread actually receives CPU time. Once the four worker threads are created, they are all placed in a "ready" state and the OS scheduler (e.g., Linux's Completely Fair Scheduler) assigns them to available logical cores based on load balancing, priority, and the precise timing of when each thread becomes runnable. Small, essentially random variations in thread-creation latency, cache/branch-predictor warm-up state, and pending interrupts on each core mean the relative completion order of the print statements is not guaranteed to match logical rank order. The two architectural components responsible are (1) the OS kernel's thread/process scheduler, which controls dispatch order and time-slicing across logical cores, and (2) the CPU's own microarchitectural state (per-core caches, pipeline occupancy, out-of-order execution) which makes the actual wall-clock duration of "identical" work on different cores slightly unequal run to run.

#### Question 1.2 (Oversubscription)

When the requested thread count P exceeds the number of physical hardware cores (or hardware threads under SMT), performance degrades due to three compounding effects. First, context switching: the OS must repeatedly save and restore each thread's full execution context (register file, program counter, stack pointer) as it time-slices multiple software threads across fewer physical execution units, and each switch costs hundreds to thousands of CPU cycles of pure overhead with no useful work performed. Second, thread state preservation: switching a thread off a core evicts its working set from that core's registers and often its L1/L2 cache, so when the thread is rescheduled it must re-warm its cache lines from L3 or main memory before resuming productive work. Third, cache thrashing: with more active threads than cores, multiple unrelated working sets compete for the same limited cache capacity, causing threads to repeatedly evict each other's useful data, driving up cache miss rates and effective memory latency for everyone. Together, these mean wall-clock time increases superlinearly once P exceeds the physical core count, even though total CPU work is unchanged.

#### Question 1.3 (Barriers)

The implicit barrier at the end of an OpenMP parallel region forces every thread in the team to reach that point before any of them is allowed to proceed into the code that follows. Its role is twofold: it establishes a synchronization point that guarantees all writes made by every thread inside the parallel region are complete and visible (a memory-consistency guarantee), and it ensures the program returns to a known, single-threaded state before continuing serial execution. Without this barrier, a "fast" worker thread could race ahead into subsequent code while other threads are still mid-computation, reading shared variables that are only partially updated, working with stale or torn values, or triggering data races on results that other threads haven't finished producing. This would make program output non-deterministic and could produce results that depend on which threads happen to finish first — a serious correctness hazard, not just a performance one.

#### Question 1.4 (Architectural Mapping)

A hardware execution thread (Hyper-Threading / SMT) is a physical CPU feature in which a single physical core duplicates just enough architectural state (register file, program counter) to track two or more independent instruction streams, while sharing the core's actual execution units, caches, and pipeline resources between them; it improves throughput by filling execution-unit idle cycles (e.g., during a cache miss on one stream) with work from the other stream, but it is not equivalent to a second full core — speedup from SMT is typically well under 2x. An operating-system kernel thread is a software abstraction maintained by the OS: it has its own stack, register-save area, and scheduling metadata, and the OS scheduler is responsible for mapping the (potentially larger) set of kernel threads onto the (smaller) set of available hardware threads/cores via preemptive, priority-based time-slicing. A language-level green/virtual thread (e.g., Java's Project Loom virtual threads, Go's goroutines) is a further software abstraction layered on top of kernel threads: a language runtime multiplexes many (potentially millions of) lightweight logical threads onto a much smaller pool of OS kernel threads, using cooperative or M:N scheduling to avoid the cost of an OS-level context switch for every logical thread switch. The three form a hierarchy — many green threads may map to few kernel threads, which are in turn scheduled across a fixed number of hardware threads/cores — and performance characteristics differ substantially at each layer.

### Lab 2: Numerical Integration (Pi Approximation) & Parallel Reductions

#### Question 2.1 (Memory Incoherency)

A non-atomic read-modify-write on a shared accumulator decomposes, at the assembly level, into three separate steps: LOAD (copy the current value from memory into a register), ADD (increment the register value), and STORE (write the register back to memory). If two threads interleave these steps — for example, Thread A executes LOAD and reads value V, then Thread B also executes LOAD and reads the same value V before A has stored its result, then both threads independently ADD their term and STORE — whichever thread stores last overwrites the other's update entirely. The net effect is that one of the two increments is silently lost; the shared sum only reflects one thread's contribution even though both threads performed work. Because this non-deterministic interleaving happens on essentially every one of the millions of loop iterations, and the probability of at least one lost update per thread grows with both iteration count and thread count, the measured Pi value's error grows systematically worse as more threads are added to the naive, unsynchronized implementation.

#### Question 2.2 (Reduction Trees)

A binary reduction tree achieves O(log P) scaling because each of the P threads first accumulates its own private, contention-free partial sum over its assigned iterations (this phase is embarrassingly parallel and needs no synchronization at all), and only the P private partial sums must then be combined. That combination is performed as a balanced binary tree: pairs of partial sums are combined in parallel in round 1 (P/2 independent combine operations), pairs of those results in round 2 (P/4 operations), and so on, requiring only ceil(log2 P) sequential rounds regardless of how large P grows. A centralized critical section, by contrast, forces every single accumulation operation from every thread through one mutually-exclusive lock, so the P threads' updates are fully serialized — there is zero parallelism during the accumulation phase, and as P grows, contention on that one lock (and the associated cache-line bouncing on the lock variable and the shared sum) grows linearly, making the critical-section variant's overhead scale as O(P) rather than O(log P). At scale, the reduction tree exploits parallelism during both computation and combination, while the critical section only ever exploits it during computation and then throws it away during the strictly serial combination phase.

#### Question 2.3 (Amdahl's Law)

By Amdahl's Law, if 5% of a program's execution time is strictly serial and cannot be parallelized, the maximum theoretical speedup with an infinite number of processors is S_max = 1 / 0.05 = 20x. If the measured speedup in your own data flattens out well before reaching 20x, several secondary architectural factors beyond the idealized Amdahl model are typically responsible: memory-bandwidth saturation (all cores contend for the same shared memory bus/interconnect, so beyond some P, cores stall waiting for data rather than computing); cache and false-sharing effects that add synchronization/coherence traffic not modeled by Amdahl's simple serial/parallel split; thread creation, scheduling, and (for the reduction combine step) synchronization overhead, which grows with P and is treated as zero-cost overhead in the idealized model; hardware constraints such as SMT/Hyper-Threading sharing execution resources within a physical core rather than providing a full extra core; and thermal/power throttling under sustained multi-core load. In practice, the "5% serial" figure itself is often optimistic — real serial overhead (I/O, thread management, final reduction combine) tends to be larger than the theoretical/algorithmic serial fraction, further lowering the practically achievable ceiling.

#### Question 2.4 (Hardware Primitives)

Modern 64-bit multi-core processors implement atomic memory updates primarily through cache-coherence-based locking rather than locking the entire system memory bus. On x86-64, a LOCK-prefixed instruction (e.g., LOCK ADD, or CMPXCHG for compare-and-swap) causes the core to acquire exclusive (Modified/Exclusive) ownership of the relevant 64-byte cache line via the MESI (or MOESI) cache-coherence protocol — any other core's cached copy of that line is invalidated via a coherence snoop/broadcast, and the requesting core is guaranteed to hold sole write ownership of the line for the duration of the read-modify-write, so no other core can observe or interleave with a partial update. This is often called "cache locking" and is far cheaper than the legacy full bus lock older processors used. ARM and other RISC-style architectures instead commonly use a LOAD-LINKED / STORE-CONDITIONAL (LL/SC) pair: the LOAD-LINKED instruction reads a value and places a hardware "reservation" on that address (tracked via the coherence protocol), and the subsequent STORE-CONDITIONAL only succeeds if no other core has written to that address since the reservation was placed — otherwise it fails and the software retries the LL/SC pair in a loop. Both approaches ultimately rely on the same underlying mechanism: the cache-coherence protocol's ability to grant one core exclusive, momentarily uninterruptible ownership of a cache line, and to detect (via coherence snooping) when another core has interfered.

### Lab 3: Work-Sharing & Loop Scheduling Policies (Mandelbrot Fractal)

#### Question 3.1 (Fine-Grained Contention)

Dynamic scheduling with chunk size C = 1 performs poorly despite offering theoretically ideal load balancing because every single loop iteration (one Mandelbrot row) requires a separate trip to the shared work-dispenser to claim the next unit of work. Each of those trips requires acquiring a lock (or performing an atomic fetch-and-add on a shared counter), and with C = 1 the number of such synchronization operations equals the total iteration count — for a 1080-row image this means 1080 lock acquisitions competing across however many threads are running. The specific hardware resource being contested is the cache line holding the shared work-queue index (the lock variable or AtomicInteger): every thread's attempt to claim work causes that cache line to bounce between cores' caches via the coherence protocol (similar in kind to the false-sharing mechanism studied in Lab 4), and this cache-line-bouncing/lock-contention overhead, repeated once per iteration, ends up dwarfing the actual per-row computational work, especially for rows that escape quickly (2–5 iterations of the Mandelbrot recurrence).

#### Question 3.2 (Spatial Imbalance)

Under static scheduling, the threads whose contiguous row blocks fall near the vertical center of the rendered image become the stragglers. This is a direct consequence of the geometry of the Mandelbrot set in the complex plane: the set itself, and the densest region of high-iteration-count "boundary" points, is concentrated around the origin and the real axis, so rows near the vertical middle of a centered render contain a much higher proportion of pixels that require close to the full 1000-iteration budget before escaping (or that never escape at all). Rows near the top and bottom edges of the image correspond to points far from the set, which typically escape after only a handful of iterations. Because static scheduling assigns fixed, contiguous row ranges to each thread without regard to their eventual cost, whichever thread's block overlaps the central, computationally dense rows finishes far later than threads assigned to the sparse, cheap outer rows, and the whole team must wait at the implicit barrier for that one straggler.

#### Question 3.3 (Guided Policy)

OpenMP's schedule(guided, chunk) assigns iterations in a series of shrinking chunks: at the start, each chunk handed out is proportional to the remaining unassigned iterations divided by the number of threads (a large chunk, similar in spirit to static scheduling), and after each chunk is dispensed, the size of the next chunk is reduced — typically following an exponential/geometric decay — down to a specified minimum chunk size. This combines the strengths of both extremes: early on, when there is plenty of remaining work and thread completion times haven't yet diverged, large chunks are handed out, minimizing the number of synchronized trips to the shared work queue (the same low-overhead property that makes static scheduling efficient). As the loop nears completion and any load imbalance that has developed becomes more consequential, the chunks shrink, allowing idle threads to grab progressively smaller units of remaining work and finish closer together in time — the same fine-grained adaptivity that makes dynamic scheduling load-balanced. The net effect is scheduling overhead close to static scheduling's, with load-balancing quality close to dynamic scheduling's.

#### Question 3.4 (Engineering Decision Framework)

A practical rule of thumb: use static scheduling when loop iterations have uniform or highly predictable per-iteration cost, since it incurs essentially zero runtime scheduling overhead and gives the best data locality/cache behavior — this is the right default whenever you can reasonably guarantee balanced work. Use dynamic scheduling when iteration cost is highly non-uniform and cannot be predicted or computed in advance (as with the Mandelbrot escape-time algorithm), accepting the ongoing synchronization overhead of the shared work queue in exchange for load balance; when using dynamic scheduling, tune the chunk size to be as large as possible while still achieving acceptable balance, since larger chunks amortize the fixed per-claim synchronization cost over more work. Use guided scheduling as a strong default choice for workloads with moderate-to-significant, but not extreme, non-uniformity, or whenever you are unsure what static chunk size would balance the load well — it requires no manual chunk-size tuning to get reasonable performance and adapts automatically as the loop progresses.

### Lab 4: Memory Hierarchy, Cache Coherency, and False Sharing

#### Question 4.1 (MESI State Transitions)

Consider a single 64-byte cache line holding four adjacent 64-bit (8-byte) integer counters, with Core 0 "owning" counter\[0\] and Core 1 "owning" counter\[1\]. Initially the line is Invalid in both caches (not yet loaded). When Core 0 first writes to counter\[0\], it issues a Read-For-Ownership; since no other cache holds the line, Core 0's cache brings it in and transitions the line directly to Modified (M) — Core 0 has the sole, up-to-date, dirty copy. When Core 1 then writes to counter\[1\] (a logically different 8-byte word within the same line), Core 1 must also gain exclusive ownership: it broadcasts a Read-For-Ownership request, which the coherence protocol snoops; Core 0's copy is invalidated (M → Invalid) and, because it was dirty, its contents are flushed either to Core 1 directly or through the shared last-level cache/memory, and Core 1's line transitions to Modified. If Core 0 subsequently writes to counter\[0\] again, the same invalidate-and-flush cycle repeats in reverse, evicting Core 1's now-dirty copy. Because both cores keep writing (to their own, logically independent words) in alternation, the single physical cache line "ping-pongs" between the Modified state in Core 0's cache and the Modified state in Core 1's cache, with a full cache-line invalidation, flush, and reload required on every single alternation — despite neither core ever reading or writing the other's data.

#### Question 4.2 (Bus Bouncing)

Adding more CPU cores makes the unpadded test slower than a single thread because every core's increment to its own counter forces the shared 64-byte cache line to be invalidated in every other core's cache and re-fetched before the next write can proceed — with more cores actively writing to counters on the same line, these invalidate/re-fetch cycles happen far more frequently within the same wall-clock window, so the effective time-per-increment becomes dominated by waiting for cache-coherence traffic (invalidation broadcasts, dirty-line flushes, and reloads across the shared interconnect / last-level cache) rather than by the near-instant cost of an ALU increment. The physical bottleneck is therefore not any compute unit at all — it is the shared cache-coherence interconnect and the associated coherence-protocol messaging bandwidth/latency between cores, which becomes saturated and serializes what should have been independent, parallel work.

#### Question 4.3 (True vs. False Sharing)

True sharing occurs when multiple threads genuinely read and/or write the identical memory location — the same logical variable — concurrently; this is a real data-race/correctness concern that requires explicit synchronization (a lock, an atomic operation, or a reduction pattern) to produce a correct result, and the associated cache-coherence traffic is a necessary cost of that correctness. False sharing occurs when threads access entirely different, logically independent variables that simply happen to be laid out within the same hardware cache-line granularity (commonly 64 bytes); the program is fully correct and free of any data race, because no thread ever touches another thread's variable — but the cache-coherence protocol operates at the granularity of whole cache lines, not individual bytes, so it cannot distinguish "two different 8-byte words in this line" from "the same 8-byte word," and it triggers the same invalidate/reload traffic regardless. False sharing is therefore purely a performance hazard (fixed by data layout / padding), never a correctness hazard.

#### Question 4.4 (Adjacent Line Prefetching)

The JVM's @jdk.internal.vm.annotation.Contended annotation enforces 128 bytes of padding, rather than exactly 64, because many modern x86-64 processors implement an adjacent-line (spatial/streamer) hardware prefetcher: whenever one 64-byte cache line is accessed, the prefetcher speculatively also fetches the neighboring 64-byte line, effectively treating the pair of lines as a single 128-byte coherence-relevant unit in practice. If a hot, contended field were padded to occupy exactly one 64-byte line, the prefetcher could still opportunistically pull in the adjacent line — which might hold another thread's equally hot, contended field — into the same core's cache alongside it, re-introducing false-sharing-style invalidation traffic between the two "separated" fields even though they technically sit in different architectural cache lines. Padding to a full 128 bytes guarantees that a contended field's line and the adjacent-prefetched line around it never overlap with another thread's contended field, providing a safety margin against this prefetching behavior across the range of x86-64 microarchitectures the JVM targets.

### Lab 5: Recursive Task-Based Parallelism (Parallel Merge Sort)

#### Question 5.1 (Granularity Breakdown)

With a sequential cutoff of K = 1, task creation is driven all the way down to single-element base cases, meaning the recursion generates on the order of N independent task objects for an algorithm whose total useful work is only O(N log N) comparisons. In a real OpenMP or Java ForkJoinTask runtime, each task still carries non-trivial fixed overhead — allocating a task descriptor, a stack frame or continuation record, and performing a push/pop against a work-stealing deque — typically on the order of tens to hundrame of nanoseconds to low microseconds per task; when there are \~N such tasks and each does O(1) work, that fixed per-task overhead can end up dominating the actual computation by one or more orders of magnitude. The effect is far more severe in this lab's Python process-pool-based implementation, because each "task" there corresponds to spawning an entire new OS process (interpreter start-up, memory-space setup, IPC pipe creation) — costing tens of milliseconds rather than nanoseconds — so a K = 1 sweep without some form of parallel-recursion-depth cap can attempt to create thousands to millions of OS processes, hang the benchmark, and exhaust the operating system's process-table or file-descriptor limits. (This is precisely why the accompanying source code caps parallel/process-based recursion at a fixed depth and falls back to sequential recursion below it — see the lab5_merge_sort.py docstring.)

#### Question 5.2 (Work-Stealing Mechanics)

A work-stealing scheduler gives every worker thread its own private double-ended queue (deque) of pending tasks. When a thread spawns a new child task, it pushes that task onto the head of its own local deque; when the thread finishes its current work and needs something new to do, it first pops from the head of its own deque (LIFO order) — this favors tasks that were spawned most recently and are therefore likely to still be cache-warm, and requires no synchronization with other threads since only the owner touches its own head. When a thread's local deque runs empty, it becomes a "thief": it picks another worker's deque (often at random or round-robin) and steals a task from the tail of that deque (FIFO end) rather than the head. Stealing from the opposite end the owner operates on minimizes contention between the owning thread (working at the head) and any thieves (working at the tail), and because tasks near the tail of a deque tend to be the oldest, coarsest-grained tasks spawned earliest in the recursion, a successful steal typically captures a large chunk of remaining work, amortizing the cost of the steal operation itself.

#### Question 5.3 (Span Bottlenecks)

In the standard recursive merge sort, the two halves of the array are sorted independently and in parallel (an O(log N)-deep recursion tree), but merging those two already-sorted halves back together is implemented as a strictly sequential O(N) linear scan through both halves. Because the merge step at any given level of the recursion cannot begin until both of its children's results are fully available, and the merge itself has no parallelism, the length of the longest dependency chain (the Span, T_infinity) through the computation is dominated by that top-level O(N) merge — giving Span = O(N) overall, even though the total Work is O(N log N). This caps the theoretical parallelism (Work / Span) at O(log N), meaning throwing more than roughly log2(N) processors at this particular implementation yields diminishing returns. The merge step itself can be parallelized to remove this bottleneck: given two sorted subarrays, a parallel merge can locate a partition point in the second subarray (via binary search) that corresponds to the median element of the first subarray, split both subarrays at their respective partition points into two pairs of sub-ranges, and then merge each of the two resulting pairs independently and recursively in parallel — this reduces the merge's own span to O(log N), bringing the overall algorithm's span down to O(log^2 N) and raising the theoretical parallelism back up toward O(N / log N).

#### Question 5.4 (Loop vs. Tasking Paradigm)

OpenMP loop work-sharing (#pragma omp for) is built for problems with a regular, statically (or near-statically) determinable iteration space and, ideally, roughly uniform per-iteration cost — the compiler and runtime can partition a known range of loop indices across threads cheaply and efficiently up front. Dynamic tasking (#pragma omp task) is fundamentally superior whenever the shape or amount of parallel work is irregular, recursive, or only discovered at runtime: divide-and-conquer algorithms such as merge sort, quicksort, and FFT; traversal of trees and graphs with unpredictable branching factors; pointer-chasing over dynamically linked data structures; and any problem where the total number of independent units of work is not known until execution reaches that point in the program. Loop constructs cannot naturally express a recursive fork/join pattern with a data-dependent depth, whereas task-based models let the program simply spawn a task wherever independent work is discovered and rely on a work-stealing runtime to dynamically load-balance those irregular subtrees of work across available threads — something a fixed, up-front loop partition fundamentally cannot do.

---

## Section V: Conclusions & Insights

Across Labs 1 through 5, several unifying themes about shared-memory concurrency emerge. First, correctness and performance are separable concerns that both demand explicit engineering attention: a program can be perfectly data-race-free (as in Lab 4's false-sharing benchmark) and still suffer severe performance degradation purely from hardware-level cache-coherence effects, while a program can look correct on paper and still silently corrupt results under unsynchronized concurrent access (Lab 2's naive race condition). Second, synchronization is never free: every mechanism studied here — locks, critical sections, atomic operations, barriers, and shared work queues — trades correctness or load-balance guarantees for measurable overhead, and the central engineering skill this course develops is choosing the coarsest-grained synchronization mechanism that still satisfies correctness and load-balance requirements, because coarser granularity almost always means lower overhead (Lab 2's reduction tree vs. critical section; Lab 3's chunk-size tuning; Lab 5's cutoff threshold). Third, the memory hierarchy is not a passive backdrop to parallel performance but an active participant: Lab 4 demonstrated that a program's logical structure can be entirely correct while its physical data layout single-handedly determines whether the program scales or actively gets slower as cores are added. Finally, different problem shapes call for fundamentally different work-distribution strategies — the regular, iteration-count-known loops in Labs 1–2 are well served by OpenMP-style loop work-sharing and reduction constructs, the highly non-uniform Mandelbrot workload in Lab 3 required dynamic or guided scheduling to avoid straggler threads, and the irregular, recursive structure of merge sort in Lab 5 was fundamentally a better fit for task-based parallelism than for any loop construct.

> *\[ Personalize this section with 2–3 sentences reflecting on your own specific measured results — e.g., what speedup ceiling you actually observed vs. Amdahl's prediction, which lab's scaling behavior most surprised you, and what you would try next (e.g., parallel merge in Lab 5, or NUMA-aware placement) if you extended this work further. \]*