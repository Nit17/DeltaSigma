# Theory: Python Basics, Complexity, and Memory

This document expands the ideas from the Day1 modules with deeper theory, guidance, and trade‑offs.

## 1) Algorithm analysis and asymptotics

- Big-O (O): Upper bound up to constant factors. Example: Python Timsort is O(n log n) worst-case.
- Big-Theta (Θ): Tight bound; both upper and lower. Example: Summing a list is Θ(n).
- Big-Omega (Ω): Lower bound. Example: Any algorithm that must read all input is Ω(n).
- little-o / little-ω: Strict bounds (non-tight); used less in practice.
- Worst/average/best case: For hash tables, average O(1) lookups; worst-case O(n) if all keys collide.
- Amortized analysis: Average over a sequence of ops. Example: list.append is O(1) amortized despite occasional resizes.
- Space complexity: Memory used as function of input size. Include temporaries and data structure overhead.

Key caveats:
- Asymptotics ignore constants; real timings can favor algorithms with better constants for practical sizes.
- Access patterns and cache locality matter. array/list scans can beat pointer-heavy structures for realistic n.

## 2) Python containers: internals and complexity

- list
  - Dynamic array (contiguous memory). Appends are O(1) amortized; inserts/deletes in the middle are O(n) due to shifts.
  - Indexing is O(1). Membership `x in list` is O(n).
  - sort is Timsort (stable), O(n log n) worst, near O(n) on partially ordered data.
  - Slicing creates new lists (O(k) for slice length k).
- tuple
  - Immutable sequence; same indexing complexity as list. Membership is O(n). Can be a dict/set key if elements are hashable.
- dict
  - Hash table with open addressing; insertion order preserved (Py ≥3.7).
  - Average O(1) for get/set/del; iteration is O(n).
  - Rehashing/resizing happens automatically; individual ops may be slow, but amortized O(1).
  - Hash randomization mitigates adversarial collisions (security).
- set
  - Essentially a dict with keys only. Average O(1) membership/add/remove.
  - No duplicate elements; iteration order is insertion order by accident? No—sets are unordered, but the iteration order is deterministic within a run; don't depend on it.

Practical tips:
- Prefer set/dict for membership tests and lookups.
- Use list when order and index-based access matter, and you modify at the end.
- Prefer deque for O(1) append/pop on both ends.

## 3) Sorting in Python

- Python’s list.sort and sorted are Timsort (stable). Complexity:
  - Worst-case O(n log n); nearly-sorted inputs can approach O(n).
  - Stable: equal keys preserve original order—important for multi-key sorts.
- Key functions and reverse flags are applied efficiently; avoid decorating lists manually unless profiling shows benefit.
- For selecting k best elements:
  - heapq.nsmallest/nlargest: O(n log k)
  - sort then slice: O(n log n) (simple, but slower for small k)

## 4) collections essentials

- Counter: frequency counting; Counter.most_common(k) uses a heap internally.
- defaultdict(list/set): simplifies grouping and accumulation patterns.
- deque: O(1) append/pop both ends; indexing is O(n). Good for queues, sliding windows.
- namedtuple/dataclass: structured records with minimal overhead. namedtuple is immutable and tuple-compatible.
- OrderedDict: keeps insertion order and provides move_to_end. Plain dicts are ordered since 3.7, but lack some APIs.

## 5) Heaps and heapq

- Binary heap stored in a list; parent at i, children at 2i+1 and 2i+2.
- Operations:
  - heapify: O(n)
  - heappush / heappop: O(log n)
  - pushpop/heapreplace: O(log n)
  - nlargest/nsmallest: O(n log k)
- Min-heap by default; use negative values for max-heap behavior or use `heapq._heapify_max` (private; not recommended).
- Heaps are best for streaming top-k problems and priority queues, not for fast membership tests.

## 6) Two Sum: algorithmic trade-offs

- Brute-force: O(n^2) time, O(1) extra space; simplest; fine for tiny n.
- Hash map: O(n) average time, O(n) extra space; fast and practical; handles duplicates naturally.
- Two-pointer (on sorted copy): O(n log n) due to sort; O(1) extra space for the two-pointer pass, but sorting creates a copy if preserving original indices.
- Considerations:
  - Need original indices? Track (value, index) pairs before sorting.
  - Multiple solutions? Define tie-breaking (first found, smallest indices, etc.).
  - Large inputs with memory constraints? Two-pointer may save memory at cost of sorting.

## 7) Memory and data layout for ML pipelines

- Numpy arrays
  - Views vs copies: Slicing often returns views (no copy); operations that change shape/dtype usually copy.
  - Dtypes: Prefer float32/int32 when adequate; halves memory vs float64/int64.
  - Memory order: C-order (row-major) vs Fortran-order (column-major) impacts cache behavior.
  - memmap: Maps a file into memory; enables working with arrays larger than RAM. IO patterns still matter.
- Pandas DataFrames
  - Dtypes drive memory: use downcasting for numerics and `Categorical` for low-cardinality strings.
  - Chunked processing: read/process/write in chunks to cap peak memory.
  - Avoid chained operations that materialize large temporaries; use in-place-like patterns cautiously.
  - GroupBy and joins can be memory-hungry; consider sorting keys and working in partitions.
- Sparse data
  - CSR (row-major) vs CSC (column-major): choose based on row vs column operations.
  - COO is convenient for construction; convert to CSR/CSC for efficient math.
  - Many sklearn estimators support sparse inputs; avoid dense conversions.
- General strategies
  - Stream with iterators/generators; avoid storing all intermediates.
  - Cache expensive intermediate results to disk (e.g., joblib.Memory or parquet files).
  - Profile memory: psutil, tracemalloc; test on representative slices.

## 8) Benchmarking methodology

- Warm up the interpreter/JIT? Python has no JIT by default, but caches and branch prediction affect first runs.
- Use time.perf_counter for timing, discard outliers, repeat and take medians/means.
- Benchmark representative scenarios; micro-benchmarks can mislead due to constant factors and CPU cache effects.
- Keep environments consistent (Python version, libraries, hardware, OS power settings).

## 9) Pitfalls and edge cases

- Hash collisions: In worst-case adversarial inputs, dict/set operations degrade; Python mitigates with hash randomization.
- Floating-point keys: Beware of precision and NaN behavior in dict/set.
- Mutability:
  - Never mutate an object used as a dict key or set element; keys must remain hashable and equal-stable.
  - list/set are mutable; tuple is immutable—but only hashable if all elements are hashable.
- Sorting with mixed types (e.g., ints and strings) raises TypeError in Python 3.

## 10) When to choose what

- Membership tests for large collections: set or dict.
- Maintain sorted order with frequent inserts/deletes: consider bisect + list for small-med n; for heavy ops, use a balanced tree (not in stdlib; use `sortedcontainers` library) or a database index.
- Streaming top-k: heapq.nlargest/nsmallest.
- Fast queue/stack: deque for queue (FIFO), list for stack (LIFO with append/pop end).
- Memory-constrained pipelines: chunked IO, downcasting, categorical dtypes, sparse matrices, memmap.

References to explore further:
- Python docs: Data model, Built-in types, `collections`, `heapq`.
- Timsort paper (Tim Peters): “Timsort: A stable, adaptive, natural mergesort.”
- Wes McKinney: Python for Data Analysis (pandas practices).
