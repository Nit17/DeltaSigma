# Exercises: Python Basics & Complexity

Short, focused practice for each section. Hints and expected complexities included.

## 1) Containers

1. Stable dedupe while preserving order
   - Input: `[3, 1, 3, 2, 1]` → Output: `[3, 1, 2]`
   - Hint: Use a `set` for "seen" and append unseen to a new list.
   - Target: O(n) time, O(n) extra space.

2. Rotate a list right by k
   - Input: `[1,2,3,4,5], k=2` → `[4,5,1,2,3]`
   - Hint: Use slicing or reverse-then-reverse trick.
   - Target: O(n) time, in-place version O(1) extra space.

3. Maintain a sorted list under inserts
   - Insert values one by one, always keeping the list sorted.
   - Hint: `bisect.insort`.
   - Target: Each insert O(n); with a heap you get different trade-offs.

4. Flatten one level of nesting
   - Input: `[[1,2],[3],[4,5]]` → `[1,2,3,4,5]`
   - Hint: Use a simple loop or `itertools.chain.from_iterable`.

## 2) collections and heapq

1. Sliding window maximum using deque
   - Input: `nums=[1,3,-1,-3,5,3,6,7], k=3` → `[3,3,5,5,6,7]`
   - Hint: Store indices; pop from back while current ≥ tail, and pop front if out of window.
   - Target: O(n).

2. Moving average over a stream
   - Maintain a window of size k and compute average at each step.
   - Hint: Use deque for O(1) push/pop, and keep a running sum.
   - Target: O(n).

3. Top-k frequent elements
   - Hint: `collections.Counter` + `heapq.nlargest(k, counter.items(), key=lambda kv: kv[1])`.
   - Target: O(n log k).

## 3) Two Sum

1. Return all index pairs whose values sum to target
   - Input: `[1,3,2,2,4,0,3], target=4` → `[(0,1),(0,6),(2,3),(4,5)]` (order of pairs doesn’t matter)
   - Hint: Map value→indices; for v<c, cross pairs; for v==c, combinations of indices.
   - Target: O(n + a) where a is number of output pairs.

2. Return unique value pairs (no index duplicates), sorted by value
   - Input: `[1,1,2,2,3,3], target=4` → `[(1,3),(2,2)]`
   - Hint: Use a set for values and handle v==c separately.

3. Three Sum (challenge)
   - Return unique triplets that sum to target.
   - Hint: Sort and use two pointers per fixed i. Target: O(n^2).

## 4) ML memory

1. Report memory savings after downcasting (pandas)
   - Compute memory before/after and return a small summary dict.
   - Hint: `df.memory_usage(deep=True).sum()`.

2. Chunked CSV sum (pandas)
   - Read a large CSV in chunks and compute the sum of a numeric column.
   - Hint: iterate chunks; avoid collecting them into a list.

3. Sparse vs dense experiment (numpy/scipy)
   - Build a high-dimensional sparse vector matrix, compute a simple op both dense and sparse, compare memory/time.

Answers: You’ll find worked solutions embedded or hinted within the code files for key tasks (e.g., deque window, all Two Sum pairs).