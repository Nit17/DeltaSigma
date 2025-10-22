"""
03-two-sum: Implementations and a tiny benchmark helper.

Functions:
- two_sum_bruteforce: O(n^2)
- two_sum_hash: O(n) average
- two_sum_two_pointers: O(n log n) on sorted copy

Theory: See Day1/THEORY.md §6 for trade-offs (time/space, duplicates, index retention,
and when to prefer sorting vs hashing under memory constraints).

Run this file to see quick correctness checks and a small benchmark.
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Dict, List
import random
import time


def two_sum_bruteforce(nums: Sequence[int], target: int) -> Optional[Tuple[int, int]]:
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return i, j
    return None


def two_sum_hash(nums: Sequence[int], target: int) -> Optional[Tuple[int, int]]:
    seen: Dict[int, int] = {}
    for j, x in enumerate(nums):
        need = target - x
        if need in seen:
            return seen[need], j
        seen[x] = j
    return None


def two_sum_two_pointers(nums: Sequence[int], target: int) -> Optional[Tuple[int, int]]:
    with_idx = [(v, i) for i, v in enumerate(nums)]
    with_idx.sort(key=lambda x: x[0])
    lo, hi = 0, len(with_idx) - 1
    while lo < hi:
        s = with_idx[lo][0] + with_idx[hi][0]
        if s == target:
            i, j = with_idx[lo][1], with_idx[hi][1]
            return (i, j) if i < j else (j, i)
        if s < target:
            lo += 1
        else:
            hi -= 1
    return None


def two_sum_all_pairs(nums: Sequence[int], target: int) -> List[Tuple[int, int]]:
    """Return all unique index pairs (i, j), i < j, such that nums[i] + nums[j] == target.

    Strategy:
    - Build value -> list of indices
    - For each unique value v, let c = target - v
      - If v < c: cross product of indices[v] x indices[c]
      - If v == c: all index combinations within indices[v]
    Returns pairs sorted lexicographically.
    Complexity: O(n + a) where a is number of output pairs.
    """
    from collections import defaultdict

    idxs: Dict[int, List[int]] = defaultdict(list)
    for i, v in enumerate(nums):
        idxs[v].append(i)

    pairs: List[Tuple[int, int]] = []
    seen_vals = set()
    for v in list(idxs.keys()):
        if v in seen_vals:
            continue
        c = target - v
        if c not in idxs:
            seen_vals.add(v)
            continue
        if v < c:
            for i in idxs[v]:
                for j in idxs[c]:
                    pairs.append((i, j) if i < j else (j, i))
            # processed both v and c
            seen_vals.add(v)
            seen_vals.add(c)
        elif v == c:
            lst = idxs[v]
            m = len(lst)
            for a in range(m):
                for b in range(a + 1, m):
                    i, j = lst[a], lst[b]
                    pairs.append((i, j))
            seen_vals.add(v)
        else:
            # v > c: defer processing to when we encounter c (the smaller value)
            seen_vals.add(v)

    pairs = sorted(set(pairs))
    return pairs


def bench_two_sum(sizes: List[int] = [1000, 5000], reps: int = 2) -> List[Dict[str, float]]:
    """Return a list of timing dicts for each input size.

    - Skips brute-force for sizes > 3000 by setting it to None
    """
    results: List[Dict[str, float]] = []
    rng = random.Random(42)
    for n in sizes:
        nums = [rng.randrange(n * 2) for _ in range(n)]
        target = nums[n // 3] + nums[2 * n // 3]
        row: Dict[str, float] = {"n": float(n)}

        # bruteforce (only small n)
        if n <= 3000:
            start = time.perf_counter()
            for _ in range(reps):
                two_sum_bruteforce(nums, target)
            row["bruteforce_s"] = (time.perf_counter() - start) / reps
        else:
            row["bruteforce_s"] = float("nan")

        # hash
        start = time.perf_counter()
        for _ in range(reps):
            two_sum_hash(nums, target)
        row["hash_s"] = (time.perf_counter() - start) / reps

        # two pointers
        start = time.perf_counter()
        for _ in range(reps):
            two_sum_two_pointers(nums, target)
        row["two_pointers_s"] = (time.perf_counter() - start) / reps

        results.append(row)
    return results


if __name__ == "__main__":
    # quick correctness
    nums = [2, 7, 11, 15]
    assert two_sum_bruteforce(nums, 9) in {(0, 1), (1, 0)}
    assert two_sum_hash(nums, 9) in {(0, 1), (1, 0)}
    assert two_sum_two_pointers(nums, 9) == (0, 1)

    print("Two Sum quick tests: PASS")

    # small benches
    rows = bench_two_sum(sizes=[1000, 3000], reps=1)
    print("\nTwo Sum timings (seconds):")
    for r in rows:
        n = int(r["n"])  # type: ignore
        print(f"  n={n:5d} | brute={r['bruteforce_s']:.6f} | hash={r['hash_s']:.6f} | 2ptr={r['two_pointers_s']:.6f}")

    # exercise: all pairs
    ex_nums = [1, 3, 2, 2, 4, 0, 3]
    ex_pairs = two_sum_all_pairs(ex_nums, 4)
    expected_pairs = {(0, 1), (0, 6), (2, 3), (4, 5)}
    print("All-pairs exercise result:", ex_pairs)
    if set(ex_pairs) == expected_pairs:
        print("All-pairs exercise: PASS")
    else:
        print("All-pairs exercise: CHECK manually — expected", sorted(expected_pairs))
