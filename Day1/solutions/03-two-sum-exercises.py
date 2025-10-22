"""
Solutions: Two Sum exercises

Includes commented solutions for:
1) All index pairs for target sum
2) Unique value pairs for target sum
3) Three Sum (unique triplets)
"""

from __future__ import annotations
from typing import List, Tuple, Iterable


def two_sum_all_pairs(nums: List[int], target: int) -> List[Tuple[int, int]]:
    """Return all unique index pairs (i<j) where nums[i]+nums[j]==target.

    Approach:
    - Map value -> list of indices.
    - For each distinct value v, complement c=target-v:
      - v<c: cross product of indices[v] and indices[c]
      - v==c: all index combinations within indices[v]
    - Deduplicate and sort pairs lexicographically.
    """
    from collections import defaultdict

    idxs = defaultdict(list)
    for i, v in enumerate(nums):
        idxs[v].append(i)

    pairs: set[Tuple[int, int]] = set()
    seen: set[int] = set()
    for v in list(idxs.keys()):
        if v in seen:
            continue
        c = target - v
        if c not in idxs:
            seen.add(v)
            continue
        if v < c:
            for i in idxs[v]:
                for j in idxs[c]:
                    a, b = (i, j) if i < j else (j, i)
                    pairs.add((a, b))
            seen.add(v); seen.add(c)
        elif v == c:
            lst = idxs[v]
            m = len(lst)
            for a in range(m):
                for b in range(a + 1, m):
                    pairs.add((lst[a], lst[b]))
            seen.add(v)
        else:
            seen.add(v)
    return sorted(pairs)


def two_sum_unique_value_pairs(nums: Iterable[int], target: int) -> List[Tuple[int, int]]:
    """Return unique value pairs (x,y) with x<=y and x+y==target, sorted by value.

    Approach:
    - Use a set of values for membership; iterate distinct values.
    - Handle v==c to ensure the value appears at least twice in the input.
    """
    from collections import Counter

    cnt = Counter(nums)
    vals = sorted(cnt.keys())
    out: List[Tuple[int, int]] = []
    i, j = 0, len(vals) - 1
    while i <= j:
        x, y = vals[i], vals[j]
        s = x + y
        if s == target:
            if x < y:
                out.append((x, y))
            else:  # x == y
                if cnt[x] >= 2:
                    out.append((x, y))
            i += 1
            j -= 1
        elif s < target:
            i += 1
        else:
            j -= 1
    return out


def three_sum(nums: List[int], target: int = 0) -> List[Tuple[int, int, int]]:
    """Return unique triplets (a,b,c) such that a+b+c==target.

    Classic solution:
    - Sort nums and iterate i; for each i, do two-pointer on the rest.
    - Skip duplicates for i and within two-pointer to ensure uniqueness.
    - O(n^2) time, O(1) extra (ignoring output).
    """
    nums = sorted(nums)
    n = len(nums)
    res: List[Tuple[int, int, int]] = []
    for i in range(n):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        lo, hi = i + 1, n - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s == target:
                res.append((nums[i], nums[lo], nums[hi]))
                lo += 1
                hi -= 1
                while lo < hi and nums[lo] == nums[lo - 1]:
                    lo += 1
                while lo < hi and nums[hi] == nums[hi + 1]:
                    hi -= 1
            elif s < target:
                lo += 1
            else:
                hi -= 1
    return res


if __name__ == "__main__":
    # 1) all index pairs
    nums = [1, 3, 2, 2, 4, 0, 3]
    pairs = two_sum_all_pairs(nums, 4)
    assert set(pairs) == {(0, 1), (0, 6), (2, 3), (4, 5)}

    # 2) unique value pairs
    v_pairs = two_sum_unique_value_pairs([1, 1, 2, 2, 3, 3], 4)
    assert v_pairs == [(1, 3), (2, 2)]

    # 3) three sum
    triples = three_sum([-1, 0, 1, 2, -1, -4], 0)
    assert set(triples) == {(-1, -1, 2), (-1, 0, 1)}

    print("Two Sum exercises: PASS")
