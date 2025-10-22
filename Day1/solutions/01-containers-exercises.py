"""
Solutions: Containers exercises

Includes commented solutions for:
1) Stable dedupe (order-preserving)
2) Rotate list right by k (in-place and slice variants)
3) Maintain sorted list under inserts (bisect)
4) Flatten one level of nesting
"""

from __future__ import annotations
from typing import Iterable, List
import bisect


def stable_dedupe(seq: Iterable[int]) -> List[int]:
    """Order-preserving dedupe using a seen set.

    Idea:
    - Walk once, append if not seen, mark as seen.
    - O(n) time and O(n) extra space.
    """
    seen = set()
    out: List[int] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def rotate_right_slice(arr: List[int], k: int) -> List[int]:
    """Return a new list rotated right by k using slicing.

    Idea:
    - Normalize k = k % n, then take tail+head: arr[-k:] + arr[:-k].
    - O(n) time, O(n) extra space.
    """
    n = len(arr)
    if n == 0:
        return []
    k %= n
    if k == 0:
        return arr.copy()
    return arr[-k:] + arr[:-k]


def rotate_right_inplace(arr: List[int], k: int) -> None:
    """Rotate in-place using reverse-then-reverse trick.

    Steps:
    - Reverse whole list
    - Reverse first k elements
    - Reverse remaining n-k elements
    This achieves rotation in O(n) time, O(1) extra space.
    """
    n = len(arr)
    if n == 0:
        return
    k %= n
    if k == 0:
        return

    def rev(a: List[int], i: int, j: int) -> None:
        while i < j:
            a[i], a[j] = a[j], a[i]
            i += 1
            j -= 1

    rev(arr, 0, n - 1)
    rev(arr, 0, k - 1)
    rev(arr, k, n - 1)


def insert_sorted_stream(values: Iterable[int]) -> List[int]:
    """Keep a running sorted list as values arrive using bisect.insort.

    Each insert is O(n) due to shifting, but simple and effective for modest sizes.
    """
    sorted_list: List[int] = []
    for v in values:
        bisect.insort(sorted_list, v)
    return sorted_list


def flatten_one_level(lists: Iterable[Iterable[int]]) -> List[int]:
    """Flatten one level of nesting.

    Alternatives: list comprehension or itertools.chain.from_iterable.
    """
    out: List[int] = []
    for sub in lists:
        out.extend(sub)
    return out


if __name__ == "__main__":
    # 1) stable dedupe
    assert stable_dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]

    # 2) rotate right
    lst = [1, 2, 3, 4, 5]
    assert rotate_right_slice(lst, 2) == [4, 5, 1, 2, 3]
    rotate_right_inplace(lst, 2)
    assert lst == [4, 5, 1, 2, 3]

    # 3) maintain sorted
    assert insert_sorted_stream([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]

    # 4) flatten
    assert flatten_one_level([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]

    print("Containers exercises: PASS")
