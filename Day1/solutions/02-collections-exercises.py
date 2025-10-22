"""
Solutions: collections and heapq exercises

Includes commented solutions for:
1) Sliding window maximum (deque, O(n))
2) Moving average over a stream (deque + running sum)
3) Top-k frequent elements (Counter + heapq.nlargest)
"""

from __future__ import annotations
from typing import Iterable, List, Tuple
from collections import deque, Counter
import heapq


def sliding_window_max(nums: List[int], k: int) -> List[int]:
    """Return max for each sliding window of size k using a deque (O(n)).

    Maintains a decreasing deque of indices:
    - Pop from back while current >= tail value
    - Pop from front if out of window (index <= i - k)
    - Front holds index of maximum
    """
    if k <= 0:
        return []
    n = len(nums)
    if n == 0 or k > n:
        return []

    q = deque()
    out: List[int] = []
    for i, x in enumerate(nums):
        while q and q[0] <= i - k:
            q.popleft()
        while q and nums[q[-1]] <= x:
            q.pop()
        q.append(i)
        if i >= k - 1:
            out.append(nums[q[0]])
    return out


def moving_average_stream(stream: Iterable[float], k: int) -> List[float]:
    """Compute moving average with a fixed-size window using deque.

    Keep a running sum; when the window exceeds k, pop left and subtract.
    """
    if k <= 0:
        return []
    q = deque()
    s = 0.0
    out: List[float] = []
    for x in stream:
        q.append(x)
        s += float(x)
        if len(q) > k:
            s -= float(q.popleft())
        if len(q) == k:
            out.append(s / k)
    return out


def top_k_frequent(nums: Iterable[int], k: int) -> List[Tuple[int, int]]:
    """Return k most frequent (value, count) pairs.

    Uses Counter + heapq.nlargest with key on counts. O(n log k).
    """
    if k <= 0:
        return []
    c = Counter(nums)
    return heapq.nlargest(k, c.items(), key=lambda kv: kv[1])


if __name__ == "__main__":
    # 1) sliding window max
    assert sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]

    # 2) moving average
    avgs = moving_average_stream([1, 2, 3, 4, 5], 3)
    assert avgs == [2.0, 3.0, 4.0]

    # 3) top-k frequent
    tk = top_k_frequent([1, 1, 2, 3, 3, 3, 2, 2, 2], 2)
    # counts: 2->4, 3->3, 1->2
    assert tk[0][0] == 2 and tk[0][1] == 4 and tk[1][0] == 3

    print("Collections exercises: PASS")
