"""
02-collections: Counter, defaultdict, deque, namedtuple, OrderedDict, and heapq basics.

See Day1/THEORY.md for deeper notes on when to use each, algorithmic complexities,
and how heaps compare to full sorts for top-k selection.
"""

from __future__ import annotations
from typing import Any, Dict
from collections import Counter, defaultdict, deque, namedtuple, OrderedDict
import heapq


def collections_examples() -> Dict[str, Any]:
    data = ["a", "b", "a", "c", "b", "a"]
    counter = Counter(data)
    top2 = counter.most_common(2)

    groups = defaultdict(list)
    for i, ch in enumerate(data):
        groups[ch].append(i)

    dq = deque([1, 2, 3])
    dq.appendleft(0)
    left = dq.popleft()

    Point = namedtuple("Point", ["x", "y"])
    p = Point(3, 4)

    od = OrderedDict()
    od["x"] = 1
    od["y"] = 2
    od.move_to_end("x")

    arr = [5, 3, 8, 1]
    heapq.heapify(arr)
    smallest = heapq.heappop(arr)
    heapq.heappush(arr, 2)

    maxheap = []
    for v in [5, 3, 8, 1]:
        heapq.heappush(maxheap, -v)
    max_val = -heapq.heappop(maxheap)

    k_largest = heapq.nlargest(2, [5, 3, 8, 1])
    k_smallest = heapq.nsmallest(2, [5, 3, 8, 1])

    assert left == 0 and smallest == 1 and max_val == 8
    return {
        "counter": counter,
        "top2": top2,
        "groups": dict(groups),
        "deque_after_ops": list(dq),
        "point": p,
        "ordered_keys": list(od.keys()),
        "heap_after_ops": list(arr),
        "k_largest": k_largest,
        "k_smallest": k_smallest,
    }


def sliding_window_max(nums: list[int], k: int) -> list[int]:
    """Return max for each sliding window of size k using a deque.

    Maintains a decreasing deque of indices. O(n) time, O(k) space.
    """
    if k <= 0:
        return []
    n = len(nums)
    if n == 0 or k > n:
        return []

    q = deque()  # indices, nums[q] is decreasing
    out: list[int] = []

    for i, x in enumerate(nums):
        # remove indices out of window
        while q and q[0] <= i - k:
            q.popleft()
        # maintain decreasing order
        while q and nums[q[-1]] <= x:
            q.pop()
        q.append(i)
        if i >= k - 1:
            out.append(nums[q[0]])
    return out


if __name__ == "__main__":
    res = collections_examples()
    print("Top2:", res["top2"])
    print("Groups:", res["groups"])
    # exercise check: sliding window max
    _nums = [1, 3, -1, -3, 5, 3, 6, 7]
    _k = 3
    assert sliding_window_max(_nums, _k) == [3, 3, 5, 5, 6, 7]
    print("Sliding window max: PASS")