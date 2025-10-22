"""
01-containers: Python containers and Big-O quick reference + tiny benchmarks.

Includes:
- containers_quick_reference: brief descriptions
- container_examples: small sanity examples
- BIG_O_NOTES: cheat sheet (see Day1/THEORY.md for deeper theory: asymptotics, internals, and trade-offs)
- bench_container_ops: micro-benchmarks for membership and insert patterns
"""

from __future__ import annotations
from typing import Dict
import random
import time


def containers_quick_reference() -> Dict[str, str]:
    return {
        "list": "Dynamic array; fast append/pop end; slow middle ops; ordered.",
        "dict": "Hash map; O(1) avg get/set; insertion-ordered (3.7+).",
        "set": "Hash set; unique items; O(1) avg membership.",
        "tuple": "Immutable sequence; hashable if elements are hashable.",
    }


# Big-O cheat notes
BIG_O_NOTES = {
    "Access": {"list[i]": "O(1)", "dict[key]": "O(1) avg", "set membership": "O(1) avg"},
    "Insert/Delete": {
        "list append/pop end": "O(1) amortized",
        "list insert/pop middle": "O(n)",
        "dict/set put/delete": "O(1) avg",
    },
    "Sorting": {"Timsort (list.sort)": "O(n log n)"},
    "Traversal": {"list/dict/set iteration": "O(n)"},
}


def container_examples() -> None:
    # list
    a = [3, 1, 4]
    a.append(1)
    a.extend([5, 9])
    a_sorted = sorted(a)
    a.sort()
    # dict
    d = {"a": 1}
    d["b"] = 2
    d.setdefault("c", 0)
    # set
    s = {1, 2, 2, 3}
    s.add(4)
    # tuple
    t = (1, 2, 3)
    assert ("a" in d) and (2 in s) and (t[0] == 1)
    assert a_sorted == [1, 1, 3, 4, 5, 9]


def bench_container_ops(n: int = 20000, trials: int = 3) -> Dict[str, float]:
    """Micro-benchmarks to visualize typical complexity trade-offs.

    - list membership vs set membership
    - list append vs insert(0)

    Keep n small for quick runs. Returns average seconds per trial.
    """
    rng = random.Random(0)
    base = list(range(n))
    probe = [rng.randrange(n) for _ in range(n // 5)]

    timings: Dict[str, float] = {}

    # list membership
    start = time.perf_counter()
    for _ in range(trials):
        count = 0
        for x in probe:
            if x in base:
                count += 1
    timings["list_membership"] = (time.perf_counter() - start) / trials

    # set membership
    s = set(base)
    start = time.perf_counter()
    for _ in range(trials):
        count = 0
        for x in probe:
            if x in s:
                count += 1
    timings["set_membership"] = (time.perf_counter() - start) / trials

    # append at end
    start = time.perf_counter()
    for _ in range(trials):
        arr = []
        for x in base:
            arr.append(x)
    timings["list_append_end"] = (time.perf_counter() - start) / trials

    # insert at front (slow)
    start = time.perf_counter()
    for _ in range(trials):
        arr = []
        for x in base:
            arr.insert(0, x)
    timings["list_insert_front"] = (time.perf_counter() - start) / trials

    return timings


if __name__ == "__main__":
    container_examples()
    times = bench_container_ops(n=10000, trials=2)
    print("Containers benchmarks (avg seconds):")
    for k in sorted(times):
        print(f"  {k:20s} : {times[k]:.6f}")
