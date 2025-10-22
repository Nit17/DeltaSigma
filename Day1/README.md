# Day 1: Python Basics & Complexity

This directory splits content into focused, runnable modules:

- `01-containers.py` — Lists, dicts, sets, tuples; Big-O cheat sheet; tiny benchmarks (membership/insert).
- `02-collections.py` — Counter, defaultdict, deque, namedtuple, OrderedDict, and heapq basics.
- `03-two-sum.py` — Three solutions (O(n^2), O(n), O(n log n)) + a small benchmark helper.
- `04-ml-memory.py` — Practical patterns for memory-efficient ML preprocessing (optional deps guarded).

## How to run

From the project root (or the `Day1` folder):

```bash
python Day1/01-containers.py
python Day1/02-collections.py
python Day1/03-two-sum.py
python Day1/04-ml-memory.py
```

If you’re using the workspace virtual environment, use its interpreter explicitly:

```bash
/Users/nithinbm/Practice-study/Practice-study/.venv/bin/python Day1/01-containers.py
```

## Notes

- The original `PythonBasics&Complexity.py` now only points to these split modules.
- Heavy dependencies in `04-ml-memory.py` (pandas/numpy/scipy) are optional and imported only when used.
- Benchmarks are tiny, meant to give intuition rather than precise measurements. Adjust sizes if you want deeper comparisons.
