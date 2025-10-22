"""
04-ml-memory: Efficient preprocessing & memory control patterns for ML pipelines.

All heavy dependencies are optional; functions raise ImportError if missing.
See Day1/THEORY.md §7 for deeper notes on numpy views vs copies, pandas dtypes,
sparse formats (CSR/CSC/COO), memmap, and general memory strategies.
"""

from __future__ import annotations
from typing import Any, Iterable, List, Tuple


def pandas_downcast_df(df):
    """Downcast numeric columns to reduce memory footprint (pandas required)."""
    try:
        import pandas as pd  # type: ignore
    except Exception:
        raise ImportError("pandas is required for pandas_downcast_df")

    df = df.copy()
    for col in df.select_dtypes(include=["int", "int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def read_csv_in_chunks(path: str, chunksize: int = 100_000):
    """Yield downcasted chunks from a CSV (pandas required)."""
    try:
        import pandas as pd  # type: ignore
    except Exception:
        raise ImportError("pandas is required for read_csv_in_chunks")

    for chunk in pd.read_csv(path, chunksize=chunksize):
        yield pandas_downcast_df(chunk)


def numpy_memmap_example(path: str, shape: Tuple[int, ...], dtype: str = "float32") -> None:
    """Create and read large arrays with numpy.memmap to avoid full RAM usage."""
    try:
        import numpy as np  # type: ignore
    except Exception:
        raise ImportError("numpy is required for numpy_memmap_example")

    mm = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
    mm[...] = 0.0
    mm.flush()

    mm2 = np.memmap(path, dtype=dtype, mode="r", shape=shape)
    _ = float(mm2[0])


def generator_pipeline(source: Iterable[Any], transform) -> Iterable[Any]:
    for item in source:
        yield transform(item)


def sparse_matrix_example(values: List[Tuple[int, int, float]]):
    """Return CSR sparse matrix from (row, col, val) triplets (scipy required)."""
    try:
        import numpy as np  # type: ignore
        from scipy.sparse import coo_matrix  # type: ignore
    except Exception:
        raise ImportError("scipy (and numpy) required for sparse_matrix_example")

    if not values:
        raise ValueError("values must be non-empty")

    rows, cols, vals = zip(*values)
    n_rows = max(rows) + 1
    n_cols = max(cols) + 1
    coo = coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))
    return coo.tocsr()


def ml_memory_tips() -> List[str]:
    return [
        "Prefer float32/int32 over float64/int64 when precision permits.",
        "Chunk large reads; avoid loading entire datasets into RAM.",
        "Use generators/iterators for streaming pipelines.",
        "Leverage numpy.memmap for large arrays.",
        "Use sparse matrices for high-dimensional sparse data.",
        "Avoid unnecessary copies; watch chained ops.",
        "Downcast to pandas.Categorical for low-cardinality strings.",
        "Cache intermediates to disk when recomputation is expensive.",
    ]


if __name__ == "__main__":
    print("ML memory tips:")
    for tip in ml_memory_tips():
        print(" -", tip)
