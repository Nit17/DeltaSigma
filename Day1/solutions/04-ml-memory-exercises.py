"""
Solutions: ML memory exercises (optional dependencies: pandas, numpy, scipy)

Includes commented solutions for:
1) Downcasting report (pandas)
2) Chunked CSV sum (pandas)
3) Sparse vs dense experiment (numpy/scipy)

These functions are import-guarded and will raise ImportError if the
required package is missing. The __main__ section tries small demos if available.
"""


from __future__ import annotations
from typing import Dict, Optional

def downcasting_report(df) -> Dict[str, float]:
    """Return memory before/after downcasting and savings.

    Steps:
    - Measure memory with df.memory_usage(deep=True).sum()
    - Downcast numeric columns using pandas.to_numeric
    - Measure again and compute savings
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise ImportError("pandas required for downcasting_report") from e

    before = float(df.memory_usage(deep=True).sum())
    df2 = df.copy()
    for col in df2.select_dtypes(include=["int", "int64", "int32"]).columns:
        df2[col] = pd.to_numeric(df2[col], downcast="integer")
    for col in df2.select_dtypes(include=["float"]).columns:
        df2[col] = pd.to_numeric(df2[col], downcast="float")
    after = float(df2.memory_usage(deep=True).sum())
    savings = before - after
    pct = (savings / before * 100.0) if before else 0.0
    return {"before_bytes": before, "after_bytes": after, "savings_bytes": savings, "savings_pct": pct}


def chunked_csv_sum(path: str, column: str, chunksize: int = 100_000) -> float:
    """Sum a numeric column from a CSV using chunked reads to cap memory.

    Avoids collecting all chunks; converts to numeric to handle parse issues.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise ImportError("pandas required for chunked_csv_sum") from e

    total = 0.0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        total += pd.to_numeric(chunk[column], errors="coerce").sum()
    return float(total)


def sparse_vs_dense_experiment(n_rows: int = 5000, n_cols: int = 2000, density: float = 0.001, seed: Optional[int] = 0) -> Dict[str, float]:
    """Compare memory/time between dense numpy arrays and scipy.sparse CSR.

    Creates a random sparse matrix and a dense array of the same shape.
    Measures rough memory and simple operation timings.
    """
    try:
        import numpy as np  # type: ignore
        from scipy import sparse  # type: ignore
        import time
    except Exception as e:
        raise ImportError("numpy and scipy required for sparse_vs_dense_experiment") from e

    rng = np.random.default_rng(seed)

    # Sparse random CSR
    t0 = time.perf_counter()
    S = sparse.random(n_rows, n_cols, density=density, format="csr", random_state=rng, dtype=np.float32)
    t_sparse_build = time.perf_counter() - t0

    # Dense array (float32)
    t0 = time.perf_counter()
    D = (rng.random((n_rows, n_cols), dtype=np.float32) < density) * rng.random((n_rows, n_cols), dtype=np.float32)
    t_dense_build = time.perf_counter() - t0

    # Memory usage
    dense_bytes = float(D.nbytes)
    sparse_bytes = float(S.data.nbytes + S.indptr.nbytes + S.indices.nbytes)

    # Simple ops timing (sum)
    t0 = time.perf_counter(); s_dense = float(D.sum()); t_dense_sum = time.perf_counter() - t0
    t0 = time.perf_counter(); s_sparse = float(S.sum()); t_sparse_sum = time.perf_counter() - t0

    return {
        "dense_bytes": dense_bytes,
        "sparse_bytes": sparse_bytes,
        "t_dense_build_s": t_dense_build,
        "t_sparse_build_s": t_sparse_build,
        "t_dense_sum_s": t_dense_sum,
        "t_sparse_sum_s": t_sparse_sum,
        "sum_dense": s_dense,
        "sum_sparse": s_sparse,
    }


if __name__ == "__main__":
    # Lightweight self-checks if deps exist
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
        rep = downcasting_report(df)
        assert rep["after_bytes"] <= rep["before_bytes"]
        print("Downcasting report:", rep)
    except Exception as e:
        print("(skip) pandas not available:", e)

    # chunked_csv_sum is not demonstrated here to avoid file IO; call it with your CSV path.

    try:
        res = sparse_vs_dense_experiment(1000, 1000, 0.001)
        print("Sparse vs dense:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})
    except Exception as e:
        print("(skip) numpy/scipy not available:", e)
