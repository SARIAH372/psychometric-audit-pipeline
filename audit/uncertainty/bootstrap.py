from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Tuple


def bootstrap_statistic(
    df: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    n_boot: int = 200,
    seed: int = 42
) -> Tuple[float, float, int]:
    """
    Generic bootstrap for a scalar statistic.

    Parameters
    ----------
    df : DataFrame
        Data to resample (rows)
    stat_fn : callable
        Function computing a scalar statistic from df
    n_boot : int
        Number of bootstrap resamples
    seed : int
        Random seed

    Returns
    -------
    ci_lo, ci_hi, n_eff
    """
    rng = np.random.default_rng(seed)
    n = df.shape[0]
    vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            v = stat_fn(df.iloc[idx])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue

    vals = np.asarray(vals, dtype=float)
    if vals.size < max(30, n_boot // 5):
        return (np.nan, np.nan, int(vals.size))

    lo = float(np.quantile(vals, 0.025))
    hi = float(np.quantile(vals, 0.975))
    return (lo, hi, int(vals.size))
