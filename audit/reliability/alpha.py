from __future__ import annotations
import numpy as np
import pandas as pd


def cronbach_alpha(df_items: pd.DataFrame, item_cols: list[str]) -> float:
    """
    Cronbach's alpha with listwise deletion across the item columns.
    """
    X = df_items[item_cols].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=0, how="any")

    if X.shape[0] < 3 or X.shape[1] < 2:
        return float("nan")

    arr = X.to_numpy(dtype=float)
    k = arr.shape[1]
    item_vars = arr.var(axis=0, ddof=1)
    total_var = arr.sum(axis=1).var(ddof=1)

    if not np.isfinite(total_var) or total_var <= 0:
        return float("nan")

    return float((k / (k - 1)) * (1.0 - (item_vars.sum() / total_var)))


def alpha_by_group(
    df_sub: pd.DataFrame,
    item_cols: list[str],
    group_col: str,
    min_n_per_group: int = 30
) -> pd.DataFrame:
    """
    Cronbach alpha per group (listwise within each group).
    Filters groups with n_used < min_n_per_group.
    """
    out = []
    if group_col not in df_sub.columns:
        return pd.DataFrame(columns=["group", "n_rows", "n_used", "alpha", "included"])

    for g, part in df_sub.groupby(group_col, dropna=False):
        items = part[item_cols].apply(pd.to_numeric, errors="coerce")
        n_rows = int(part.shape[0])
        n_used = int(items.dropna(axis=0, how="any").shape[0])
        a = cronbach_alpha(part, item_cols=item_cols)

        out.append({
            "group": str(g),
            "n_rows": n_rows,
            "n_used": n_used,
            "alpha": a,
            "included": bool(n_used >= min_n_per_group),
        })

    df_out = pd.DataFrame(out).sort_values(["included", "n_used"], ascending=[False, False])
    return df_out
