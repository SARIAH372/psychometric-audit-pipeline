from __future__ import annotations
import numpy as np
import pandas as pd


def corrected_item_total(df_items: pd.DataFrame, item_cols: list[str]) -> pd.DataFrame:
    """
    Corrected item-total correlation:
      corr(item, total_score_without_item)
    Uses listwise deletion across items.
    """
    X = df_items[item_cols].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=0, how="any")

    if X.shape[0] < 3:
        return pd.DataFrame({"item": item_cols, "corrected_item_total_r": [np.nan] * len(item_cols)})

    total = X.sum(axis=1)
    rows = []
    for c in item_cols:
        corrected_total = total - X[c]
        if X[c].std(ddof=1) == 0 or corrected_total.std(ddof=1) == 0:
            r = np.nan
        else:
            r = float(X[c].corr(corrected_total))
        rows.append({"item": c, "corrected_item_total_r": r})

    return pd.DataFrame(rows).sort_values("corrected_item_total_r", ascending=True)
