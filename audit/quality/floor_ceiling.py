from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Sequence


def floor_ceiling_table(
    df_items: pd.DataFrame,
    item_cols: list[str],
    allowed_values: Optional[Sequence[float]] = None
) -> pd.DataFrame:
    """
    Floor rate = P(item == min_value)
    Ceiling rate = P(item == max_value)

    If allowed_values is provided, uses min/max of that scale.
    Otherwise uses observed min/max across the dataset.
    """
    X = df_items[item_cols].copy()

    # choose scale min/max
    if allowed_values and len(allowed_values) >= 2:
        lo = float(min(allowed_values))
        hi = float(max(allowed_values))
    else:
        arr = X.to_numpy(dtype=float)
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))

    rows = []
    for c in item_cols:
        s = pd.to_numeric(X[c], errors="coerce").dropna()
        if s.empty:
            rows.append({"item": c, "floor_rate": np.nan, "ceiling_rate": np.nan})
            continue
        rows.append({
            "item": c,
            "floor_rate": float((s == lo).mean()),
            "ceiling_rate": float((s == hi).mean())
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["ceiling_rate", "floor_rate"], ascending=False)
