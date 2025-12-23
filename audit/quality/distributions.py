from __future__ import annotations
import pandas as pd


def response_distribution_long(df_items: pd.DataFrame, item_cols: list[str]) -> pd.DataFrame:
    """
    Long-format table: item, value, proportion
    Useful for spotting weird coding or extreme skew.
    """
    rows = []
    X = df_items[item_cols]

    for c in item_cols:
        s = pd.to_numeric(X[c], errors="coerce")
        vc = s.value_counts(dropna=False, normalize=True).sort_index()
        for v, p in vc.items():
            rows.append({"item": c, "value": v, "proportion": float(p)})

    return pd.DataFrame(rows)
