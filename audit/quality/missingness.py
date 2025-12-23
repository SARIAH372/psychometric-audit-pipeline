from __future__ import annotations
import pandas as pd


def missingness_table(df_items: pd.DataFrame, item_cols: list[str]) -> pd.DataFrame:
    """
    Returns missingness rate per item.
    """
    miss = df_items[item_cols].isna().mean().sort_values(ascending=False)
    return pd.DataFrame({"item": miss.index, "missing_rate": miss.values})
