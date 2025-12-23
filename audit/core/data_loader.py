from __future__ import annotations
import os
import pandas as pd


def load_table(path: str, fmt: str = "auto") -> pd.DataFrame:
    """
    Load CSV or Parquet from disk.
    fmt: auto | csv | parquet
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    if fmt == "auto":
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            fmt = "csv"
        elif ext in [".parquet", ".pq"]:
            fmt = "parquet"
        else:
            raise ValueError(f"Unsupported file extension: {ext} (use .csv or .parquet)")

    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unknown format: {fmt}")


def select_columns(df: pd.DataFrame, item_cols: list[str], group_col: str | None) -> pd.DataFrame:
    """
    Return a dataframe containing item columns (+ group column if provided).
    """
    if not item_cols:
        raise ValueError("config.data.item_cols is empty. Add your item column names to configs/config.yaml")

    missing = [c for c in item_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing item columns in dataset: {missing}")

    cols = list(item_cols)
    if group_col:
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in dataset.")
        cols.append(group_col)

    return df[cols].copy()
