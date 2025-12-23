from __future__ import annotations
import pandas as pd


def assert_columns_exist(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def assert_min_rows(df: pd.DataFrame, min_rows: int, context: str = "") -> None:
    """Ensure minimum rows exist before analysis."""
    if df.shape[0] < min_rows:
        raise ValueError(
            f"Insufficient rows ({df.shape[0]}) for {context}. "
            f"Minimum required: {min_rows}."
        )
