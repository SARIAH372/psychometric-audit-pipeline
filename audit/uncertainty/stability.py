from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict

from audit.bias.dif_screen import dif_screen_linear


def dif_stability(
    df_sub: pd.DataFrame,
    item_cols: list[str],
    group_col: str,
    min_n_per_group: int,
    q_threshold: float = 0.05,
    n_boot: int = 50,
    seed: int = 123
) -> pd.DataFrame:
    """
    Bootstrap stability analysis for DIF.

    Returns a DataFrame with:
      item, flag_rate, n_boot_used
    """
    rng = np.random.default_rng(seed)
    n = df_sub.shape[0]

    flag_counts: Dict[str, int] = {it: 0 for it in item_cols}
    used = 0

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = df_sub.iloc[idx]

        try:
            dif_df = dif_screen_linear(
                df_sub=sample,
                item_cols=item_cols,
                group_col=group_col,
                min_n_per_group=min_n_per_group,
                standardize_total=True
            )
        except Exception:
            continue

        if dif_df is None or dif_df.empty or "q_group_joint" not in dif_df.columns:
            continue

        if "item" in dif_df.columns and dif_df["item"].isna().all():
            continue

        used += 1
        flagged = dif_df[dif_df["q_group_joint"] <= q_threshold]["item"].dropna().astype(str).tolist()

        for it in flagged:
            if it in flag_counts:
                flag_counts[it] += 1

    rows = []
    for it in item_cols:
        rate = flag_counts[it] / used if used > 0 else np.nan
        rows.append({
            "item": it,
            "flag_rate": rate,
            "n_boot_used": used
        })

    return pd.DataFrame(rows).sort_values("flag_rate", ascending=False)
