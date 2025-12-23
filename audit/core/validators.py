from __future__ import annotations
import pandas as pd


def coerce_items_numeric(df_items: pd.DataFrame, item_cols: list[str]) -> pd.DataFrame:
    out = df_items.copy()
    for c in item_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def validate_allowed_values(items: pd.DataFrame, item_cols: list[str], allowed_values: list[float]) -> pd.DataFrame:
    allowed = set(float(v) for v in allowed_values)
    rows = []
    for c in item_cols:
        vals = items[c].dropna().astype(float)
        bad = vals[~vals.isin(allowed)]
        if len(bad) > 0:
            examples = sorted(set(bad.head(10).tolist()))
            rows.append({
                "item": c,
                "n_violations": int(len(bad)),
                "examples": ", ".join(str(x) for x in examples)
            })
    return pd.DataFrame(rows)


def validate_group_counts(df: pd.DataFrame, group_col: str, min_n_per_group: int) -> pd.DataFrame:
    counts = df[group_col].value_counts(dropna=False).rename_axis("group").reset_index(name="n")
    counts["ok"] = counts["n"] >= min_n_per_group
    return counts
