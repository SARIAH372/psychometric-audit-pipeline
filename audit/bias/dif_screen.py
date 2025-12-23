from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _empty_result() -> pd.DataFrame:
    """Standard empty DIF output table (never None)."""
    return pd.DataFrame(columns=[
        "item", "n", "n_groups", "min_group_n",
        "p_group_joint", "q_group_joint", "largest_abs_group_beta", "note"
    ])


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg FDR correction.
    Returns q-values, same order as pvals.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(p)
    ranked = p[order]
    q_ranked = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q_ranked[i] = prev

    q = np.empty(n, dtype=float)
    q[order] = q_ranked
    return q


def dif_screen_linear(
    df_sub: pd.DataFrame,
    item_cols: list[str],
    group_col: str,
    min_n_per_group: int = 30,
    standardize_total: bool = True
) -> pd.DataFrame:
    """
    DIF screening via OLS regression (fast, explainable):

      item ~ total_score + group_dummies

    Uses a joint Wald test for all group terms per item, then BH-FDR across items.
    Returns a DataFrame (never None).
    """

    # --- Basic checks ---
    if group_col not in df_sub.columns:
        out = _empty_result()
        out.loc[0] = [None, None, None, None, None, None, None, f"Group column '{group_col}' not found."]
        return out

    if not item_cols:
        out = _empty_result()
        out.loc[0] = [None, None, None, None, None, None, None, "item_cols is empty."]
        return out

    # Coerce numeric items
    X_items = df_sub[item_cols].apply(pd.to_numeric, errors="coerce")
    g = df_sub[group_col].astype("category")

    df = pd.concat([X_items, g.rename(group_col)], axis=1).dropna(axis=0, how="any")

    if df.shape[0] < 50:
        out = _empty_result()
        out.loc[0] = [None, int(df.shape[0]), int(df[group_col].nunique()), None, None, None, None,
                      "Too few complete rows (<50) for DIF screening."]
        return out

    # Filter groups with small n
    counts = df[group_col].value_counts()
    valid_groups = counts[counts >= min_n_per_group].index
    df = df[df[group_col].isin(valid_groups)]

    if df[group_col].nunique() < 2:
        out = _empty_result()
        min_gn = int(df[group_col].value_counts().min()) if df.shape[0] > 0 else 0
        out.loc[0] = [None, int(df.shape[0]), int(df[group_col].nunique()), min_gn, None, None, None,
                      f"Not enough groups after filtering (min_n_per_group={min_n_per_group})."]
        return out

    items = df[item_cols]
    g = df[group_col].astype("category")

    # Total score
    total = items.sum(axis=1)
    if standardize_total:
        sd = float(total.std(ddof=1))
        total = (total - float(total.mean())) / sd if sd != 0 else (total * 0.0)

    # Group dummies
    G = pd.get_dummies(g, drop_first=True)
    X = pd.concat([total.rename("total_score"), G], axis=1)
    X = sm.add_constant(X, has_constant="add")

    # Prepare joint test matrix size
    start = 2  # const + total_score
    m = G.shape[1]  # number of group dummy cols

    # If somehow still no dummy columns (shouldn't happen if >=2 groups)
    if m == 0:
        out = _empty_result()
        out.loc[0] = [None, int(df.shape[0]), int(df[group_col].nunique()), int(g.value_counts().min()),
                      None, None, None, "No group dummy variables created."]
        return out

    pvals: list[float] = []
    rows: list[dict] = []

    for item in item_cols:
        y = pd.to_numeric(items[item], errors="coerce")

        try:
            model = sm.OLS(y, X).fit()

            # Joint Wald test: all group terms = 0
            R = np.zeros((m, X.shape[1]))
            for i in range(m):
                R[i, start + i] = 1.0

            wald = model.wald_test(R)
            p_joint = float(wald.pvalue)

            betas = model.params[start:start + m]
            largest_abs_beta = float(np.max(np.abs(betas))) if len(betas) else np.nan

            rows.append({
                "item": item,
                "n": int(len(y)),
                "n_groups": int(g.cat.categories.size),
                "min_group_n": int(g.value_counts().min()),
                "p_group_joint": p_joint,
                "largest_abs_group_beta": largest_abs_beta,
                "note": ""
            })
            pvals.append(p_joint)

        except Exception as e:
            # Never crash the run; return a row noting failure for that item
            rows.append({
                "item": item,
                "n": int(len(y)),
                "n_groups": int(g.cat.categories.size),
                "min_group_n": int(g.value_counts().min()),
                "p_group_joint": np.nan,
                "largest_abs_group_beta": np.nan,
                "note": f"Model failed: {type(e).__name__}"
            })
            pvals.append(1.0)  # conservative

    qvals = _benjamini_hochberg(np.array(pvals, dtype=float))

    out = pd.DataFrame(rows)
    out["q_group_joint"] = qvals
    out = out.sort_values(["q_group_joint", "p_group_joint"], ascending=True)

    # Guarantee DataFrame return
    return out



