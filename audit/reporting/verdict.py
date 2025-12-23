from __future__ import annotations
import pandas as pd


def compute_verdict(
    alpha_overall: float | None,
    dif_df: pd.DataFrame | None,
    alpha_threshold: float = 0.70,
    q_threshold: float = 0.05,
    max_flagged_for_caution: int = 2
) -> dict:
    """
    Transparent rule-based verdict.
    Returns dict with: verdict, reason, flagged_items_count
    """

    if alpha_overall is None or not pd.notnull(alpha_overall):
        return {"verdict": "NOT_COMPARABLE", "reason": "Alpha missing/NA.", "flagged_items_count": None}

    if alpha_overall < alpha_threshold:
        return {
            "verdict": "NOT_COMPARABLE",
            "reason": f"Low reliability: alpha={alpha_overall:.3f} < {alpha_threshold:.2f}.",
            "flagged_items_count": None
        }

    if dif_df is None or dif_df.empty or "q_group_joint" not in dif_df.columns:
        return {"verdict": "COMPARABLE", "reason": "DIF results unavailable; reliability acceptable.", "flagged_items_count": 0}

    flagged = dif_df[dif_df["q_group_joint"] <= q_threshold]
    n_flagged = int(flagged.shape[0])

    if n_flagged == 0:
        return {"verdict": "COMPARABLE", "reason": f"No DIF flags at q≤{q_threshold}.", "flagged_items_count": 0}

    if n_flagged <= max_flagged_for_caution:
        return {
            "verdict": "CAUTION",
            "reason": f"{n_flagged} item(s) flagged for DIF (q≤{q_threshold}). Review flagged items.",
            "flagged_items_count": n_flagged
        }

    return {
        "verdict": "NOT_COMPARABLE",
        "reason": f"{n_flagged} items flagged for DIF (q≤{q_threshold}). Group comparisons may be biased.",
        "flagged_items_count": n_flagged
    }
