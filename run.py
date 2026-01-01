from __future__ import annotations

import argparse
import os
import yaml
import pandas as pd
import numpy as np

# ---------- CORE ----------
from audit.core.data_loader import load_table
from audit.core.validators import coerce_items_numeric

# ---------- QUALITY ----------
from audit.quality.missingness import missingness_table
from audit.quality.floor_ceiling import floor_ceiling_table
from audit.quality.distributions import response_distribution_long

# ---------- RELIABILITY ----------
from audit.reliability.alpha import cronbach_alpha, alpha_by_group
from audit.reliability.item_total import corrected_item_total

# ---------- DIF ----------
from audit.bias.dif_screen import dif_screen_linear

# ---------- UNCERTAINTY ----------
from audit.uncertainty.bootstrap import bootstrap_statistic
from audit.uncertainty.stability import dif_stability


# ---------- IRT ----------
from audit.irt.grm import recode_ordinal, fit_grm_pymc
from audit.irt.information import information_curve

# ---------- REPORTING ----------
from audit.reporting.verdict import compute_verdict
from audit.reporting.report_html import write_report_html


def parse_args():
    p = argparse.ArgumentParser(description="Mental Health Scale Audit Platform (v4/v5)")
    p.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    return p.parse_args()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_read_csv(path: str) -> pd.DataFrame | None:
    return pd.read_csv(path) if os.path.exists(path) else None


def _maybe_create_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Create age_group if requested but missing and age exists."""
    if "age_group" in df.columns:
        return df
    if "age" not in df.columns:
        return df

    bins = [0, 17, 24, 34, 44, 54, 64, 120]
    labels = ["<=17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True, include_lowest=True)
    return df


def main():
    # =========================
    # 1) Load config
    # =========================
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    run_cfg = cfg.get("run", {})
    rules = cfg.get("rules", {})
    modules = cfg.get("modules", {})
    bootstrap_cfg = cfg.get("bootstrap", {})
    irt_cfg = cfg.get("irt", {})

    path = data_cfg["path"]
    fmt = data_cfg.get("format", "auto")
    item_cols = list(data_cfg["item_cols"])

    # Multi-group support
    group_cols = data_cfg.get("group_cols", [])
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    group_cols = [g for g in group_cols if g]

    allowed_values = data_cfg.get("allowed_values", None)

    outdir = run_cfg.get("outdir", "outputs")
    tag = run_cfg.get("tag", "run")

    min_n_per_group = int(rules.get("min_n_per_group", 200))

    # =========================
    # 2) Prepare folders
    # =========================
    _ensure_dir(outdir)
    tables_dir = os.path.join(outdir, "tables")
    reports_dir = os.path.join(outdir, "reports")
    plots_dir = os.path.join(outdir, "plots")
    _ensure_dir(tables_dir)
    _ensure_dir(reports_dir)
    _ensure_dir(plots_dir)

    # =========================
    # 3) Load data + preprocessing
    # =========================
    df = load_table(path, fmt=fmt)

    # If we want age_group but it's missing, create it from age
    if "age_group" in group_cols and "age_group" not in df.columns:
        df = _maybe_create_age_group(df)

    required = list(item_cols)
    for g in group_cols:
        if g not in required:
            required.append(g)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    df_sub = df[required].copy()

    # Ensure items numeric
    df_sub = coerce_items_numeric(df_sub, item_cols=item_cols)

    # Critical:  dataset uses -1 as missing for PCL items
    df_sub[item_cols] = df_sub[item_cols].replace(-1, pd.NA)

    # =========================
    # A) QUALITY
    # =========================
    if modules.get("quality", True):
        missingness_table(df_sub, item_cols).to_csv(
            os.path.join(tables_dir, f"{tag}_missingness.csv"), index=False
        )
        floor_ceiling_table(df_sub, item_cols, allowed_values).to_csv(
            os.path.join(tables_dir, f"{tag}_floor_ceiling.csv"), index=False
        )
        response_distribution_long(df_sub, item_cols).to_csv(
            os.path.join(tables_dir, f"{tag}_response_distribution.csv"), index=False
        )

    # =========================
    # B) RELIABILITY + Bootstrap alpha CI
    # =========================
    alpha_val = None
    if modules.get("reliability", True):
        alpha_val = cronbach_alpha(df_sub, item_cols)

        alpha_ci_lo = alpha_ci_hi = alpha_boot_n = None
        if modules.get("uncertainty", False):
            acfg = bootstrap_cfg.get("alpha", {})
            if acfg.get("enabled", True):
                n_boot = int(acfg.get("n_boot", 200))
                seed = int(acfg.get("seed", 42))

                def alpha_fn(sample_df: pd.DataFrame) -> float:
                    return cronbach_alpha(sample_df, item_cols)

                ci_lo, ci_hi, n_eff = bootstrap_statistic(df_sub, alpha_fn, n_boot=n_boot, seed=seed)
                alpha_ci_lo, alpha_ci_hi, alpha_boot_n = ci_lo, ci_hi, n_eff

        pd.DataFrame([{
            "alpha": alpha_val,
            "alpha_ci_lo": alpha_ci_lo,
            "alpha_ci_hi": alpha_ci_hi,
            "alpha_boot_n_eff": alpha_boot_n
        }]).to_csv(os.path.join(tables_dir, f"{tag}_alpha_overall.csv"), index=False)

        corrected_item_total(df_sub, item_cols).to_csv(
            os.path.join(tables_dir, f"{tag}_item_total.csv"), index=False
        )

        for gcol in group_cols:
            alpha_by_group(df_sub, item_cols, gcol, min_n_per_group=min_n_per_group).to_csv(
                os.path.join(tables_dir, f"{tag}_alpha_by_{gcol}.csv"), index=False
            )

    # =========================
    # C) DIF per group + DIF stability bootstrap
    # =========================
    if modules.get("dif_screen", True) and len(group_cols) > 0:
        for gcol in group_cols:
            dif_df = dif_screen_linear(
                df_sub=df_sub,
                item_cols=item_cols,
                group_col=gcol,
                min_n_per_group=min_n_per_group,
                standardize_total=True
            )
            dif_df.to_csv(os.path.join(tables_dir, f"{tag}_dif_{gcol}.csv"), index=False)

            if modules.get("uncertainty", False):
                dcfg = bootstrap_cfg.get("dif_stability", {})
                if dcfg.get("enabled", False):
                    n_boot = int(dcfg.get("n_boot", 30))
                    seed = int(dcfg.get("seed", 123))
                    q_thr = float(dcfg.get("q_threshold", 0.05))

                    stab_df = dif_stability(
                        df_sub=df_sub,
                        item_cols=item_cols,
                        group_col=gcol,
                        min_n_per_group=min_n_per_group,
                        q_threshold=q_thr,
                        n_boot=n_boot,
                        seed=seed
                    )
                    stab_df.to_csv(os.path.join(tables_dir, f"{tag}_dif_stability_{gcol}.csv"), index=False)

    # =========================
    # IRT (GRM) AI model (fit on subset)
    # =========================
    if modules.get("irt", False):
        max_rows = int(irt_cfg.get("max_rows", 20000))
        draws = int(irt_cfg.get("draws", 400))
        tune = int(irt_cfg.get("tune", 400))
        chains = int(irt_cfg.get("chains", 2))
        target_accept = float(irt_cfg.get("target_accept", 0.9))
        seed = int(irt_cfg.get("seed", 42))

        complete = df_sub[item_cols].dropna(axis=0, how="any")
        if complete.shape[0] == 0:
            raise ValueError("IRT enabled but no complete rows available after -1 → missing conversion.")

        if complete.shape[0] > max_rows:
            complete = complete.sample(n=max_rows, random_state=seed)

        Y_all, value_map, idx_all = recode_ordinal(df_sub, item_cols, allowed_values)
        Y_all_df = pd.DataFrame(Y_all, index=idx_all)
        Y_sub = Y_all_df.loc[complete.index].to_numpy(dtype=int)

        fit = fit_grm_pymc(Y_sub, draws=draws, tune=tune, chains=chains, target_accept=target_accept, seed=seed)

        pd.DataFrame({
            "row_id": complete.index.astype(str),
            "theta_mean": fit["theta_mean"]
        }).to_csv(os.path.join(tables_dir, f"{tag}_theta_scores_subset.csv"), index=False)

        # item params
        b_mean = fit["b_mean"]
        Kminus1 = b_mean.shape[1]
        rows = []
        for j, item in enumerate(item_cols):
            row = {"item": item, "a_mean": float(fit["a_mean"][j])}
            for k in range(Kminus1):
                row[f"b{k+1}_mean"] = float(b_mean[j, k])
            rows.append(row)
        params_df = pd.DataFrame(rows)
        params_path = os.path.join(tables_dir, f"{tag}_irt_item_params.csv")
        params_df.to_csv(params_path, index=False)

        # Optional: information curve (saved as CSV)
        grid, info = information_curve(
            a_vec=params_df["a_mean"].to_numpy(dtype=float),
            b_mat=params_df[[c for c in params_df.columns if c.startswith("b")]].to_numpy(dtype=float),
        )
        info_df = pd.DataFrame({"theta": grid, "test_information": info})
        info_df.to_csv(os.path.join(tables_dir, f"{tag}_irt_test_information.csv"), index=False)

    # =========================
    # D) Reports (one per group)
    # =========================
    alpha_overall_df = _safe_read_csv(os.path.join(tables_dir, f"{tag}_alpha_overall.csv"))
    item_total_df = _safe_read_csv(os.path.join(tables_dir, f"{tag}_item_total.csv"))
    missing_df = _safe_read_csv(os.path.join(tables_dir, f"{tag}_missingness.csv"))
    floor_ceiling_df = _safe_read_csv(os.path.join(tables_dir, f"{tag}_floor_ceiling.csv"))

    if len(group_cols) == 0:
        verdict = compute_verdict(alpha_overall=alpha_val, dif_df=None)
        write_report_html(
            out_path=os.path.join(reports_dir, f"{tag}_report.html"),
            tag=tag,
            group_col=None,
            alpha_overall_df=alpha_overall_df,
            item_total_df=item_total_df,
            missing_df=missing_df,
            floor_ceiling_df=floor_ceiling_df,
            alpha_by_group_df=None,
            dif_df=None,
            verdict=verdict
        )
    else:
        for gcol in group_cols:
            abg_df = _safe_read_csv(os.path.join(tables_dir, f"{tag}_alpha_by_{gcol}.csv"))
            dif_df = _safe_read_csv(os.path.join(tables_dir, f"{tag}_dif_{gcol}.csv"))

            verdict = compute_verdict(alpha_overall=alpha_val, dif_df=dif_df)
            write_report_html(
                out_path=os.path.join(reports_dir, f"{tag}_report_{gcol}.html"),
                tag=f"{tag} ({gcol})",
                group_col=gcol,
                alpha_overall_df=alpha_overall_df,
                item_total_df=item_total_df,
                missing_df=missing_df,
                floor_ceiling_df=floor_ceiling_df,
                alpha_by_group_df=abg_df,
                dif_df=dif_df,
                verdict=verdict
            )

    print("✅ Run complete.")
    print(f"✅ Tables written to:  {tables_dir}")
    print(f"✅ Reports written to: {reports_dir}")


if __name__ == "__main__":
    main()




