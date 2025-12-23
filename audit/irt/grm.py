from __future__ import annotations

import numpy as np
import pandas as pd


def recode_ordinal(items: pd.DataFrame, item_cols: list[str], allowed_values: list[float] | None):
    """
    Recode item responses into ordinal integers 0..K-1 for GRM.
    Returns:
      Y (N,J) int array, value_map dict, row_index (original df index)
    """
    X = items[item_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

    if allowed_values is None:
        # infer from data (only safe if data is clean)
        uniq = sorted(set(np.unique(X.to_numpy())))
        allowed_values = [float(u) for u in uniq if np.isfinite(u)]

    allowed_values = [float(v) for v in allowed_values]
    value_map = {v: i for i, v in enumerate(sorted(allowed_values))}

    # validate values
    bad_mask = ~X.isin(list(value_map.keys()))
    if bad_mask.any().any():
        bad_vals = sorted(set(pd.unique(X[bad_mask].stack().head(20))))
        raise ValueError(f"IRT recode failed: values not in allowed_values. Examples: {bad_vals}")

    Y = X.replace(value_map).to_numpy(dtype=int)
    return Y, value_map, X.index


def fit_grm_pymc(
    Y: np.ndarray,
    draws: int = 400,
    tune: int = 400,
    chains: int = 2,
    target_accept: float = 0.9,
    seed: int = 42,
):
    """
    Fit a Samejima Graded Response Model using PyMC.
    Returns posterior means for theta, a, b.
    """
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az

    N, J = Y.shape
    K = int(Y.max() + 1)  # categories 0..K-1
    if K < 2:
        raise ValueError("Need >=2 categories for GRM.")

    with pm.Model() as model:
        theta = pm.Normal("theta", mu=0.0, sigma=1.0, shape=N)
        a = pm.LogNormal("a", mu=0.0, sigma=0.6, shape=J)

        # ordered thresholds per item (J, K-1)
        b_raw = pm.Normal("b_raw", mu=0.0, sigma=1.0, shape=(J, K - 1))
        b = pm.Deterministic("b", pt.sort(b_raw, axis=1))

        def sigmoid(x):
            return 1 / (1 + pt.exp(-x))

        # P(Y >= k) for k=1..K-1
        theta_ = theta[:, None, None]
        a_ = a[None, :, None]
        b_ = b[None, :, :]
        P_ge = sigmoid(a_ * (theta_ - b_))  # (N,J,K-1)

        # category probabilities
        P0 = 1 - P_ge[:, :, 0]
        probs = [P0]
        for k in range(1, K - 1):
            probs.append(P_ge[:, :, k - 1] - P_ge[:, :, k])
        probs.append(P_ge[:, :, K - 2])
        p = pt.stack(probs, axis=2)  # (N,J,K)

        pm.Categorical("y", p=p, observed=Y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=True,
        )

    post = idata.posterior
    theta_mean = post["theta"].mean(dim=("chain", "draw")).values
    a_mean = post["a"].mean(dim=("chain", "draw")).values
    b_mean = post["b"].mean(dim=("chain", "draw")).values

    rhat = az.rhat(idata)
    return {
        "theta_mean": theta_mean,
        "a_mean": a_mean,
        "b_mean": b_mean,
        "rhat": rhat,
    }
