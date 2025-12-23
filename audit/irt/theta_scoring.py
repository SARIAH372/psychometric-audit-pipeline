from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def grm_category_probs(theta: float, a: float, b: np.ndarray) -> np.ndarray:
    """
    GRM category probabilities for a single item.

    Parameters
    ----------
    theta : float
        Latent trait value
    a : float
        Discrimination parameter
    b : np.ndarray shape (K-1,)
        Ordered thresholds

    Returns
    -------
    p : np.ndarray shape (K,)
        Probabilities for categories 0..K-1
    """
    # P(Y >= k) for k=1..K-1
    # b has length K-1
    P_ge = _sigmoid(a * (theta - b))  # shape (K-1,)

    # Convert to category probs
    K = b.shape[0] + 1
    p = np.empty(K, dtype=float)
    p[0] = 1.0 - P_ge[0]
    for k in range(1, K - 1):
        p[k] = P_ge[k - 1] - P_ge[k]
    p[K - 1] = P_ge[K - 2]

    # Numerical safety
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p


def loglik_grm(theta: float, y: np.ndarray, a_vec: np.ndarray, b_mat: np.ndarray) -> float:
    """
    Log-likelihood for a response vector y given theta under GRM.
    y should be integer-coded 0..K-1 with no missing.

    a_vec shape (J,)
    b_mat shape (J, K-1)
    """
    ll = 0.0
    for j in range(len(a_vec)):
        p = grm_category_probs(theta, a_vec[j], b_mat[j])
        ll += float(np.log(p[int(y[j])]))
    return ll


def theta_map_newton(
    y: np.ndarray,
    a_vec: np.ndarray,
    b_mat: np.ndarray,
    prior_sd: float = 1.0,
    max_iter: int = 25,
    tol: float = 1e-4,
    init: float = 0.0
) -> float:
    """
    Compute MAP estimate of theta using Newton updates with a Normal(0, prior_sd^2) prior.

    This is a practical scorer (fast, explainable) used after fitting.
    """
    theta = float(init)

    # Finite difference gradient/Hessian for stability (simple + robust)
    eps = 1e-3

    for _ in range(max_iter):
        # posterior = loglik + logprior
        def post(t):
            return loglik_grm(t, y, a_vec, b_mat) - 0.5 * (t / prior_sd) ** 2

        f0 = post(theta)
        f1 = post(theta + eps)
        f_1 = post(theta - eps)

        # gradient approx
        g = (f1 - f_1) / (2 * eps)
        # hessian approx
        h = (f1 - 2 * f0 + f_1) / (eps ** 2)

        # Add small damping if h is too small
        if abs(h) < 1e-8:
            break

        step = g / h
        theta_new = theta - step

        if abs(theta_new - theta) < tol:
            theta = theta_new
            break

        theta = theta_new

        # bound theta to reasonable range to avoid numerical blow-up
        theta = float(np.clip(theta, -6, 6))

    return float(theta)


def score_theta_dataframe(
    df_items_ordinal: pd.DataFrame,
    a_vec: np.ndarray,
    b_mat: np.ndarray,
    prior_sd: float = 1.0
) -> pd.DataFrame:
    """
    Score theta (MAP) for each row of an ordinal-coded dataframe.
    df_items_ordinal contains integers 0..K-1 and no missing.

    Returns a DataFrame with theta_map.
    """
    thetas = []
    for _, row in df_items_ordinal.iterrows():
        y = row.to_numpy(dtype=int)
        thetas.append(theta_map_newton(y, a_vec, b_mat, prior_sd=prior_sd))
    return pd.DataFrame({"theta_map": thetas}, index=df_items_ordinal.index)


def load_item_params_csv(path: str) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load item parameters from the CSV you already save:
      columns: item, a_mean, b1_mean, b2_mean, ...

    Returns:
      a_vec shape (J,)
      b_mat shape (J, K-1)
      item_order list[str]
    """
    df = pd.read_csv(path)
    item_order = df["item"].astype(str).tolist()
    a_vec = df["a_mean"].astype(float).to_numpy()

    b_cols = [c for c in df.columns if c.startswith("b") and c.endswith("_mean")]
    b_cols_sorted = sorted(b_cols, key=lambda x: int(x.split("_")[0][1:]))  # b1_mean, b2_mean...
    b_mat = df[b_cols_sorted].astype(float).to_numpy()

    return a_vec, b_mat, item_order
