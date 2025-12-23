from __future__ import annotations

import numpy as np
from .theta_scoring import grm_category_probs


def item_information_grm(theta: float, a: float, b: np.ndarray) -> float:
    """
    Approximate item information for GRM at theta.
    Uses a practical numerical derivative of log probability.
    """
    K = b.shape[0] + 1
    eps = 1e-3

    # Compute category probs at theta, theta+eps, theta-eps
    p0 = grm_category_probs(theta, a, b)
    p_plus = grm_category_probs(theta + eps, a, b)
    p_minus = grm_category_probs(theta - eps, a, b)

    # derivative of log p_k w.r.t theta (finite difference)
    dlogp = (np.log(p_plus) - np.log(p_minus)) / (2 * eps)

    # Fisher information: sum_k p_k * (d/dtheta log p_k)^2
    info = float(np.sum(p0 * (dlogp ** 2)))
    return info


def test_information(theta: float, a_vec: np.ndarray, b_mat: np.ndarray) -> float:
    """
    Test information = sum of item informations.
    """
    total = 0.0
    for j in range(len(a_vec)):
        total += item_information_grm(theta, float(a_vec[j]), b_mat[j])
    return float(total)


def information_curve(a_vec: np.ndarray, b_mat: np.ndarray, grid: np.ndarray | None = None):
    """
    Compute test information curve over a theta grid.
    Returns (grid, info_values)
    """
    if grid is None:
        grid = np.linspace(-4, 4, 81)
    infos = np.array([test_information(float(t), a_vec, b_mat) for t in grid], dtype=float)
    return grid, infos
