"""Joint Graphical Lasso via ADMM.

Reference: Danaher, Wang & Witten (2014), "The joint graphical lasso for
inverse covariance estimation across multiple classes", JRSS-B.
"""

from __future__ import annotations

import numpy as np


def _soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    """Element-wise soft thresholding: sign(x) * max(|x| - lam, 0)."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def _theta_update(
    S_k: np.ndarray,
    Z_k: np.ndarray,
    U_k: np.ndarray,
    n_k: int,
    rho: float,
) -> np.ndarray:
    """Proximal step for Theta_k via eigendecomposition.

    Solves: min_Theta  n_k * [-logdet(Theta) + tr(S_k @ Theta)]
                       + (rho/2) * ||Theta - Z_k + U_k||_F^2
    """
    p = S_k.shape[0]
    # Target for proximal
    A = Z_k - U_k - (n_k / rho) * S_k
    # A is symmetric; eigendecompose
    A_sym = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    # Closed-form solution: each eigenvalue becomes
    # (eigval + sqrt(eigval^2 + 4*n_k/rho)) / 2
    xi = (eigvals + np.sqrt(eigvals**2 + 4 * n_k / rho)) / 2
    Theta_k = eigvecs @ np.diag(xi) @ eigvecs.T
    return (Theta_k + Theta_k.T) / 2


def _z_update_group_penalty(
    thetas: list[np.ndarray],
    Us: list[np.ndarray],
    lambda1: float,
    lambda2: float,
    rho: float,
    penalize_diagonal: bool,
) -> list[np.ndarray]:
    """Z-update with group lasso penalty across groups."""
    K = len(thetas)
    p = thetas[0].shape[0]
    Zs = [np.zeros_like(t) for t in thetas]

    for i in range(p):
        for j in range(p):
            if i == j and not penalize_diagonal:
                # No penalty on diagonal — just average
                for k in range(K):
                    Zs[k][i, j] = thetas[k][i, j] + Us[k][i, j]
                continue

            # Collect v = [theta_k[i,j] + U_k[i,j] for each k]
            v = np.array([thetas[k][i, j] + Us[k][i, j] for k in range(K)])
            # First: soft-threshold each element by lambda1/rho (sparsity)
            v = _soft_threshold(v, lambda1 / rho)
            # Then: group-lasso shrinkage across groups
            norm_v = np.linalg.norm(v)
            if norm_v > 0:
                scale = max(1.0 - lambda2 / (rho * norm_v), 0.0)
                v = v * scale
            for k in range(K):
                Zs[k][i, j] = v[k]

    return Zs


def _z_update_fused_penalty(
    thetas: list[np.ndarray],
    Us: list[np.ndarray],
    lambda1: float,
    lambda2: float,
    rho: float,
    penalize_diagonal: bool,
) -> list[np.ndarray]:
    """Z-update with fused lasso penalty across groups."""
    K = len(thetas)
    p = thetas[0].shape[0]
    Zs = [np.zeros_like(t) for t in thetas]

    for i in range(p):
        for j in range(p):
            if i == j and not penalize_diagonal:
                for k in range(K):
                    Zs[k][i, j] = thetas[k][i, j] + Us[k][i, j]
                continue

            v = np.array([thetas[k][i, j] + Us[k][i, j] for k in range(K)])
            # Soft-threshold for sparsity
            v = _soft_threshold(v, lambda1 / rho)

            if K == 2:
                # Closed-form fused lasso proximal for K=2
                # Minimize (rho/2) * sum_k(z_k - v_k)^2 + lambda2 * |z_1 - z_2|
                mean_v = (v[0] + v[1]) / 2
                half_diff = (v[0] - v[1]) / 2
                thresh = lambda2 / (2 * rho)
                if abs(half_diff) <= thresh:
                    v[0] = v[1] = mean_v
                else:
                    shrink = np.sign(half_diff) * thresh
                    v[0] = mean_v + (half_diff - shrink)
                    v[1] = mean_v - (half_diff - shrink)
            else:
                # General case: iterative fused lasso proximal via FLSA
                v = _fused_proximal(v, lambda2 / rho)

            for k in range(K):
                Zs[k][i, j] = v[k]

    return Zs


def _fused_proximal(v: np.ndarray, lam: float, max_iter: int = 50) -> np.ndarray:
    """Proximal operator for the fused penalty sum_{k<l} |z_k - z_l|.

    Uses an iterative approach: for each adjacent pair, apply pairwise fusion.
    """
    z = v.copy()
    K = len(z)
    for _ in range(max_iter):
        z_old = z.copy()
        for k in range(K - 1):
            diff = z[k] - z[k + 1]
            if abs(diff) <= 2 * lam:
                mean_val = (z[k] + z[k + 1]) / 2
                z[k] = mean_val
                z[k + 1] = mean_val
            else:
                shrink = np.sign(diff) * lam
                z[k] -= shrink
                z[k + 1] += shrink
        if np.max(np.abs(z - z_old)) < 1e-8:
            break
    return z


def joint_graphical_lasso(
    S: list[np.ndarray],
    n_samples: list[int],
    lambda1: float,
    lambda2: float,
    penalty: str = "fused",
    max_iter: int = 500,
    tol: float = 1e-4,
    rho: float = 1.0,
    penalize_diagonal: bool = False,
) -> list[np.ndarray]:
    """Joint Graphical Lasso via ADMM.

    Parameters
    ----------
    S : list[np.ndarray]
        K empirical covariance matrices (p x p each).
    n_samples : list[int]
        Per-group sample sizes.
    lambda1 : float
        Sparsity penalty (within-group).
    lambda2 : float
        Similarity penalty (cross-group).
    penalty : str
        ``"fused"`` for fused lasso penalty on differences, or
        ``"group"`` for group lasso penalty across groups.
    max_iter : int
        Maximum ADMM iterations.
    tol : float
        Convergence tolerance on primal residual.
    rho : float
        ADMM augmented Lagrangian parameter.
    penalize_diagonal : bool
        Whether to penalize diagonal elements.

    Returns
    -------
    list[np.ndarray]
        K estimated precision matrices.
    """
    K = len(S)
    p = S[0].shape[0]

    # Initialize
    Thetas = [np.eye(p) for _ in range(K)]
    Zs = [np.eye(p) for _ in range(K)]
    Us = [np.zeros((p, p)) for _ in range(K)]

    z_update = (
        _z_update_fused_penalty if penalty == "fused"
        else _z_update_group_penalty
    )

    for iteration in range(max_iter):
        # Theta update (per group)
        Thetas_new = [
            _theta_update(S[k], Zs[k], Us[k], n_samples[k], rho)
            for k in range(K)
        ]

        # Z update (across groups)
        Zs_new = z_update(Thetas_new, Us, lambda1, lambda2, rho, penalize_diagonal)

        # U update
        Us_new = [
            Us[k] + Thetas_new[k] - Zs_new[k]
            for k in range(K)
        ]

        # Check convergence (primal residual)
        primal_res = sum(
            np.linalg.norm(Thetas_new[k] - Zs_new[k]) for k in range(K)
        )
        if primal_res < tol:
            Thetas = Thetas_new
            break

        Thetas = Thetas_new
        Zs = Zs_new
        Us = Us_new

    # Return the Z variables (consensus) as final precision estimates
    # They incorporate the penalty structure
    return Zs
