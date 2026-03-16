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
    """Z-update with group lasso penalty across groups (vectorized)."""
    K = len(thetas)
    p = thetas[0].shape[0]

    # Stack into (K, p, p) arrays
    V = np.array([thetas[k] + Us[k] for k in range(K)])  # (K, p, p)

    # Soft-threshold for sparsity (all elements)
    V = _soft_threshold(V, lambda1 / rho)

    # Group-lasso shrinkage: for each (i,j), shrink the K-vector jointly
    # norm across groups: shape (p, p)
    norms = np.sqrt(np.sum(V**2, axis=0))  # (p, p)
    norms_safe = np.where(norms > 0, norms, 1.0)
    scale = np.maximum(1.0 - lambda2 / (rho * norms_safe), 0.0)  # (p, p)
    scale = np.where(norms > 0, scale, 0.0)
    V = V * scale[np.newaxis, :, :]  # broadcast (K, p, p)

    # Restore diagonal (no penalty)
    if not penalize_diagonal:
        for k in range(K):
            np.fill_diagonal(V[k], thetas[k].diagonal() + Us[k].diagonal())

    return [V[k] for k in range(K)]


def _z_update_fused_penalty(
    thetas: list[np.ndarray],
    Us: list[np.ndarray],
    lambda1: float,
    lambda2: float,
    rho: float,
    penalize_diagonal: bool,
) -> list[np.ndarray]:
    """Z-update with fused lasso penalty across groups (vectorized)."""
    K = len(thetas)
    p = thetas[0].shape[0]

    # Stack into (K, p, p)
    V = np.array([thetas[k] + Us[k] for k in range(K)])

    # Soft-threshold for sparsity
    V = _soft_threshold(V, lambda1 / rho)

    # Fused penalty: shrink differences across groups
    if K == 2:
        # Closed-form for K=2: vectorized over all (i,j)
        mean_v = (V[0] + V[1]) / 2              # (p, p)
        half_diff = (V[0] - V[1]) / 2           # (p, p)
        thresh = lambda2 / (2 * rho)
        fuse_mask = np.abs(half_diff) <= thresh  # (p, p)
        # Where fused: both become mean
        V[0] = np.where(fuse_mask, mean_v, V[0])
        V[1] = np.where(fuse_mask, mean_v, V[1])
        # Where not fused: shrink toward each other
        shrink = np.sign(half_diff) * thresh
        V[0] = np.where(~fuse_mask, mean_v + (half_diff - shrink), V[0])
        V[1] = np.where(~fuse_mask, mean_v - (half_diff - shrink), V[1])
    else:
        # General case K>2: apply pairwise fused proximal per element
        # Vectorize over (i,j) by reshaping to (K, p*p)
        V_flat = V.reshape(K, -1)  # (K, p*p)
        lam_scaled = lambda2 / rho
        for col in range(V_flat.shape[1]):
            V_flat[:, col] = _fused_proximal(V_flat[:, col], lam_scaled)
        V = V_flat.reshape(K, p, p)

    # Restore diagonal (no penalty)
    if not penalize_diagonal:
        for k in range(K):
            np.fill_diagonal(V[k], thetas[k].diagonal() + Us[k].diagonal())

    return [V[k] for k in range(K)]


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
        Thetas = [
            _theta_update(S[k], Zs[k], Us[k], n_samples[k], rho)
            for k in range(K)
        ]

        # Z update (across groups)
        Zs_new = z_update(Thetas, Us, lambda1, lambda2, rho, penalize_diagonal)

        # U update
        Us = [Us[k] + Thetas[k] - Zs_new[k] for k in range(K)]

        # Check convergence (primal residual)
        primal_res = sum(
            np.linalg.norm(Thetas[k] - Zs_new[k]) for k in range(K)
        )

        Zs = Zs_new

        if primal_res < tol:
            break

    # Return the Z variables (consensus) as final precision estimates
    return Zs
