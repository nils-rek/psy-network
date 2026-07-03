"""Joint Graphical Lasso via ADMM.

Reference: Danaher, Wang & Witten (2014), "The joint graphical lasso for
inverse covariance estimation across multiple classes", JRSS-B.
"""

from __future__ import annotations

import warnings

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
    """Z-update with fused lasso penalty across groups (vectorized).

    The prox of ``lambda1 * ||.||_1 + lambda2 * sum_{k<l} |z_k - z_l|``
    decomposes as L1 soft-thresholding applied to the output of the fused
    prox (Friedman et al., 2007), so fusion is applied first.
    """
    K = len(thetas)
    p = thetas[0].shape[0]

    # Stack into (K, p, p)
    V = np.array([thetas[k] + Us[k] for k in range(K)])

    # Fused penalty first: shrink differences across groups
    if K == 2:
        # Closed-form for K=2: z0 - z1 = soft(v0 - v1, 2*lambda2/rho),
        # i.e. the half-difference is soft-thresholded at lambda2/rho.
        mean_v = (V[0] + V[1]) / 2              # (p, p)
        half_diff = (V[0] - V[1]) / 2           # (p, p)
        shrunk = _soft_threshold(half_diff, lambda2 / rho)
        V[0] = mean_v + shrunk
        V[1] = mean_v - shrunk
    else:
        # General case K>2: exact all-pairs fused prox per element
        # Vectorize over (i,j) by reshaping to (K, p*p)
        V_flat = V.reshape(K, -1)  # (K, p*p)
        lam_scaled = lambda2 / rho
        for col in range(V_flat.shape[1]):
            V_flat[:, col] = _fused_proximal(V_flat[:, col], lam_scaled)
        V = V_flat.reshape(K, p, p)

    # Then soft-threshold for sparsity
    V = _soft_threshold(V, lambda1 / rho)

    # Restore diagonal (no penalty)
    if not penalize_diagonal:
        for k in range(K):
            np.fill_diagonal(V[k], thetas[k].diagonal() + Us[k].diagonal())

    return [V[k] for k in range(K)]


def _fused_proximal(v: np.ndarray, lam: float) -> np.ndarray:
    """Exact proximal operator for the all-pairs fused penalty.

    Solves ``min_z 0.5 * ||z - v||^2 + lam * sum_{k<l} |z_k - z_l|``.

    The solution preserves the ordering of ``v``; subject to that ordering
    the objective is separable with linear tilts, so the minimizer is the
    isotonic regression (PAVA) of ``v_(k) - lam * (2k - K + 1)`` over the
    sorted values.
    """
    K = len(v)
    order = np.argsort(v, kind="stable")
    w = v[order] - lam * (2.0 * np.arange(K) - K + 1)

    # PAVA: non-decreasing isotonic regression with unit weights
    blocks: list[list[float]] = []  # [mean, count]
    for x in w:
        blocks.append([x, 1])
        while len(blocks) > 1 and blocks[-2][0] >= blocks[-1][0]:
            m2, c2 = blocks.pop()
            m1, c1 = blocks.pop()
            blocks.append([(m1 * c1 + m2 * c2) / (c1 + c2), c1 + c2])

    z_sorted = np.concatenate([np.full(int(c), m) for m, c in blocks])
    z = np.empty_like(v)
    z[order] = z_sorted
    return z


def joint_graphical_lasso(
    S: list[np.ndarray],
    n_samples: list[int],
    lambda1: float,
    lambda2: float,
    penalty: str = "fused",
    max_iter: int = 500,
    tol: float = 1e-4,
    tol_abs: float = 1e-6,
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
        Relative convergence tolerance (``eps_rel`` in Boyd et al., 2011,
        §3.3), applied to both the primal and dual residuals.
    tol_abs : float
        Absolute convergence tolerance (``eps_abs``).
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

    sqrt_dim = np.sqrt(K * p * p)
    converged = False

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

        # Primal and dual residuals with relative scaling (Boyd et al. §3.3)
        primal_res = np.sqrt(sum(
            np.linalg.norm(Thetas[k] - Zs_new[k]) ** 2 for k in range(K)
        ))
        dual_res = rho * np.sqrt(sum(
            np.linalg.norm(Zs_new[k] - Zs[k]) ** 2 for k in range(K)
        ))
        theta_norm = np.sqrt(sum(np.linalg.norm(T) ** 2 for T in Thetas))
        z_norm = np.sqrt(sum(np.linalg.norm(Z) ** 2 for Z in Zs_new))
        u_norm = np.sqrt(sum(np.linalg.norm(U) ** 2 for U in Us))
        eps_pri = sqrt_dim * tol_abs + tol * max(theta_norm, z_norm)
        eps_dual = sqrt_dim * tol_abs + tol * rho * u_norm

        Zs = Zs_new

        if primal_res <= eps_pri and dual_res <= eps_dual:
            converged = True
            break

    if not converged:
        warnings.warn(
            f"Joint graphical lasso ADMM did not converge within "
            f"{max_iter} iterations (primal residual {primal_res:.2e}, "
            f"dual residual {dual_res:.2e}). Results may be inaccurate; "
            f"consider increasing max_iter.",
            UserWarning,
            stacklevel=2,
        )

    # Return the Z variables (consensus) as final precision estimates
    return Zs
