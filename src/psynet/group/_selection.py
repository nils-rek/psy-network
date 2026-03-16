"""Lambda selection for Joint Graphical Lasso."""

from __future__ import annotations

import numpy as np

from ._jgl import joint_graphical_lasso


def _group_ebic(
    precisions: list[np.ndarray],
    S: list[np.ndarray],
    n_samples: list[int],
    gamma: float,
    criterion: str,
) -> float:
    """Compute information criterion summed across groups.

    Extends the single-group EBIC pattern:
    IC = sum_k [ -2 * loglik_k + E_k * pen(n_k, p) ]
    """
    total = 0.0
    for k, (P, Sk, nk) in enumerate(zip(precisions, S, n_samples)):
        p = P.shape[0]
        sign, logdet = np.linalg.slogdet(P)
        if sign <= 0:
            return np.inf
        loglik = (nk / 2) * (logdet - np.trace(Sk @ P))

        # Count unique non-zero off-diagonal edges
        upper = np.triu(P, k=1)
        n_edges = np.count_nonzero(np.abs(upper) > 1e-10)

        if criterion == "ebic":
            ic = -2 * loglik + n_edges * np.log(nk) + 4 * n_edges * gamma * np.log(p)
        elif criterion == "bic":
            ic = -2 * loglik + n_edges * np.log(nk)
        else:  # aic
            ic = -2 * loglik + 2 * n_edges
        total += ic
    return total


def select_lambdas(
    S: list[np.ndarray],
    n_samples: list[int],
    penalty: str,
    criterion: str = "ebic",
    gamma: float = 0.5,
    search: str = "sequential",
    n_lambda1: int = 30,
    n_lambda2: int = 30,
    lambda1_min_ratio: float = 0.01,
    lambda2_min_ratio: float = 0.01,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> tuple[float, float, list[np.ndarray]]:
    """Select lambda1 and lambda2 via information criterion.

    Parameters
    ----------
    S : list[np.ndarray]
        K empirical covariance matrices.
    n_samples : list[int]
        Per-group sample sizes.
    penalty : str
        ``"fused"`` or ``"group"``.
    criterion : str
        ``"ebic"``, ``"bic"``, or ``"aic"``.
    gamma : float
        EBIC gamma parameter (only used when criterion="ebic").
    search : str
        ``"sequential"`` (faster: grid lambda1 then lambda2) or
        ``"simultaneous"`` (2D grid search).
    n_lambda1, n_lambda2 : int
        Grid sizes for lambda1 and lambda2.
    lambda1_min_ratio, lambda2_min_ratio : float
        Min-to-max ratio for lambda grids (log-spaced).
    max_iter, tol : int, float
        ADMM parameters.

    Returns
    -------
    tuple[float, float, list[np.ndarray]]
        Best (lambda1, lambda2, precision_matrices).
    """
    p = S[0].shape[0]

    # Derive lambda_max from pooled off-diagonal maximum
    pooled = np.mean(S, axis=0)
    lambda1_max = np.max(np.abs(np.triu(pooled, k=1)))
    if lambda1_max < 1e-10:
        lambda1_max = 1.0
    lambda1_min = lambda1_max * lambda1_min_ratio

    lambda1_grid = np.logspace(
        np.log10(lambda1_max), np.log10(lambda1_min), n_lambda1,
    )

    lambda2_max = lambda1_max
    lambda2_min = lambda2_max * lambda2_min_ratio
    lambda2_grid = np.logspace(
        np.log10(lambda2_max), np.log10(lambda2_min), n_lambda2,
    )

    if search == "sequential":
        return _search_sequential(
            S, n_samples, penalty, criterion, gamma,
            lambda1_grid, lambda2_grid, max_iter, tol,
        )
    else:
        return _search_simultaneous(
            S, n_samples, penalty, criterion, gamma,
            lambda1_grid, lambda2_grid, max_iter, tol,
        )


def _search_sequential(
    S, n_samples, penalty, criterion, gamma,
    lambda1_grid, lambda2_grid, max_iter, tol,
):
    """Sequential search: optimize lambda1 with lambda2=0, then lambda2."""
    best_ic = np.inf
    best_l1 = lambda1_grid[0]
    best_prec = None

    # Phase 1: search lambda1 with lambda2=0
    for l1 in lambda1_grid:
        try:
            prec = joint_graphical_lasso(
                S, n_samples, l1, 0.0, penalty, max_iter, tol,
            )
            ic = _group_ebic(prec, S, n_samples, gamma, criterion)
            if ic < best_ic:
                best_ic = ic
                best_l1 = l1
                best_prec = prec
        except Exception:
            continue

    # Phase 2: search lambda2 with fixed lambda1
    best_l2 = 0.0
    for l2 in lambda2_grid:
        try:
            prec = joint_graphical_lasso(
                S, n_samples, best_l1, l2, penalty, max_iter, tol,
            )
            ic = _group_ebic(prec, S, n_samples, gamma, criterion)
            if ic < best_ic:
                best_ic = ic
                best_l2 = l2
                best_prec = prec
        except Exception:
            continue

    if best_prec is None:
        raise RuntimeError("JGL failed to converge at all lambda values")

    return best_l1, best_l2, best_prec


def _search_simultaneous(
    S, n_samples, penalty, criterion, gamma,
    lambda1_grid, lambda2_grid, max_iter, tol,
):
    """Simultaneous 2D grid search over lambda1 × lambda2."""
    best_ic = np.inf
    best_l1 = lambda1_grid[0]
    best_l2 = lambda2_grid[0]
    best_prec = None

    for l1 in lambda1_grid:
        for l2 in lambda2_grid:
            try:
                prec = joint_graphical_lasso(
                    S, n_samples, l1, l2, penalty, max_iter, tol,
                )
                ic = _group_ebic(prec, S, n_samples, gamma, criterion)
                if ic < best_ic:
                    best_ic = ic
                    best_l1 = l1
                    best_l2 = l2
                    best_prec = prec
            except Exception:
                continue

    if best_prec is None:
        raise RuntimeError("JGL failed to converge at all lambda values")

    return best_l1, best_l2, best_prec
