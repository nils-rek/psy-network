"""Group network estimation via Joint Graphical Lasso."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .._types import PenaltyType, SelectionCriterion
from ..network import Network
from ._jgl import joint_graphical_lasso
from ._selection import select_lambdas
from .network import GroupNetwork

if TYPE_CHECKING:
    pass


def estimate_group_network(
    data: pd.DataFrame | list[pd.DataFrame],
    *,
    group_col: str | None = None,
    penalty: str | PenaltyType = "fused",
    criterion: str | SelectionCriterion = "ebic",
    gamma: float = 0.5,
    lambda1: float | None = None,
    lambda2: float | None = None,
    search: str = "sequential",
    n_lambda1: int = 30,
    n_lambda2: int = 30,
    lambda1_min_ratio: float = 0.01,
    lambda2_min_ratio: float = 0.01,
    threshold: float = 1e-4,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> GroupNetwork:
    """Estimate group networks using Joint Graphical Lasso.

    Parameters
    ----------
    data : pd.DataFrame or list[pd.DataFrame]
        Either a single DataFrame with a ``group_col`` column, or a list
        of DataFrames (one per group).
    group_col : str, optional
        Column name identifying groups (required if data is a single DataFrame).
    penalty : str
        ``"fused"`` for fused lasso or ``"group"`` for group lasso penalty.
    criterion : str
        ``"ebic"``, ``"bic"``, or ``"aic"`` for lambda selection.
    gamma : float
        EBIC gamma (sparsity tuning, only used with ``"ebic"``).
    lambda1, lambda2 : float, optional
        Manual lambda values. If both are provided, skips selection.
    search : str
        ``"sequential"`` or ``"simultaneous"`` lambda search strategy.
    n_lambda1, n_lambda2 : int
        Grid sizes for lambda search.
    lambda1_min_ratio, lambda2_min_ratio : float
        Min-to-max ratio for log-spaced lambda grids.
    threshold : float
        Threshold for zeroing small partial correlations.
    max_iter, tol : int, float
        ADMM parameters.

    Returns
    -------
    GroupNetwork
    """
    penalty = PenaltyType(penalty).value
    criterion = SelectionCriterion(criterion).value

    # Parse input data into per-group DataFrames
    if isinstance(data, list):
        group_dfs = {f"Group{i+1}": df for i, df in enumerate(data)}
    else:
        if group_col is None:
            raise ValueError("group_col is required when data is a single DataFrame")
        group_dfs = {
            name: grp.drop(columns=[group_col]).reset_index(drop=True)
            for name, grp in data.groupby(group_col)
        }

    group_labels = sorted(group_dfs.keys())
    var_names = list(group_dfs[group_labels[0]].columns)

    # Compute empirical covariance matrices
    S_list = []
    n_list = []
    for label in group_labels:
        df = group_dfs[label]
        cov = df.corr().values.copy()
        S_list.append(cov)
        n_list.append(len(df))

    # Lambda selection or manual
    if lambda1 is not None and lambda2 is not None:
        precisions = joint_graphical_lasso(
            S_list, n_list, lambda1, lambda2, penalty, max_iter, tol,
        )
        best_l1, best_l2 = lambda1, lambda2
    else:
        best_l1, best_l2, precisions = select_lambdas(
            S_list, n_list, penalty, criterion, gamma, search,
            n_lambda1, n_lambda2, lambda1_min_ratio, lambda2_min_ratio,
            max_iter, tol,
        )

    # Convert precision matrices to partial correlation networks
    networks = {}
    for k, label in enumerate(group_labels):
        P = precisions[k]
        diag = np.sqrt(np.abs(np.diag(P)))
        diag[diag < 1e-10] = 1.0
        pcor = -P / np.outer(diag, diag)
        np.fill_diagonal(pcor, 0.0)
        pcor[np.abs(pcor) < threshold] = 0.0

        networks[label] = Network(
            adjacency=pcor,
            labels=var_names,
            method=f"JGL-{penalty}",
            n_observations=n_list[k],
            weighted=True,
            signed=True,
            directed=False,
        )

    return GroupNetwork(
        networks=networks,
        group_labels=group_labels,
        lambda1=best_l1,
        lambda2=best_l2,
        penalty=penalty,
        criterion=criterion,
    )


# Re-exports
from .bootstrap import GroupBootstrapResult, bootnet_group  # noqa: E402

__all__ = [
    "estimate_group_network",
    "bootnet_group",
    "GroupNetwork",
    "GroupBootstrapResult",
]
