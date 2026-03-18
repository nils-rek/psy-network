"""Stability analysis — CS-coefficient and difference tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .results import BootstrapResult


def cs_coefficient(
    boot_result: BootstrapResult,
    statistic: str = "strength",
    threshold: float = 0.7,
    quantile: float = 0.05,
) -> float:
    """Compute the CS-coefficient (correlation stability coefficient).

    The CS-coefficient is defined as the maximum proportion of cases that
    can be dropped such that, with probability ≥ (1 - quantile), the
    correlation between original and subsample centralities remains ≥ threshold.

    Uses a proportion-of-samples method matching R's ``corStability()``:
    for each drop proportion, the fraction of bootstrap correlations
    exceeding *threshold* must be ≥ ``1 - quantile``.

    Parameters
    ----------
    boot_result : BootstrapResult
        Result from a case-dropping bootstrap.
    statistic : str
        Centrality measure to evaluate.
    threshold : float
        Minimum acceptable correlation (default 0.7).
    quantile : float
        Significance level (default 0.05). The proportion of bootstrap
        samples with correlation above *threshold* must be ≥ ``1 - quantile``.

    Returns
    -------
    float
        The CS-coefficient (between 0 and 1). Higher is better;
        ≥ 0.5 is considered acceptable, ≥ 0.7 is good.
    """
    if boot_result.case_drop_correlations is None:
        raise ValueError("CS-coefficient requires a case-dropping bootstrap result")

    df = boot_result.case_drop_correlations
    sub = df[df["statistic"] == statistic].copy()

    if sub.empty:
        raise ValueError(f"No case-drop results for statistic {statistic!r}")

    proportions = sorted(sub["proportion"].unique(), reverse=True)

    best_prop = 0.0
    for prop in proportions:
        prop_corrs = sub[sub["proportion"] == prop]["correlation"].dropna()
        if len(prop_corrs) == 0:
            continue
        p_above = np.mean(prop_corrs > threshold)
        if p_above >= (1 - quantile):
            # proportion is fraction *retained*, so dropped = 1 - prop
            best_prop = max(best_prop, 1.0 - prop)

    return best_prop


def difference_test(
    boot_result: BootstrapResult,
    statistic: str = "edge",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Pairwise bootstrap difference test.

    For edges: tests whether edge_i differs significantly from edge_j.
    For centrality: tests whether node_i's centrality differs from node_j's.

    Parameters
    ----------
    boot_result : BootstrapResult
        Result from a nonparametric bootstrap.
    statistic : str
        Statistic to test (``"edge"``, ``"strength"``, etc.).
    alpha : float
        Significance level.

    Returns
    -------
    pd.DataFrame
        Square matrix of p-values (or boolean significant flags).
    """
    df = boot_result.boot_statistics
    sub = df[(df["statistic"] == statistic) & (df["boot_id"] != "original")]

    if statistic == "edge":
        # Create edge identifiers
        sub = sub.copy()
        sub["edge_id"] = sub["node1"] + " -- " + sub["node2"]
        edge_ids = sorted(sub["edge_id"].unique())
        n_items = len(edge_ids)

        # Pivot: boot_id × edge_id -> value
        pivot = sub.pivot_table(index="boot_id", columns="edge_id", values="value")
        pivot = pivot.reindex(columns=edge_ids)
        item_labels = edge_ids
    else:
        # Centrality: node-level
        sub = sub.copy()
        node_ids = sorted(sub["node1"].unique())
        n_items = len(node_ids)
        pivot = sub.pivot_table(index="boot_id", columns="node1", values="value")
        pivot = pivot.reindex(columns=node_ids)
        item_labels = node_ids

    # Compute pairwise difference p-values
    p_matrix = np.ones((n_items, n_items))
    significant = np.zeros((n_items, n_items), dtype=bool)

    for i in range(n_items):
        for j in range(i + 1, n_items):
            diffs = pivot.iloc[:, i].values - pivot.iloc[:, j].values
            diffs = diffs[~np.isnan(diffs)]
            if len(diffs) == 0:
                continue
            # Two-tailed p-value: proportion of bootstrap diffs that
            # include zero in their range
            p_val = np.mean(diffs > 0)
            p_val = 2 * min(p_val, 1 - p_val)  # two-tailed
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val
            significant[i, j] = p_val < alpha
            significant[j, i] = p_val < alpha

    result = pd.DataFrame(
        significant.astype(int),
        index=item_labels,
        columns=item_labels,
    )
    result.attrs["p_values"] = pd.DataFrame(
        p_matrix, index=item_labels, columns=item_labels,
    )
    return result
