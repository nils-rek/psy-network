"""Group bootstrap — nonparametric bootstrap for JGL networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..bootstrap.engine import _extract_statistics
from ._utils import parse_group_data

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from .network import GroupNetwork


@dataclass
class GroupBootstrapResult:
    """Container for group bootstrap results.

    Parameters
    ----------
    original : GroupNetwork
        The group network estimated from the full data.
    boot_statistics : pd.DataFrame
        Long-form DataFrame with columns: boot_id, group, statistic,
        node1, node2, value.
    n_boots : int
        Number of bootstrap samples.
    """

    original: GroupNetwork
    boot_statistics: pd.DataFrame
    n_boots: int

    def summary(
        self,
        statistic: str = "edge",
        group: str | None = None,
    ) -> pd.DataFrame:
        """Summarize bootstrap distributions with CIs.

        Parameters
        ----------
        statistic : str
            Statistic to summarize (``"edge"``, ``"strength"``, etc.).
        group : str, optional
            Filter to a single group. If None, includes all groups.

        Returns
        -------
        pd.DataFrame
        """
        df = self.boot_statistics
        sub = df[df["statistic"] == statistic]
        if group is not None:
            sub = sub[sub["group"] == group]

        if statistic == "edge":
            group_cols = ["group", "node1", "node2"]
        else:
            group_cols = ["group", "node1"]

        grouped = sub.groupby(group_cols)["value"]
        summary = grouped.agg(
            mean="mean",
            sd="std",
            ci_lower=lambda x: np.quantile(x, 0.025),
            ci_upper=lambda x: np.quantile(x, 0.975),
        ).reset_index()

        # Add sample (original) values
        orig_df = sub[sub["boot_id"] == "original"]
        if not orig_df.empty:
            orig_sub = orig_df[group_cols + ["value"]].rename(
                columns={"value": "sample"},
            )
            summary = summary.merge(orig_sub, on=group_cols, how="left")

        return summary

    def plot_edge_accuracy(self, **kwargs) -> Figure:
        """Plot edge accuracy with bootstrap CIs, faceted by group."""
        from ..plotting.group_plot import plot_group_edge_accuracy
        return plot_group_edge_accuracy(self, **kwargs)


def _single_group_boot(
    group_dfs: dict[str, pd.DataFrame],
    group_labels: list[str],
    statistics: list[str],
    boot_id: int,
    rng_seed: int,
    est_kwargs: dict,
) -> list[dict]:
    """Run one group bootstrap iteration: resample within each group, re-estimate JGL."""
    from . import estimate_group_network

    rng = np.random.default_rng(rng_seed)

    # Resample within each group, preserving original labels
    resampled_dfs = {}
    for label in group_labels:
        df = group_dfs[label]
        idx = rng.choice(len(df), size=len(df), replace=True)
        resampled_dfs[label] = df.iloc[idx].reset_index(drop=True)

    # Pass as list, but we know labels are sorted and will become Group1, Group2...
    # So we map back using position
    resampled_list = [resampled_dfs[label] for label in group_labels]

    try:
        gn = estimate_group_network(resampled_list, **est_kwargs)
        records = []
        # gn.group_labels will be ["Group1", "Group2", ...] for list input
        # Map back to original labels by position
        for orig_label, gn_label in zip(group_labels, gn.group_labels):
            net = gn[gn_label]
            stats = _extract_statistics(net, statistics)
            for r in stats:
                r["boot_id"] = boot_id
                r["group"] = orig_label
            records.extend(stats)
        return records
    except Exception:
        return []


def bootnet_group(
    data: pd.DataFrame | list[pd.DataFrame],
    *,
    group_col: str | None = None,
    n_boots: int = 1000,
    statistics: list[str] | None = None,
    n_cores: int = 1,
    seed: int | None = None,
    verbose: bool = True,
    **est_kwargs,
) -> GroupBootstrapResult:
    """Run nonparametric bootstrap for group network estimation.

    Resamples within each group independently, then re-estimates the
    joint model. Uses ``_extract_statistics()`` from the single-group
    bootstrap engine.

    Parameters
    ----------
    data : pd.DataFrame or list[pd.DataFrame]
        Input data (same format as ``estimate_group_network``).
    group_col : str, optional
        Group column name.
    n_boots : int
        Number of bootstrap samples.
    statistics : list[str], optional
        Statistics to extract. Defaults to edge + centrality measures.
    n_cores : int
        Number of parallel workers.
    seed : int, optional
        Random seed.
    verbose : bool
        Print progress.
    **est_kwargs
        Keyword arguments passed to ``estimate_group_network``.

    Returns
    -------
    GroupBootstrapResult
    """
    from . import estimate_group_network

    if statistics is None:
        statistics = ["edge", "strength", "closeness", "betweenness", "expectedInfluence"]

    group_dfs = parse_group_data(data, group_col)
    group_labels = list(group_dfs.keys())

    # Estimate original
    original = estimate_group_network(data, group_col=group_col, **est_kwargs)

    # Extract original statistics
    orig_records = []
    for label in group_labels:
        net = original[label]
        stats = _extract_statistics(net, statistics)
        for r in stats:
            r["boot_id"] = "original"
            r["group"] = label
        orig_records.extend(stats)

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_boots)

    if verbose:
        print(f"Running {n_boots} group bootstraps...")

    results = Parallel(n_jobs=n_cores, verbose=int(verbose))(
        delayed(_single_group_boot)(
            group_dfs, group_labels, statistics, i, int(seeds[i]), est_kwargs,
        )
        for i in range(n_boots)
    )

    all_records = orig_records.copy()
    for res in results:
        all_records.extend(res)

    boot_statistics = pd.DataFrame(all_records)

    return GroupBootstrapResult(
        original=original,
        boot_statistics=boot_statistics,
        n_boots=n_boots,
    )
