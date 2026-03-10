"""BootstrapResult dataclass — stores bootstrap output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .._types import BootstrapType

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from ..network import Network


@dataclass
class BootstrapResult:
    """Container for bootstrap analysis results.

    Parameters
    ----------
    original_network : Network
        The network estimated from the full sample.
    boot_statistics : pd.DataFrame
        Long-form DataFrame with columns: boot_id, statistic, node1, node2, value.
        For centrality statistics node2 is NaN.
    boot_type : BootstrapType
        Type of bootstrap performed.
    n_boots : int
        Number of bootstrap samples.
    case_drop_correlations : pd.DataFrame | None
        For case-dropping bootstrap: DataFrame with columns
        proportion, statistic, boot_id, correlation.
    """

    original_network: Network
    boot_statistics: pd.DataFrame
    boot_type: BootstrapType
    n_boots: int
    case_drop_correlations: pd.DataFrame | None = None

    def summary(self, statistic: str = "edge") -> pd.DataFrame:
        """Summarize bootstrap distribution with mean, SD, and quantile CIs.

        Parameters
        ----------
        statistic : str
            Which statistic to summarize (e.g. ``"edge"``, ``"strength"``).

        Returns
        -------
        pd.DataFrame
            Summary with columns: node1, node2, sample, mean, sd, ci_lower, ci_upper.
        """
        df = self.boot_statistics
        sub = df[df["statistic"] == statistic]

        if statistic == "edge":
            group_cols = ["node1", "node2"]
        else:
            group_cols = ["node1"]

        grouped = sub.groupby(group_cols)["value"]
        summary = grouped.agg(
            mean="mean", sd="std",
            ci_lower=lambda x: np.quantile(x, 0.025),
            ci_upper=lambda x: np.quantile(x, 0.975),
        ).reset_index()

        # Add sample (original) values
        orig_df = df[(df["statistic"] == statistic) & (df["boot_id"] == "original")]
        if not orig_df.empty:
            merge_cols = group_cols
            orig_sub = orig_df[merge_cols + ["value"]].rename(columns={"value": "sample"})
            summary = summary.merge(orig_sub, on=merge_cols, how="left")

        return summary

    def cs_coefficient(self, statistic: str = "strength") -> float:
        """Compute the CS-coefficient for a centrality measure.

        Delegates to :func:`psynet.bootstrap.stability.cs_coefficient`.
        """
        from .stability import cs_coefficient
        return cs_coefficient(self, statistic)

    def difference_test(self, statistic: str = "edge") -> pd.DataFrame:
        """Pairwise bootstrap difference test.

        Delegates to :func:`psynet.bootstrap.stability.difference_test`.
        """
        from .stability import difference_test
        return difference_test(self, statistic)

    # ------------------------------------------------------------------
    # Plot shortcuts
    # ------------------------------------------------------------------

    def plot_edge_accuracy(self, **kwargs) -> Figure:
        from ..plotting.bootstrap_plot import plot_edge_accuracy
        return plot_edge_accuracy(self, **kwargs)

    def plot_centrality_stability(self, **kwargs) -> Figure:
        from ..plotting.bootstrap_plot import plot_centrality_stability
        return plot_centrality_stability(self, **kwargs)

    def plot_difference(self, **kwargs) -> Figure:
        from ..plotting.bootstrap_plot import plot_difference
        return plot_difference(self, **kwargs)
