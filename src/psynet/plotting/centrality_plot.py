"""Centrality visualization — dot/bar plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from ..network import Network


def plot_centrality(
    net: Network,
    *,
    measures: list[str] | None = None,
    standardized: bool = True,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Horizontal dot plot of centrality measures, one panel per measure.

    Parameters
    ----------
    net : Network
        The network whose centrality to plot.
    measures : list[str] | None
        Which measures to include. Defaults to all four.
    standardized : bool
        If True, z-standardize within each measure.
    figsize : tuple | None
        Figure size. Auto-calculated if None.

    Returns
    -------
    Figure
    """
    cent = net.centrality()
    if measures is None:
        measures = list(cent.columns)
    cent = cent[measures]

    if standardized:
        cent = (cent - cent.mean()) / (cent.std() + 1e-10)

    n_measures = len(measures)
    if figsize is None:
        figsize = (4 * n_measures, max(4, len(cent) * 0.3))

    fig, axes = plt.subplots(1, n_measures, figsize=figsize, sharey=True)
    if n_measures == 1:
        axes = [axes]

    for ax, measure in zip(axes, measures):
        values = cent[measure].sort_values()
        ax.hlines(
            y=range(len(values)), xmin=0, xmax=values.values,
            color="#666666", linewidth=0.8,
        )
        ax.scatter(values.values, range(len(values)), color="#2166AC", s=40, zorder=3)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(values.index, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(measure, fontsize=10)
        ax.set_xlabel("z-score" if standardized else "value", fontsize=8)

    fig.suptitle("Centrality Indices", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig
