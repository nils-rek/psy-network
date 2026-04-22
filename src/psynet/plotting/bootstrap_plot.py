"""Bootstrap visualization — edge accuracy, stability, difference plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from . import _theme as T

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from ..bootstrap.results import BootstrapResult


def plot_edge_accuracy(
    boot_result: BootstrapResult,
    *,
    order: str = "mean",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Horizontal interval plot: sample edge weights with bootstrap CIs.

    Parameters
    ----------
    boot_result : BootstrapResult
        Nonparametric bootstrap result.
    order : str
        Sort edges by ``"mean"`` (bootstrap mean) or ``"sample"`` (original).
    figsize : tuple | None
        Figure size.

    Returns
    -------
    Figure
    """
    summary = boot_result.summary(statistic="edge")
    if summary.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No edge data", ha="center", va="center")
        return fig

    summary["edge_label"] = summary["node1"] + " -- " + summary["node2"]

    sort_col = "mean" if order == "mean" else "sample"
    if sort_col in summary.columns:
        summary = summary.sort_values(sort_col).reset_index(drop=True)

    n_edges = len(summary)
    if figsize is None:
        figsize = (8, max(4, n_edges * 0.25))

    fig, ax = plt.subplots(figsize=figsize)
    T.apply_theme_to_axes(ax)
    y_pos = np.arange(n_edges)

    # CI bars
    ax.hlines(
        y_pos,
        summary["ci_lower"].values,
        summary["ci_upper"].values,
        color=T.CI_BAR_COLOR, linewidth=2,
    )
    # Bootstrap mean
    ax.scatter(summary["mean"].values, y_pos, color=T.ACCENT_COLORS[0],
               s=15, zorder=3, label="Boot mean")
    # Sample value
    if "sample" in summary.columns:
        ax.scatter(summary["sample"].values, y_pos, color=T.ACCENT_COLORS[1],
                   s=15, zorder=4, marker="D", label="Sample")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary["edge_label"].values, fontsize=6)
    ax.axvline(0, color=T.ZERO_LINE_COLOR, linewidth=0.5, linestyle="--")
    ax.set_xlabel("Edge weight", color=T.TEXT_PRIMARY)
    ax.set_title("Edge Accuracy (Bootstrap CIs)", color=T.TEXT_PRIMARY)
    ax.legend(fontsize=8, labelcolor=T.TEXT_PRIMARY, frameon=False)
    fig.tight_layout()
    return fig


def plot_centrality_stability(
    boot_result: BootstrapResult,
    *,
    statistics: list[str] | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> Figure:
    """Line plot: correlation vs proportion retained, with CI bands.

    Parameters
    ----------
    boot_result : BootstrapResult
        Case-dropping bootstrap result.
    statistics : list[str] | None
        Centrality measures to plot.
    figsize : tuple
        Figure size.

    Returns
    -------
    Figure
    """
    if boot_result.case_drop_correlations is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No case-drop data", ha="center", va="center")
        return fig

    df = boot_result.case_drop_correlations
    if statistics is None:
        statistics = sorted(df["statistic"].unique())

    fig, ax = plt.subplots(figsize=figsize)
    T.apply_theme_to_axes(ax)

    for i, stat in enumerate(statistics):
        sub = df[df["statistic"] == stat]
        grouped = sub.groupby("proportion")["correlation"]
        means = grouped.mean()
        lower = grouped.quantile(0.025)
        upper = grouped.quantile(0.975)

        color = T.ACCENT_COLORS[i % len(T.ACCENT_COLORS)]
        props = means.index.values
        ax.plot(props, means.values, color=color, label=stat, linewidth=1.5)
        ax.fill_between(props, lower.values, upper.values, color=color, alpha=0.15)

    ax.axhline(0.7, color=T.THRESHOLD_LINE_COLOR, linestyle="--", linewidth=0.8,
               label="CS threshold (0.7)")
    ax.set_xlabel("Proportion of cases retained", color=T.TEXT_PRIMARY)
    ax.set_ylabel("Correlation with original", color=T.TEXT_PRIMARY)
    ax.set_title("Centrality Stability", color=T.TEXT_PRIMARY)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=8, labelcolor=T.TEXT_PRIMARY, frameon=False)
    fig.tight_layout()
    return fig


def plot_difference(
    boot_result: BootstrapResult,
    *,
    statistic: str = "edge",
    alpha: float = 0.05,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Heatmap of pairwise difference significance.

    Parameters
    ----------
    boot_result : BootstrapResult
        Nonparametric bootstrap result.
    statistic : str
        Statistic to test differences for.
    alpha : float
        Significance level.
    figsize : tuple | None
        Figure size.

    Returns
    -------
    Figure
    """
    diff_matrix = boot_result.difference_test(statistic=statistic)
    n = len(diff_matrix)

    if figsize is None:
        figsize = (max(6, n * 0.4), max(6, n * 0.4))

    fig, ax = plt.subplots(figsize=figsize)
    T.apply_theme_to_axes(ax)

    # Significant = dark (0.0), not significant = light gray (0.85).
    # In dark theme the cmap is inverted so significant cells stand out bright.
    display = np.where(diff_matrix.values, 0.0, 0.85)
    ax.imshow(display, cmap=T.DIFFERENCE_CMAP, vmin=0, vmax=1, aspect="equal")

    labels = diff_matrix.index.tolist()
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(f"Difference Test ({statistic})\nBlack = significant (p < {alpha})",
                 color=T.TEXT_PRIMARY)
    fig.tight_layout()
    return fig
