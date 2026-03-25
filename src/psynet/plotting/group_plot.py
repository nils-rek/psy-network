"""Group network visualizations — multi-panel plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from . import _theme as T
from ._drawing import _compute_layout, _draw_network_on_ax, _draw_legend_panel

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from ..group.network import GroupNetwork
    from ..group.bootstrap import GroupBootstrapResult


def plot_group_networks(
    group_net: GroupNetwork,
    *,
    layout: str = "spring",
    shared_layout: bool = True,
    figsize: tuple[float, float] | None = None,
    seed: int = 42,
    show_legend: bool = True,
    **kwargs,
) -> Figure:
    """Plot all group networks side by side.

    Parameters
    ----------
    group_net : GroupNetwork
        The jointly estimated group networks.
    layout : str
        Layout algorithm for each subplot.
    shared_layout : bool
        If True, compute layout from first group and reuse for all.
    figsize : tuple, optional
        Figure size.
    seed : int
        Random seed for layout.
    show_legend : bool
        If True, add a numbered variable legend panel on the right.
    **kwargs
        Additional keyword arguments passed to ``_draw_network_on_ax``.

    Returns
    -------
    Figure
    """
    K = group_net.n_groups

    if figsize is None:
        w = T.FIGSIZE_PANEL_WIDTH * K
        if show_legend:
            w += T.FIGSIZE_PANEL_WIDTH * 0.35
        figsize = (w, T.FIGSIZE_PANEL_HEIGHT)

    if show_legend:
        width_ratios = [1] * K + [0.3]
        fig, axes = plt.subplots(
            1, K + 1, figsize=figsize,
            gridspec_kw={"width_ratios": width_ratios},
        )
    else:
        fig, axes = plt.subplots(1, K, figsize=figsize)
        if K == 1:
            axes = [axes]

    # Compute shared layout from first group
    shared_pos = None
    if shared_layout:
        first_net = group_net[group_net.group_labels[0]]
        shared_pos = _compute_layout(first_net, layout, seed)

    plot_axes = axes[:K] if show_legend else axes
    for i, label in enumerate(group_net.group_labels):
        net = group_net[label]
        ax = plot_axes[i]

        if shared_pos is not None:
            _draw_network_on_ax(net, ax, shared_pos,
                                show_legend=show_legend, **kwargs)
        else:
            from .network_plot import plot_network
            plot_network(net, ax=ax, layout=layout, seed=seed,
                         show_legend=False, **kwargs)

        ax.set_title(f"{label} (n={net.n_observations})",
                     fontsize=T.PANEL_TITLE_FONT_SIZE)

    if show_legend:
        first_net = group_net[group_net.group_labels[0]]
        _draw_legend_panel(
            axes[-1], first_net.labels,
            edge_color_pos=kwargs.get("edge_color_pos", T.EDGE_COLOR_POS),
            edge_color_neg=kwargs.get("edge_color_neg", T.EDGE_COLOR_NEG),
        )

    fig.suptitle(
        f"Group Networks (JGL-{group_net.penalty}, "
        f"\u03bb\u2081={group_net.lambda1:.4f}, \u03bb\u2082={group_net.lambda2:.4f})",
        fontsize=T.SUPTITLE_FONT_SIZE,
    )
    fig.tight_layout()
    return fig


def plot_group_edge_accuracy(
    boot_result: GroupBootstrapResult,
    *,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Faceted edge accuracy CI plot, one panel per group.

    Parameters
    ----------
    boot_result : GroupBootstrapResult
        Group bootstrap result.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    Figure
    """
    groups = boot_result.original.group_labels
    K = len(groups)

    # Get summary for all groups
    summary = boot_result.summary(statistic="edge")
    if summary.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No edge data", ha="center", va="center")
        return fig

    # Determine n_edges from first group for sizing
    first_group_summary = summary[summary["group"] == groups[0]]
    n_edges = len(first_group_summary)

    if figsize is None:
        figsize = (6 * K, max(4, n_edges * 0.25))

    fig, axes = plt.subplots(1, K, figsize=figsize, sharey=True)
    if K == 1:
        axes = [axes]

    for g_idx, group in enumerate(groups):
        ax = axes[g_idx]
        gsummary = summary[summary["group"] == group].copy()
        gsummary["edge_label"] = gsummary["node1"] + " -- " + gsummary["node2"]

        if "mean" in gsummary.columns:
            gsummary = gsummary.sort_values("mean").reset_index(drop=True)

        ne = len(gsummary)
        y_pos = np.arange(ne)

        ax.hlines(
            y_pos, gsummary["ci_lower"].values, gsummary["ci_upper"].values,
            color="#CCCCCC", linewidth=2,
        )
        ax.scatter(gsummary["mean"].values, y_pos,
                   color=T.ACCENT_COLORS[0], s=15, zorder=3)
        if "sample" in gsummary.columns:
            ax.scatter(
                gsummary["sample"].values, y_pos, color=T.ACCENT_COLORS[1],
                s=15, zorder=4, marker="D",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(gsummary["edge_label"].values, fontsize=6)
        ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Edge weight")
        ax.set_title(f"{group}")

    fig.suptitle("Group Edge Accuracy (Bootstrap CIs)", fontsize=T.TITLE_FONT_SIZE)
    fig.tight_layout()
    return fig


def plot_group_centrality_comparison(
    group_net: GroupNetwork,
    *,
    statistic: str = "strength",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Grouped dot plot comparing a centrality measure across groups.

    Parameters
    ----------
    group_net : GroupNetwork
        The jointly estimated group networks.
    statistic : str
        Centrality measure to compare.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    Figure
    """
    cent_df = group_net.centrality()
    if statistic not in cent_df.columns:
        raise ValueError(f"Unknown centrality statistic: {statistic!r}")

    nodes = sorted(cent_df["node"].unique())
    groups = group_net.group_labels
    n_nodes = len(nodes)

    if figsize is None:
        figsize = (8, max(4, n_nodes * 0.4))

    fig, ax = plt.subplots(figsize=figsize)
    bar_height = 0.8 / len(groups)

    for g_idx, group in enumerate(groups):
        gdata = cent_df[cent_df["group"] == group].set_index("node")
        y_positions = np.arange(n_nodes) + g_idx * bar_height - 0.4 + bar_height / 2
        values = [gdata.loc[node, statistic] if node in gdata.index else 0 for node in nodes]
        ax.barh(
            y_positions, values, height=bar_height * 0.9,
            color=T.ACCENT_COLORS[g_idx % len(T.ACCENT_COLORS)],
            label=group, alpha=0.8,
        )

    ax.set_yticks(np.arange(n_nodes))
    ax.set_yticklabels(nodes, fontsize=8)
    ax.set_xlabel(statistic.capitalize())
    ax.set_title(f"Centrality Comparison: {statistic}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
