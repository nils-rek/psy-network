"""Community-colored network visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from . import _theme as T
from ._drawing import _draw_legend_panel

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from ..network import Network


def plot_community(
    net: Network,
    communities: pd.Series,
    *,
    layout: str = "spring",
    palette: list[str] | None = None,
    node_size: float = T.NODE_SIZE_DEFAULT,
    edge_color_pos: str = T.EDGE_COLOR_POS,
    edge_color_neg: str = T.EDGE_COLOR_NEG,
    max_edge_width: float = T.EDGE_WIDTH_MAX,
    min_edge_width: float = T.EDGE_WIDTH_MIN,
    font_size: int = T.FONT_SIZE_NODE,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = T.FIGSIZE_SINGLE,
    seed: int = 42,
    show_legend: bool = True,
    legend_title: str | None = T.LEGEND_TITLE_DEFAULT,
) -> Figure:
    """Plot a network with nodes colored by community.

    Parameters
    ----------
    net : Network
        The network to plot.
    communities : pd.Series
        Community assignments (index=node labels, values=community IDs).
    layout : str
        Layout algorithm: ``"spring"``, ``"circular"``, ``"kamada_kawai"``.
    palette : list of str or None
        Custom colors for communities. If None, uses colorblind-safe palette.
    node_size : float
        Node size.
    edge_color_pos, edge_color_neg : str
        Colors for positive / negative edges.
    max_edge_width, min_edge_width : float
        Edge width range.
    font_size : int
        Node label font size.
    title : str or None
        Plot title.
    ax : Axes or None
        Matplotlib axes. If None, a new figure is created.
    figsize : tuple
        Figure size (only used if ax is None).
    seed : int
        Random seed for spring layout.
    show_legend : bool
        Whether to show a legend with community colors and variable names.
    legend_title : str or None
        Header text for the legend panel (default ``"Legend"``).
        Pass ``None`` to suppress the header.

    Returns
    -------
    Figure
    """
    G = net.to_networkx()

    # Layout
    layout_funcs = {
        "spring": lambda: nx.spring_layout(G, seed=seed, weight="weight"),
        "fruchterman_reingold": lambda: nx.spring_layout(G, seed=seed, weight="weight"),
        "circular": lambda: nx.circular_layout(G),
        "kamada_kawai": lambda: nx.kamada_kawai_layout(
            G, dist={n: {m: 1.0 / abs(G[n][m]["weight"]) if abs(G[n][m]["weight"]) > 0 else 10.0
                         for m in G[n]} for n in G}
        ),
    }
    pos = layout_funcs.get(layout, layout_funcs["spring"])()

    # Figure
    if ax is None:
        if show_legend:
            fig, axes = plt.subplots(
                1, 2, figsize=figsize,
                gridspec_kw={"width_ratios": [1, 0.3]},
            )
            net_ax = axes[0]
            legend_ax = axes[1]
        else:
            fig, net_ax = plt.subplots(1, 1, figsize=figsize)
            legend_ax = None
    else:
        fig = ax.get_figure()
        net_ax = ax
        legend_ax = None

    # Community colors
    colors = palette if palette is not None else T.COMMUNITY_PALETTE
    node_colors = [colors[communities[node] % len(colors)] for node in net.labels]

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=net_ax, node_size=node_size, node_color=node_colors,
        edgecolors=T.NODE_BORDER_COLOR, linewidths=T.NODE_BORDER_WIDTH,
    )

    # Numbered labels when legend is shown, full labels otherwise
    if show_legend:
        label_map = {label: str(i + 1) for i, label in enumerate(net.labels)}
    else:
        label_map = {label: label for label in net.labels}
    nx.draw_networkx_labels(
        G, pos, labels=label_map, ax=net_ax,
        font_size=font_size, font_weight=T.FONT_WEIGHT_NODE,
    )

    # Draw edges
    edges = G.edges(data=True)
    if edges:
        weights = np.array([abs(d["weight"]) for _, _, d in edges])
        max_w = weights.max() if weights.max() > 0 else 1.0
        widths = min_edge_width + (weights / max_w) * (max_edge_width - min_edge_width)

        for (u, v, d), w in zip(edges, widths):
            color = edge_color_pos if d["weight"] >= 0 else edge_color_neg
            style = T.EDGE_STYLE_POS if d["weight"] >= 0 else T.EDGE_STYLE_NEG
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], ax=net_ax,
                width=float(w), edge_color=color, style=style,
                alpha=T.EDGE_ALPHA,
            )

    net_ax.set_title(title or f"Communities ({net.method})",
                     fontsize=T.TITLE_FONT_SIZE)
    net_ax.axis("off")

    # Legend panel with community-colored markers
    if legend_ax is not None:
        community_node_colors = [
            colors[communities[label] % len(colors)] for label in net.labels
        ]
        _draw_legend_panel(
            legend_ax, net.labels,
            edge_color_pos=edge_color_pos,
            edge_color_neg=edge_color_neg,
            community_colors=community_node_colors,
            communities=communities,
            legend_title=legend_title,
        )
    elif ax is not None and show_legend:
        # Inline community legend when user provides their own axes
        comm_ids = sorted(communities.unique())
        legend_handles = []
        for cid in comm_ids:
            patch = plt.Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor=colors[cid % len(colors)],
                markersize=10, label=f"Community {cid}",
            )
            legend_handles.append(patch)
        net_ax.legend(handles=legend_handles, loc="upper left",
                      fontsize=T.FONT_SIZE_LEGEND)

    fig.tight_layout()
    return fig
