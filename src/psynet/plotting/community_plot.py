"""Community-colored network visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from ..network import Network


def plot_community(
    net: Network,
    communities: pd.Series,
    *,
    layout: str = "spring",
    palette: list[str] | None = None,
    node_size: float = 300,
    edge_color_pos: str = "#2166AC",
    edge_color_neg: str = "#B2182B",
    max_edge_width: float = 3.0,
    min_edge_width: float = 0.3,
    font_size: int = 8,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 8),
    seed: int = 42,
    show_legend: bool = True,
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
        Custom colors for communities. If None, uses tab10/tab20.
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
        Whether to show a community legend.

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
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # Community colors
    n_communities = communities.nunique()
    if palette is not None:
        colors = palette
    elif n_communities <= 10:
        cmap = plt.cm.tab10
        colors = [cmap(i) for i in range(n_communities)]
    else:
        cmap = plt.cm.tab20
        colors = [cmap(i) for i in range(n_communities)]

    node_colors = [colors[communities[node] % len(colors)] for node in net.labels]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=node_colors,
                           edgecolors="#333333", linewidths=1.0)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size, font_weight="bold")

    # Draw edges
    edges = G.edges(data=True)
    if edges:
        weights = np.array([abs(d["weight"]) for _, _, d in edges])
        max_w = weights.max() if weights.max() > 0 else 1.0
        widths = min_edge_width + (weights / max_w) * (max_edge_width - min_edge_width)

        for (u, v, d), w in zip(edges, widths):
            color = edge_color_pos if d["weight"] >= 0 else edge_color_neg
            style = "solid" if d["weight"] >= 0 else "dashed"
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], ax=ax,
                width=float(w), edge_color=color, style=style, alpha=0.7,
            )

    # Legend
    if show_legend:
        comm_ids = sorted(communities.unique())
        legend_handles = []
        for cid in comm_ids:
            patch = plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=colors[cid % len(colors)],
                               markersize=10, label=f"Community {cid}")
            legend_handles.append(patch)
        ax.legend(handles=legend_handles, loc="upper left", fontsize=font_size)

    ax.set_title(title or f"Communities ({net.method})", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    return fig
