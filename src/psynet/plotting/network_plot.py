"""Network visualization — spring-layout graph plot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

from . import _theme as T
from ._drawing import _draw_legend_panel

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from ..network import Network


def plot_network(
    net: Network,
    *,
    layout: str = "spring",
    node_size: float | str = T.NODE_SIZE_DEFAULT,
    node_color: str = T.NODE_FILL_COLOR,
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
    """Plot a psychometric network.

    Parameters
    ----------
    net : Network
        The network to plot.
    layout : str
        Layout algorithm: ``"spring"``, ``"circular"``, ``"kamada_kawai"``.
    node_size : float or str
        Fixed size or a centrality measure name (e.g. ``"strength"``).
    node_color : str
        Node fill color.
    edge_color_pos, edge_color_neg : str
        Colors for positive / negative edges.
    max_edge_width, min_edge_width : float
        Edge width range.
    font_size : int
        Node label font size.
    title : str | None
        Plot title.
    ax : Axes | None
        Matplotlib axes to draw on. If None, a new figure is created.
    figsize : tuple
        Figure size (only used if ax is None).
    seed : int
        Random seed for spring layout.
    show_legend : bool
        If True, show a side panel mapping numbered nodes to variable names.
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
                gridspec_kw={"width_ratios": [1, 0.25]},
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

    # Node sizes
    if isinstance(node_size, str):
        from ..centrality import centrality
        cent = centrality(net)
        if node_size in cent.columns:
            vals = cent[node_size].values
            vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)
            lo, hi = T.NODE_SIZE_RANGE
            sizes = lo + vals * (hi - lo)
        else:
            sizes = T.NODE_SIZE_DEFAULT
    else:
        sizes = node_size

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=net_ax, node_size=sizes, node_color=node_color,
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

    net_ax.set_title(title or f"Network ({net.method})",
                     fontsize=T.TITLE_FONT_SIZE)
    net_ax.axis("off")

    # Legend panel
    if legend_ax is not None:
        _draw_legend_panel(
            legend_ax, net.labels,
            edge_color_pos=edge_color_pos,
            edge_color_neg=edge_color_neg,
            legend_title=legend_title,
        )
    elif ax is not None and show_legend:
        # Inline edge-type legend when user provides their own axes
        legend_handles = [
            Line2D([0], [0], color=edge_color_pos, linewidth=2,
                   linestyle="solid", label="Positive"),
            Line2D([0], [0], color=edge_color_neg, linewidth=2,
                   linestyle="dashed", label="Negative"),
        ]
        net_ax.legend(handles=legend_handles, loc="upper left",
                      fontsize=T.FONT_SIZE_LEGEND, frameon=False)

    fig.tight_layout()
    return fig
