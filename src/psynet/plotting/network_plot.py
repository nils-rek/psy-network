"""Network visualization — spring-layout graph plot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from . import _theme as T
from ._drawing import _draw_legend_panel

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from ..network import Network


def _draw_centrality_aura(ax, pos, labels, centrality_values, node_sizes,
                          node_color):
    """Draw a tapering arc around each node encoding centrality magnitude.

    The arc starts at 12 o'clock, wraps clockwise, tapers to a sharp point,
    and fades from opaque to transparent along its length.
    """
    # Normalize centrality to [0, 1]
    vals = np.asarray(centrality_values, dtype=float)
    vmin, vmax = vals.min(), vals.max()
    norm = (vals - vmin) / (vmax - vmin + 1e-10)

    # Compute node radius in data coordinates from layout extent
    xs = np.array([pos[label][0] for label in labels])
    ys = np.array([pos[label][1] for label in labels])
    span = max(xs.max() - xs.min(), ys.max() - ys.min(), 0.1)

    if isinstance(node_sizes, (int, float)):
        size_arr = np.full(len(labels), float(node_sizes))
    else:
        size_arr = np.asarray(node_sizes, dtype=float)

    base_rgba = to_rgba(node_color)
    n_seg = T.AURA_N_SEGMENTS

    for idx, label in enumerate(labels):
        cx, cy = pos[label]
        r_node = span * 0.04 * np.sqrt(size_arr[idx] / T.NODE_SIZE_DEFAULT)
        arc_angle_deg = norm[idx] * T.AURA_ARC_MAX_ANGLE
        if arc_angle_deg < 1.0:
            continue  # negligible arc, skip

        arc_angle = np.radians(arc_angle_deg)
        max_hw = r_node * T.AURA_WIDTH_FACTOR  # half-width at start
        r_center = r_node * T.AURA_GAP_FACTOR  # center line radius

        # Build segments: start at 90° (top), go clockwise (decreasing angle)
        for k in range(n_seg):
            t0 = k / n_seg
            t1 = (k + 1) / n_seg
            # Taper: half-width decreases to 0
            hw0 = max_hw * (1 - t0)
            hw1 = max_hw * (1 - t1)
            # Angles (clockwise from top = decreasing from pi/2)
            a0 = np.pi / 2 - t0 * arc_angle
            a1 = np.pi / 2 - t1 * arc_angle
            # Four corners of the wedge strip
            verts = [
                (cx + (r_center + hw0) * np.cos(a0),
                 cy + (r_center + hw0) * np.sin(a0)),
                (cx + (r_center + hw1) * np.cos(a1),
                 cy + (r_center + hw1) * np.sin(a1)),
                (cx + (r_center - hw1) * np.cos(a1),
                 cy + (r_center - hw1) * np.sin(a1)),
                (cx + (r_center - hw0) * np.cos(a0),
                 cy + (r_center - hw0) * np.sin(a0)),
            ]
            alpha = T.AURA_ALPHA_START * (1 - t0)
            color = (*base_rgba[:3], alpha)
            patch = Polygon(verts, closed=True, facecolor=color,
                            edgecolor="none", zorder=1)
            ax.add_patch(patch)


def plot_network(
    net: Network,
    *,
    layout: str = "spring",
    node_size: float | str = T.NODE_SIZE_DEFAULT,
    node_color: str = T.NODE_FILL_COLOR,
    centrality_aura: str | None = "strength",
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
    centrality_aura : str or None
        Centrality measure name (e.g. ``"strength"``) to draw a tapering
        arc around each node encoding centrality. ``None`` disables the aura.
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

    # Centrality aura
    if centrality_aura is not None:
        from ..centrality import centrality as compute_centrality
        cent_df = compute_centrality(net)
        if centrality_aura in cent_df.columns:
            _draw_centrality_aura(
                net_ax, pos, net.labels, cent_df[centrality_aura].values,
                sizes, node_color,
            )

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
        alphas = T.EDGE_ALPHA_MIN + (weights / max_w) * (T.EDGE_ALPHA_MAX - T.EDGE_ALPHA_MIN)

        for (u, v, d), w, a in zip(edges, widths, alphas):
            color = edge_color_pos if d["weight"] >= 0 else edge_color_neg
            style = T.EDGE_STYLE_POS if d["weight"] >= 0 else T.EDGE_STYLE_NEG
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], ax=net_ax,
                width=float(w), edge_color=color, style=style,
                alpha=float(a),
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
