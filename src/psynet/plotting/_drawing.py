"""Shared network drawing helper for matplotlib axes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

from . import _theme as T

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _compute_layout(network, layout: str = "spring", seed: int = 42) -> dict:
    """Compute node positions from a Network using a networkx layout algorithm.

    Parameters
    ----------
    network : Network
        Network to compute layout for (typically undirected for best results).
    layout : str
        Layout algorithm (``"spring"``, ``"circular"``, ``"kamada_kawai"``).
    seed : int
        Random seed for stochastic layouts.

    Returns
    -------
    dict
        Mapping of node labels to (x, y) positions.
    """
    G = network.to_networkx()
    layout_funcs = {
        "spring": lambda: nx.spring_layout(G, seed=seed, weight="weight"),
        "fruchterman_reingold": lambda: nx.spring_layout(G, seed=seed, weight="weight"),
        "circular": lambda: nx.circular_layout(G),
        "kamada_kawai": lambda: nx.kamada_kawai_layout(G),
    }
    return layout_funcs.get(layout, layout_funcs["spring"])()


def _draw_one_entry(ax, y, index, label, color=None):
    """Draw a single numbered legend entry at vertical position *y*."""
    num_text = f"{index + 1}:"
    if color is not None:
        ax.plot(
            0.05, y, "o",
            color=color,
            markersize=7,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.text(
            0.17, y, num_text,
            fontsize=T.FONT_SIZE_LEGEND, fontweight="bold",
            va="center", ha="right", transform=ax.transAxes,
        )
        ax.text(
            0.20, y, label,
            fontsize=T.FONT_SIZE_LEGEND,
            va="center", ha="left", transform=ax.transAxes,
        )
    else:
        ax.text(
            0.10, y, num_text,
            fontsize=T.FONT_SIZE_LEGEND, fontweight="bold",
            va="center", ha="right", transform=ax.transAxes,
        )
        ax.text(
            0.15, y, label,
            fontsize=T.FONT_SIZE_LEGEND,
            va="center", ha="left", transform=ax.transAxes,
        )


def _draw_legend_panel(
    ax,
    labels: list[str],
    *,
    edge_color_pos: str = T.EDGE_COLOR_POS,
    edge_color_neg: str = T.EDGE_COLOR_NEG,
    community_colors: list | None = None,
    communities=None,
    legend_title: str | None = T.LEGEND_TITLE_DEFAULT,
):
    """Draw a numbered variable-name legend on a dedicated axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw the legend on.
    labels : list[str]
        Variable names in order.
    edge_color_pos, edge_color_neg : str
        Colors for the edge-type legend entries.
    community_colors : list or None
        If provided, draw a colored marker dot for each variable.
    communities : pd.Series or None
        Community assignments (index=node labels, values=community IDs).
        When provided alongside *community_colors*, entries are grouped
        under community subheadings.
    legend_title : str or None
        Header text drawn at the top of the legend panel.
        Pass ``None`` to suppress.
    """
    ax.axis("off")

    # --- compute how many visual lines we need ---
    n_lines = float(len(labels))
    has_header = legend_title is not None
    has_groups = communities is not None and community_colors is not None
    if has_header:
        n_lines += 1.5  # header + extra gap
    if has_groups:
        n_comms = len(communities.unique())
        n_lines += n_comms * 1.5  # subheading + gap per community

    available = 0.95 - 0.15  # top − bottom reserve for edge legend
    step = min(T.LEGEND_STEP, available / max(n_lines, 1))

    y = 0.95

    # --- header ---
    if has_header:
        ax.text(
            0.05, y, legend_title,
            fontsize=T.LEGEND_HEADER_FONT_SIZE, fontweight="bold",
            va="center", ha="left", transform=ax.transAxes,
        )
        y -= step * 1.5

    # --- entries ---
    if has_groups:
        comm_ids = sorted(communities.unique())
        for comm_id in comm_ids:
            # subheading
            y -= step * 0.5
            ax.text(
                0.05, y, f"Community {comm_id}",
                fontsize=T.LEGEND_SUBHEADING_FONT_SIZE, fontweight="bold",
                fontstyle="italic", va="center", ha="left",
                transform=ax.transAxes,
            )
            y -= step
            # entries in this community, preserving original label order
            for i, label in enumerate(labels):
                if communities.get(label) != comm_id:
                    continue
                _draw_one_entry(ax, y, i, label,
                                color=community_colors[i])
                y -= step
    else:
        for i, label in enumerate(labels):
            color = community_colors[i] if community_colors is not None else None
            _draw_one_entry(ax, y, i, label, color=color)
            y -= step

    # --- edge-type legend at bottom ---
    legend_handles = [
        Line2D([0], [0], color=edge_color_pos, linewidth=2,
               linestyle="solid", label="Positive"),
        Line2D([0], [0], color=edge_color_neg, linewidth=2,
               linestyle="dashed", label="Negative"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower left",
        fontsize=T.FONT_SIZE_LEGEND, frameon=False,
        bbox_to_anchor=(0.0, 0.0),
    )


def _plot_network_panels(
    panels: list[tuple[str, object, bool]],
    *,
    layout: str = "spring",
    layout_network=None,
    figsize: tuple[float, float] | None = None,
    seed: int = 42,
    suptitle: str = "",
    show_legend: bool = True,
    **kwargs,
) -> Figure:
    """Plot multiple network panels side by side with a shared layout.

    Parameters
    ----------
    panels : list of (title, Network, directed)
        Each tuple is (panel title, Network object, whether to draw directed).
    layout : str
        Layout algorithm.
    layout_network : Network, optional
        Network to compute layout from. If None, uses the first undirected
        network in panels, or the first network.
    figsize : tuple, optional
        Figure size. Defaults to (7*n_panels, 6).
    seed : int
        Random seed for layout.
    suptitle : str
        Figure super-title.
    show_legend : bool
        If True, add a numbered variable legend panel on the right.
    **kwargs
        Additional drawing keyword arguments passed to ``_draw_network_on_ax``.

    Returns
    -------
    Figure
    """
    n = len(panels)

    if figsize is None:
        w = T.FIGSIZE_PANEL_WIDTH * n
        if show_legend:
            w += T.FIGSIZE_PANEL_WIDTH * 0.35
        figsize = (w, T.FIGSIZE_PANEL_HEIGHT)

    if show_legend:
        width_ratios = [1] * n + [0.3]
        fig, axes = plt.subplots(
            1, n + 1, figsize=figsize,
            gridspec_kw={"width_ratios": width_ratios},
        )
    else:
        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]

    # Determine layout source network
    if layout_network is None:
        for _, net, directed in panels:
            if not directed:
                layout_network = net
                break
        if layout_network is None:
            layout_network = panels[0][1]

    pos = _compute_layout(layout_network, layout, seed)

    plot_axes = axes[:n] if show_legend else axes
    for ax, (title, net, directed) in zip(plot_axes, panels):
        _draw_network_on_ax(net, ax, pos, directed=directed,
                            show_legend=show_legend, **kwargs)
        ax.set_title(f"{title} (n={net.n_observations})",
                     fontsize=T.PANEL_TITLE_FONT_SIZE)

    if show_legend:
        labels = panels[0][1].labels
        _draw_legend_panel(
            axes[-1], labels,
            edge_color_pos=kwargs.get("edge_color_pos", T.EDGE_COLOR_POS),
            edge_color_neg=kwargs.get("edge_color_neg", T.EDGE_COLOR_NEG),
            legend_title=kwargs.get("legend_title", T.LEGEND_TITLE_DEFAULT),
        )

    if suptitle:
        fig.suptitle(suptitle, fontsize=T.SUPTITLE_FONT_SIZE)
    fig.tight_layout()
    return fig


def _draw_network_on_ax(net, ax, pos, *, directed=False, show_legend=True,
                         **kwargs):
    """Draw a network on given axes with pre-computed positions."""
    node_color = kwargs.get("node_color", T.NODE_FILL_COLOR)
    edge_color_pos = kwargs.get("edge_color_pos", T.EDGE_COLOR_POS)
    edge_color_neg = kwargs.get("edge_color_neg", T.EDGE_COLOR_NEG)
    max_edge_width = kwargs.get("max_edge_width", T.EDGE_WIDTH_MAX)
    min_edge_width = kwargs.get("min_edge_width", T.EDGE_WIDTH_MIN)
    font_size = kwargs.get("font_size", T.FONT_SIZE_NODE)
    node_size = kwargs.get("node_size", T.NODE_SIZE_DEFAULT)

    G = net.to_networkx()

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_size, node_color=node_color,
        edgecolors=T.NODE_BORDER_COLOR, linewidths=T.NODE_BORDER_WIDTH,
    )

    # Numbered labels when legend is shown, full labels otherwise
    if show_legend:
        label_map = {label: str(i + 1) for i, label in enumerate(net.labels)}
    else:
        label_map = {label: label for label in net.labels}
    nx.draw_networkx_labels(
        G, pos, labels=label_map, ax=ax,
        font_size=font_size, font_weight=T.FONT_WEIGHT_NODE,
    )

    edges = list(G.edges(data=True))
    if edges:
        weights = np.array([abs(d["weight"]) for _, _, d in edges])
        max_w = weights.max() if weights.max() > 0 else 1.0
        widths = min_edge_width + (weights / max_w) * (max_edge_width - min_edge_width)

        for (u, v, d), w in zip(edges, widths):
            color = edge_color_pos if d["weight"] >= 0 else edge_color_neg
            style = T.EDGE_STYLE_POS if d["weight"] >= 0 else T.EDGE_STYLE_NEG
            edge_kwargs = dict(
                edgelist=[(u, v)], ax=ax,
                width=float(w), edge_color=color, style=style,
                alpha=T.EDGE_ALPHA,
            )
            if directed:
                edge_kwargs.update(
                    arrows=True,
                    arrowstyle="-|>",
                    connectionstyle="arc3,rad=0.1",
                )
            nx.draw_networkx_edges(G, pos, **edge_kwargs)

    ax.axis("off")
