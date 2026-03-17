"""Time-series network visualizations — temporal and contemporaneous side by side."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ._drawing import _draw_network_on_ax

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from ..timeseries.network import TSNetwork


def plot_ts_networks(
    ts_net: TSNetwork,
    *,
    layout: str = "spring",
    shared_layout: bool = True,
    figsize: tuple[float, float] | None = None,
    seed: int = 42,
    **kwargs,
) -> Figure:
    """Plot temporal and contemporaneous networks side by side.

    Parameters
    ----------
    ts_net : TSNetwork
        Time-series network result.
    layout : str
        Layout algorithm (``"spring"``, ``"circular"``, ``"kamada_kawai"``).
    shared_layout : bool
        If True, use the same node positions for both panels.
    figsize : tuple, optional
        Figure size.
    seed : int
        Random seed for layout.
    **kwargs
        Additional drawing keyword arguments.

    Returns
    -------
    Figure
    """
    if figsize is None:
        figsize = (12, 6)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Compute layout from contemporaneous network (undirected, better for layout)
    G_layout = ts_net.contemporaneous.to_networkx()
    layout_funcs = {
        "spring": lambda: nx.spring_layout(G_layout, seed=seed, weight="weight"),
        "circular": lambda: nx.circular_layout(G_layout),
        "kamada_kawai": lambda: nx.kamada_kawai_layout(G_layout),
    }
    pos = layout_funcs.get(layout, layout_funcs["spring"])()

    # Draw temporal (directed)
    _draw_network_on_ax(ts_net.temporal, axes[0], pos, directed=True, **kwargs)
    axes[0].set_title(f"Temporal (n={ts_net.n_observations})", fontsize=11)

    # Draw contemporaneous (undirected)
    _draw_network_on_ax(ts_net.contemporaneous, axes[1], pos, directed=False, **kwargs)
    axes[1].set_title(f"Contemporaneous (n={ts_net.n_observations})", fontsize=11)

    fig.suptitle("Time-Series Network (graphicalVAR)", fontsize=13)
    fig.tight_layout()
    return fig


