"""Time-series network visualizations — temporal and contemporaneous side by side."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._drawing import _plot_network_panels

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
    show_legend: bool = True,
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
    show_legend : bool
        If True, add a numbered variable legend panel on the right.
    **kwargs
        Additional drawing keyword arguments.

    Returns
    -------
    Figure
    """
    panels = [
        ("Temporal", ts_net.temporal, True),
        ("Contemporaneous", ts_net.contemporaneous, False),
    ]

    return _plot_network_panels(
        panels,
        layout=layout,
        layout_network=ts_net.contemporaneous,
        figsize=figsize,
        seed=seed,
        suptitle="Time-Series Network (graphicalVAR)",
        show_legend=show_legend,
        **kwargs,
    )
