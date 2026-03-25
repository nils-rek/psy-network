"""Multilevel VAR network visualization — three panels side by side."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._drawing import _plot_network_panels

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from ..multilevel.network import MultilevelNetwork


def plot_multilevel_networks(
    ml_net: MultilevelNetwork,
    *,
    layout: str = "spring",
    shared_layout: bool = True,
    figsize: tuple[float, float] | None = None,
    seed: int = 42,
    show_legend: bool = True,
    **kwargs,
) -> Figure:
    """Plot temporal, contemporaneous, and between-subjects networks.

    Parameters
    ----------
    ml_net : MultilevelNetwork
        Multilevel VAR network result.
    layout : str
        Layout algorithm (``"spring"``, ``"circular"``, ``"kamada_kawai"``).
    shared_layout : bool
        If True, use the same node positions for all panels.
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
        ("Temporal", ml_net.temporal, True),
        ("Contemporaneous", ml_net.contemporaneous, False),
        ("Between-Subjects", ml_net.between_subjects, False),
    ]

    return _plot_network_panels(
        panels,
        layout=layout,
        layout_network=ml_net.contemporaneous,
        figsize=figsize,
        seed=seed,
        suptitle="Multilevel VAR Network (mlVAR)",
        show_legend=show_legend,
        **kwargs,
    )
