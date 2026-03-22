"""Multilevel VAR network visualization — three panels side by side."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._drawing import _plot_network_panels

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from ..mlvar.network import MLVARNetwork


def plot_mlvar_networks(
    mlvar_net: MLVARNetwork,
    *,
    layout: str = "spring",
    shared_layout: bool = True,
    figsize: tuple[float, float] | None = None,
    seed: int = 42,
    **kwargs,
) -> Figure:
    """Plot temporal, contemporaneous, and between-subjects networks.

    Parameters
    ----------
    mlvar_net : MLVARNetwork
        Multilevel VAR network result.
    layout : str
        Layout algorithm (``"spring"``, ``"circular"``, ``"kamada_kawai"``).
    shared_layout : bool
        If True, use the same node positions for all panels.
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
        figsize = (18, 6)

    panels = [
        ("Temporal", mlvar_net.temporal, True),
        ("Contemporaneous", mlvar_net.contemporaneous, False),
        ("Between-Subjects", mlvar_net.between_subjects, False),
    ]

    return _plot_network_panels(
        panels,
        layout=layout,
        layout_network=mlvar_net.contemporaneous,
        figsize=figsize,
        seed=seed,
        suptitle="Multilevel VAR Network (mlVAR)",
        **kwargs,
    )
