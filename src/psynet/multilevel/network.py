"""MultilevelNetwork dataclass — wraps temporal, contemporaneous, and between-subjects networks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from ..network import Network


@dataclass(frozen=True)
class MultilevelNetwork:
    """Immutable container for a multilevel VAR network result.

    Parameters
    ----------
    temporal : Network
        Population-level directed temporal network (fixed effects).
    contemporaneous : Network
        Undirected partial correlation network of pooled residuals.
    between_subjects : Network
        Undirected GGM on subject-level means.
    subject_temporal : dict[str, Network]
        Per-subject temporal networks (fixed + random effects).
    labels : list[str]
        Variable names.
    subject_ids : list[str]
        Subject identifiers.
    method : str
        Estimation method (``"mlVAR"``).
    pvalues : np.ndarray
        p x p p-value matrix for temporal fixed effects.
    fit_info : dict
        BIC and AIC per DV model.
    """

    temporal: Network
    contemporaneous: Network
    between_subjects: Network
    subject_temporal: dict
    labels: list[str]
    subject_ids: list[str]
    method: str
    pvalues: np.ndarray
    fit_info: dict
    # Derived
    n_nodes: int = field(init=False)
    n_subjects: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_nodes", len(self.labels))
        object.__setattr__(self, "n_subjects", len(self.subject_ids))

    def centrality(self) -> pd.DataFrame:
        """Compute centrality for all three network types.

        Returns
        -------
        pd.DataFrame
            Centrality measures with a ``'network'`` column indicating
            ``'temporal'``, ``'contemporaneous'``, or ``'between_subjects'``.
        """
        frames = []
        for name, net in [
            ("temporal", self.temporal),
            ("contemporaneous", self.contemporaneous),
            ("between_subjects", self.between_subjects),
        ]:
            df = net.centrality()
            df["network"] = name
            df["node"] = df.index
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def subject_network(self, subject_id: str) -> Network:
        """Get the temporal network for a specific subject.

        Parameters
        ----------
        subject_id : str
            Subject identifier.

        Returns
        -------
        Network
            Subject-specific directed temporal network.
        """
        if subject_id not in self.subject_temporal:
            raise KeyError(
                f"Subject {subject_id!r} not found. "
                f"Available: {self.subject_ids}"
            )
        return self.subject_temporal[subject_id]

    def plot(self, **kwargs) -> Figure:
        """Plot all three networks side by side."""
        from ..plotting.multilevel_plot import plot_multilevel_networks
        return plot_multilevel_networks(self, **kwargs)
