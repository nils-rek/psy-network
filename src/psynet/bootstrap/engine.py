"""Bootstrap engine — ``bootnet()`` function."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from .._types import BootstrapType, Statistic
from ..centrality import strength, expected_influence, closeness, betweenness
from ..estimation import estimate_network
from ..network import Network
from .results import BootstrapResult

logger = logging.getLogger(__name__)


def _extract_statistics(
    net,
    statistics: list[str],
) -> list[dict]:
    """Extract requested statistics from a Network as long-form records."""
    records: list[dict] = []
    labels = net.labels
    p = net.n_nodes

    for stat in statistics:
        if stat == Statistic.EDGE.value or stat == "edge":
            for i in range(p):
                for j in range(i + 1, p):
                    records.append({
                        "statistic": "edge",
                        "node1": labels[i],
                        "node2": labels[j],
                        "value": net.adjacency[i, j],
                    })
        else:
            # Centrality measure
            func_map = {
                "strength": strength,
                "closeness": closeness,
                "betweenness": betweenness,
                "expectedInfluence": expected_influence,
            }
            if stat in func_map:
                vals = func_map[stat](net)
                for node, val in vals.items():
                    records.append({
                        "statistic": stat,
                        "node1": node,
                        "node2": np.nan,
                        "value": val,
                    })
    return records


def _single_nonparametric_boot(
    data: pd.DataFrame,
    method: str,
    statistics: list[str],
    boot_id: int,
    rng_seed: int,
    est_kwargs: dict,
) -> list[dict]:
    """Run one nonparametric bootstrap iteration."""
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(data), size=len(data), replace=True)
    boot_data = data.iloc[idx].reset_index(drop=True)
    try:
        net = estimate_network(boot_data, method=method, **est_kwargs)
        records = _extract_statistics(net, statistics)
        for r in records:
            r["boot_id"] = boot_id
        return records
    except Exception:
        return []


def _single_case_drop(
    data: pd.DataFrame,
    method: str,
    original_centralities: dict[str, pd.Series],
    proportion: float,
    boot_id: int,
    rng_seed: int,
    est_kwargs: dict,
) -> list[dict]:
    """Run one case-dropping iteration at a given proportion."""
    rng = np.random.default_rng(rng_seed)
    n_keep = max(3, int(len(data) * proportion))
    idx = rng.choice(len(data), size=n_keep, replace=False)
    sub_data = data.iloc[idx].reset_index(drop=True)

    records: list[dict] = []
    try:
        net = estimate_network(sub_data, method=method, **est_kwargs)
        for stat_name, orig_vals in original_centralities.items():
            func_map = {
                "strength": strength,
                "closeness": closeness,
                "betweenness": betweenness,
                "expectedInfluence": expected_influence,
            }
            boot_vals = func_map[stat_name](net)
            # Align indices
            common = orig_vals.index.intersection(boot_vals.index)
            if len(common) >= 3:
                corr, _ = spearmanr(orig_vals[common], boot_vals[common])
            else:
                corr = np.nan
            records.append({
                "proportion": proportion,
                "statistic": stat_name,
                "boot_id": boot_id,
                "correlation": corr,
            })
    except Exception:
        for stat_name in original_centralities:
            records.append({
                "proportion": proportion,
                "statistic": stat_name,
                "boot_id": boot_id,
                "correlation": np.nan,
            })
    return records


def bootnet(
    data: pd.DataFrame,
    *,
    n_boots: int = 1000,
    boot_type: str | BootstrapType = "nonparametric",
    method: str | None = None,
    network: Network | None = None,
    statistics: list[str] | None = None,
    n_cores: int = -1,
    case_min: float = 0.25,
    case_max: float = 0.75,
    case_n: int = 10,
    seed: int | None = None,
    verbose: bool = True,
    **est_kwargs,
) -> BootstrapResult:
    """Run bootstrap analysis on network estimation.

    Parameters
    ----------
    data : pd.DataFrame
        Observations × variables data.
    n_boots : int
        Number of bootstrap samples.
    boot_type : str or BootstrapType
        ``"nonparametric"`` for edge/centrality accuracy or ``"case"`` for
        centrality stability (case-dropping).
    method : str | None
        Estimation method name (passed to ``estimate_network``).
        Defaults to ``"EBICglasso"`` unless ``network`` is provided,
        in which case the method is inherited from the network.
    network : Network | None
        A previously estimated network. When provided, the bootstrap
        inherits the estimation method and keyword arguments from the
        network's ``estimation_info``.  Explicit ``method`` / ``est_kwargs``
        override inherited values (with a warning on conflict).
    statistics : list[str] | None
        Statistics to extract. Defaults to ``["edge", "strength",
        "closeness", "betweenness", "expectedInfluence"]`` for
        nonparametric and centrality measures for case-dropping.
    n_cores : int
        Number of parallel workers (joblib).  ``-1`` (default) uses all
        available cores.
    case_min, case_max : float
        Proportion range for case-dropping bootstrap.
    case_n : int
        Number of proportion steps for case-dropping.
    seed : int | None
        Random seed for reproducibility.
    verbose : bool
        Print progress information.
    **est_kwargs
        Additional keyword arguments passed to the estimator.

    Returns
    -------
    BootstrapResult
    """
    boot_type = BootstrapType(boot_type)
    rng = np.random.default_rng(seed)

    # --- Resolve method and est_kwargs from network if provided (R4) ---
    if network is not None and network.estimation_info is not None:
        net_info = network.estimation_info
        # Inherit method
        if method is None:
            method = net_info.method
        # Inherit est_kwargs, with user overrides
        inherited_kwargs = dict(net_info.est_kwargs)
        if est_kwargs:
            for key, val in est_kwargs.items():
                if key in inherited_kwargs and inherited_kwargs[key] != val:
                    warnings.warn(
                        f"bootnet kwarg {key!r}={val!r} overrides network's "
                        f"estimation_info value {inherited_kwargs[key]!r}",
                        stacklevel=2,
                    )
            inherited_kwargs.update(est_kwargs)
        est_kwargs = inherited_kwargs

    # Fall back to default method
    if method is None:
        method = "EBICglasso"

    logger.info("bootnet: method=%s, est_kwargs=%s", method, est_kwargs)

    # Estimate original network
    original = estimate_network(data, method=method, **est_kwargs)

    if boot_type == BootstrapType.NONPARAMETRIC:
        if statistics is None:
            statistics = ["edge", "strength", "closeness", "betweenness", "expectedInfluence"]

        # Extract original statistics
        orig_records = _extract_statistics(original, statistics)
        for r in orig_records:
            r["boot_id"] = "original"

        # Generate seeds for reproducibility
        seeds = rng.integers(0, 2**31, size=n_boots)

        if verbose:
            print(f"Running {n_boots} nonparametric bootstraps...")

        results = Parallel(n_jobs=n_cores, verbose=int(verbose))(
            delayed(_single_nonparametric_boot)(
                data, method, statistics, i, int(seeds[i]), est_kwargs,
            )
            for i in range(n_boots)
        )

        all_records = orig_records.copy()
        for res in results:
            all_records.extend(res)

        boot_statistics = pd.DataFrame(all_records)

        return BootstrapResult(
            original_network=original,
            boot_statistics=boot_statistics,
            boot_type=boot_type,
            n_boots=n_boots,
        )

    else:  # case-dropping
        centrality_stats = ["strength", "closeness", "betweenness", "expectedInfluence"]
        if statistics is not None:
            centrality_stats = [s for s in statistics if s != "edge"]

        # Compute original centralities
        from ..centrality import strength as f_str, closeness as f_clo
        from ..centrality import betweenness as f_bet, expected_influence as f_ei
        func_map = {
            "strength": f_str,
            "closeness": f_clo,
            "betweenness": f_bet,
            "expectedInfluence": f_ei,
        }
        original_centralities = {
            s: func_map[s](original) for s in centrality_stats
        }

        proportions = np.linspace(case_max, case_min, case_n)
        seeds = rng.integers(0, 2**31, size=n_boots * case_n)

        if verbose:
            print(f"Running case-dropping bootstrap ({n_boots} boots × {case_n} proportions)...")

        tasks = []
        seed_idx = 0
        for prop in proportions:
            for b in range(n_boots):
                tasks.append((prop, b, int(seeds[seed_idx])))
                seed_idx += 1

        results = Parallel(n_jobs=n_cores, verbose=int(verbose))(
            delayed(_single_case_drop)(
                data, method, original_centralities, prop, b, s, est_kwargs,
            )
            for prop, b, s in tasks
        )

        all_records: list[dict] = []
        for res in results:
            all_records.extend(res)

        case_drop_df = pd.DataFrame(all_records)

        # Also store original stats in boot_statistics for completeness
        orig_records = _extract_statistics(
            original, centrality_stats,
        )
        for r in orig_records:
            r["boot_id"] = "original"
        boot_statistics = pd.DataFrame(orig_records)

        return BootstrapResult(
            original_network=original,
            boot_statistics=boot_statistics,
            boot_type=boot_type,
            n_boots=n_boots,
            case_drop_correlations=case_drop_df,
        )
