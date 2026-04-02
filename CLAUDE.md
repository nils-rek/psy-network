# PsyNet — Python Psychometric Network Analysis Toolkit

## Overview
A Python equivalent of R's `bootnet` package for psychometric network analysis: network estimation (EBICglasso, partial correlations, correlations), time-series VAR networks (`graphicalVAR`-style), multilevel VAR networks (`mlVAR`-style), bootstrap accuracy/stability analysis, centrality measures, community detection, and publication-quality visualization.

## Project Structure
- `src/psynet/` — package source (src layout)
- `src/psynet/estimation/` — registry-based estimators (cor, pcor, EBICglasso)
- `src/psynet/timeseries/` — single-subject VAR(1) temporal + contemporaneous networks
- `src/psynet/multilevel/` — multilevel VAR for multi-subject ESM data (temporal, contemporaneous, between-subjects)
- `src/psynet/bootstrap/` — nonparametric and case-dropping bootstrap
- `src/psynet/group/` — Joint Graphical Lasso for multi-group networks
- `src/psynet/plotting/` — network, centrality, bootstrap, time-series, and multilevel plots
- `src/psynet/centrality.py` — strength, closeness, betweenness, expected influence
- `src/psynet/community.py` — walktrap, louvain, greedy modularity
- `src/psynet/network.py` — frozen Network dataclass (core result object)
- `src/psynet/datasets.py` — synthetic data generators (BFI-25, PHQ-9, multi-group, VAR, multilevel)
- `src/psynet/_glasso_utils.py` — shared EBIC-glasso pipeline (`_fit_ebic_glasso`)
- `src/psynet/_validation_utils.py` — shared VAR validation helpers
- `tests/` — pytest suite (175 tests)

## Build & Test
```bash
pip install -e ".[dev]"   # install in editable mode with dev deps
pytest tests/ -v           # run all tests
```

## Key Conventions
- **Frozen dataclass** for Network — immutability is intentional
- **`@register` decorator** for adding new estimators — just create a file in `estimation/` and decorate
- **EBICglasso pipeline is centralized** in `_glasso_utils._fit_ebic_glasso()` — used by `estimation/ebicglasso.py`, `timeseries/_contemporaneous.py`, `multilevel/_contemporaneous.py`, and `multilevel/_between.py`
- **VAR validation helpers** are centralized in `_validation_utils.py` — shared by `timeseries/` and `multilevel/`
- `.corr().values.copy()` — pandas returns read-only arrays; always copy before in-place ops
- **Long-form DataFrames** for bootstrap stats — enables easy groupby aggregation
- **joblib** for bootstrap parallelism
- **timeseries/ and multilevel/ are separate subpackages** — different return types (`TSNetwork` vs `MultilevelNetwork`), different estimation engines (sklearn LassoCV vs statsmodels mixed-effects), and different dependency requirements (statsmodels is optional)

## Known Divergences from R
- **Between-subjects network sparsity**: EBIC-glasso applied to random intercept correlations may produce different sparsity patterns than R's mlVAR due to lambda grid and solver differences (sklearn's GraphicalLasso vs R's glasso). Edge weight correlations are high (r > 0.9 with lme4) but structure agreement (which edges are non-zero) can be lower. The `between_gamma` parameter allows tuning sparsity.

## Dependencies
numpy, scipy, pandas, scikit-learn, networkx, matplotlib, joblib

Optional: statsmodels (required for multilevel estimation only), rpy2 (recommended for mlVAR — enables lme4 backend)

## GitHub
https://github.com/nils-rek/psy-network

## Related Repos
- **psy-network-eval** — https://github.com/nils-rek/psy-network-eval
  Hands-on evaluation comparing original R packages (bootnet, qgraph, etc.) with this Python implementation. Contains a `recommendations.md` with improvement suggestions derived from evaluation results.
