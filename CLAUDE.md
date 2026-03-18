# PsyNet — Python Psychometric Network Analysis Toolkit

## Overview
A Python equivalent of R's `bootnet` package for psychometric network analysis: network estimation (EBICglasso, partial correlations, correlations), bootstrap accuracy/stability analysis, centrality measures, and publication-quality visualization.

## Project Structure
- `src/psynet/` — package source (src layout)
- `src/psynet/estimation/` — registry-based estimators (cor, pcor, EBICglasso)
- `src/psynet/bootstrap/` — nonparametric and case-dropping bootstrap
- `src/psynet/plotting/` — network, centrality, and bootstrap plots
- `src/psynet/centrality.py` — strength, closeness, betweenness, expected influence
- `src/psynet/network.py` — frozen Network dataclass (core result object)
- `src/psynet/datasets.py` — synthetic data generators (BFI-25, PHQ-9)
- `tests/` — pytest suite (42 tests)

## Build & Test
```bash
pip install -e ".[dev]"   # install in editable mode with dev deps
pytest tests/ -v           # run all tests
```

## Key Conventions
- **Frozen dataclass** for Network — immutability is intentional
- **`@register` decorator** for adding new estimators — just create a file in `estimation/` and decorate
- **EBICglasso uses sklearn's GraphicalLasso** — no skggm dependency; manual EBIC lambda selection (~30 lines)
- `.corr().values.copy()` — pandas returns read-only arrays; always copy before in-place ops
- **Long-form DataFrames** for bootstrap stats — enables easy groupby aggregation
- **joblib** for bootstrap parallelism

## Dependencies
numpy, scipy, pandas, scikit-learn, networkx, matplotlib, joblib

## GitHub
https://github.com/nils-rek/psy-network

## Related Repos
- **psy-network-eval** — https://github.com/nils-rek/psy-network-eval
  Hands-on evaluation comparing original R packages (bootnet, qgraph, etc.) with this Python implementation. Contains a `recommendations.md` with improvement suggestions derived from evaluation results.
