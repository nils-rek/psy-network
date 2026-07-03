# Changelog

## 0.2.0 (unreleased)

### Breaking / numerical changes

- **Directed temporal networks are transposed.** All directed adjacency
  matrices (graphicalVAR temporal, mlVAR temporal, per-subject networks) and
  `MultilevelNetwork.pvalues` now follow the convention
  `adjacency[i, j]` = directed edge *i → j* (networkx/qgraph standard).
  Previously the estimators stored `B[outcome, predictor]` while
  `edges_df`/`to_networkx`/plots interpreted `A[i, j]` as *i → j*, so
  **arrows in directed plots and exports were reversed**; in/out-strength
  followed the opposite convention and disagreed with the plots.
- **Fused Joint Graphical Lasso estimates change.** The K=2 closed form
  applied only half the intended similarity penalty; the K≥3 prox penalized
  only adjacent group pairs instead of all pairs. Both are fixed (verified
  against brute-force optimization), and the fused prox is now applied
  before L1 soft-thresholding, matching the correct prox decomposition.
- **`difference_test` criterion changed.** Significance now follows R
  bootnet: a pair differs when the bootstrap CI of its difference excludes
  zero. Previously, ties at zero were ignored, so two structurally-zero
  glasso edges tested as "significantly different" with p = 0. P-values in
  `.attrs["p_values"]` are tie-corrected; CI bounds are exposed via
  `.attrs["ci_lower"]` / `.attrs["ci_upper"]`.
- **EBIC λ selection uses thresholded edge counts.** Numerically tiny
  solver residue no longer inflates the EBIC penalty, so the selected λ
  (and hence edge sets) can shift slightly for EBICglasso, contemporaneous,
  between-subjects, and JGL networks.
- **Synthetic datasets change for fixed seeds.** `make_depression9` could
  never generate PHQ category 0 (and piled ~50% of mass on 3);
  `make_bfi25` clustered responses mid-scale because latent scores were not
  standardized before the probability integral transform.
- **`bootnet_group` defaults changed:** `n_cores=-1` (was 1), and replicates
  now reuse the originally selected λ₁/λ₂ (`reselect_lambdas=False`),
  matching bootnet's fixed-tuning bootstrap. Pass `reselect_lambdas=True`
  for the old behavior.
- **Timeseries VAR residuals now include the Lasso intercept**, slightly
  changing the contemporaneous network input for non-centered data.
- **scikit-learn ≥ 1.7 required** (`LassoCV(alphas=<int>)` semantics).

### Fixes

- ADMM (JGL) now checks primal *and* dual residuals with Boyd-style
  relative/absolute tolerances and warns when `max_iter` is reached without
  convergence (new `tol_abs` parameter).
- Multilevel estimation no longer raises `KeyError` when a subject has
  observations but no valid consecutive lag pairs; such subjects are
  excluded with a warning.
- The statsmodels residual fallback (singular covariance) reconstructs
  per-subject conditional residuals instead of mixing marginal residuals
  into the pooled contemporaneous correlation.
- `auto_re` downgrades are now reported in the convergence summary, and
  `fit_info[var]["actual_re"]` reflects the structure actually attempted.
- Cross-sectional estimators validate input: non-numeric columns raise
  (previously they were silently dropped, desynchronizing labels from the
  adjacency), missing data warns and the effective sample size (complete
  rows) is used for `n_observations` and EBIC.
- `pcor` falls back to the pseudo-inverse (with a warning) for singular or
  near-singular correlation matrices.
- `shared_layout` was silently ignored by `plot_ts_networks` and
  `plot_multilevel_networks`; it now works.
- `kamada_kawai` layouts use the same 1/|weight| distances in single- and
  multi-panel plots.
- `closeness(normalized=False)` now returns `1/Σd` (the NetworkX
  normalization was not invertible on disconnected graphs); isolated nodes
  get closeness 0 instead of NaN.
- Case-dropping bootstrap workers receive the exact integer retain count,
  eliminating an off-by-one between the labeled and actual drop proportion.
- `Network` validates adjacency shape against labels and stores a read-only
  defensive copy; `edges_df` keeps its columns for empty networks.

### New

- `scale=` option on `estimate_var_network` (z-score standardization; the
  Lasso penalty is scale-dependent and R's graphicalVAR standardizes by
  default).
- `tests/test_datasets.py` plus regression tests for arrow orientation,
  fused-prox correctness, difference-test correctness, and input validation.

### Internal

- Deduplicated: shared contemporaneous-from-residuals helper, shared
  RE-fallback chain/warning detection (`multilevel/_re_common.py`),
  `precision_to_pcor`, and dataset PD-repair helpers. Fixed-seed outputs
  verified byte-identical across the refactor.
