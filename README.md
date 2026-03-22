# PsyNet

A Python toolkit for psychometric network analysis — estimate, bootstrap, and visualize psychological networks. Inspired by R's [`bootnet`](https://cran.r-project.org/package=bootnet) package.

## Installation

```bash
pip install psynet
```

Or install from source:

```bash
git clone https://github.com/nils-rek/psy-network.git
cd psy-network
pip install -e ".[dev]"
```

**Requires Python 3.10+.** Dependencies: numpy, scipy, pandas, scikit-learn, networkx, matplotlib, joblib.

## Quick start

```python
import psynet

# Generate example data (Big Five personality, 500 participants)
data = psynet.make_bfi25(n=500, seed=42)

# Estimate a regularized partial correlation network
net = psynet.estimate_network(data, method="EBICglasso")

# Inspect edges
print(net.edges_df.head(10))

# Plot the network
net.plot(layout="spring", title="Big Five Network")
```

## Tutorial

### 1. Network estimation

PsyNet offers three estimation methods out of the box:

| Method | Description |
|---|---|
| `"EBICglasso"` | EBIC-tuned graphical LASSO (sparse partial correlations) — **default** |
| `"pcor"` | Unregularized partial correlations |
| `"cor"` | Zero-order correlations |

```python
import psynet

data = psynet.make_bfi25(n=500, seed=42)

# EBICglasso — sparse, regularized (recommended for most use cases)
net = psynet.estimate_network(data, method="EBICglasso", gamma=0.5)

# Partial correlations — no regularization
net_pcor = psynet.estimate_network(data, method="pcor")

# Simple correlations with a threshold
net_cor = psynet.estimate_network(data, method="cor", threshold=0.1)

# Check available methods
print(psynet.available_methods())  # ['EBICglasso', 'cor', 'pcor']
```

The result is a `Network` object:

```python
net.adjacency       # p × p numpy array of edge weights
net.labels           # node names (column names from your DataFrame)
net.method           # estimation method used
net.n_observations   # sample size
net.n_nodes          # number of nodes

net.adjacency_df     # labeled DataFrame view of the adjacency matrix
net.edges_df         # long-form DataFrame of non-zero edges (node1, node2, weight)
net.to_networkx()    # convert to a NetworkX graph
```

### 2. Centrality

Compute standard centrality indices for any estimated network:

```python
import psynet

data = psynet.make_bfi25(n=500, seed=42)
net = psynet.estimate_network(data)

# All four measures at once (returns a DataFrame)
cent = psynet.centrality(net)
print(cent)

# Individual measures (each returns a pd.Series)
s  = psynet.strength(net)            # sum of absolute edge weights
ei = psynet.expected_influence(net)  # sum of signed edge weights
c  = psynet.closeness(net)           # closeness centrality
b  = psynet.betweenness(net)         # betweenness centrality
```

Visualize centrality with a dot plot:

```python
psynet.plot_centrality(net, standardized=True)
```

### 3. Community detection

Community detection identifies clusters of closely related nodes — a network-based alternative to factor analysis. PsyNet implements the three algorithms recommended by [Christensen et al. (2023)](https://doi.org/10.3758/s13428-023-02106-4) for psychological data:

| Method | Description |
|---|---|
| `"walktrap"` | Random-walk based clustering (default) — deterministic, no igraph dependency |
| `"louvain"` | Louvain modularity optimization |
| `"greedy_modularity"` | Fast greedy modularity maximization |

```python
import psynet

data = psynet.make_bfi25(n=500, seed=42)
net = psynet.estimate_network(data)

# Detect communities (walktrap by default)
comm = net.communities()
print(comm.value_counts())
# Expected: ~5 communities for Big Five data

# Use a specific algorithm
comm_louv = psynet.louvain(net, resolution=1.0, seed=42)
comm_greedy = psynet.greedy_modularity(net)

# Or via the dispatcher
comm_wt = psynet.communities(net, method="walktrap", steps=4)
```

All functions return a `pd.Series` mapping node labels to integer community IDs.

#### Community visualization

Plot the network with nodes colored by community membership:

```python
# Via Network convenience method
net.plot_communities()

# Or directly with customization
from psynet import plot_community
fig = plot_community(
    net, comm,
    layout="spring",
    palette=["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"],
    title="Big Five Communities",
)
fig.savefig("communities.png", dpi=150, bbox_inches="tight")
```

**Negative edges:** By default, absolute edge weights are used for community detection (`absolute_weights=True`), which is standard practice for signed psychometric networks.

### 4. Bootstrap analysis

Bootstrap analysis lets you assess the accuracy and stability of your network. PsyNet supports two types:

#### Edge and centrality accuracy (nonparametric bootstrap)

Resample rows with replacement and re-estimate the network many times to get confidence intervals around edge weights and centrality indices.

```python
import psynet

data = psynet.make_bfi25(n=500, seed=42)

boot = psynet.bootnet(
    data,
    n_boots=1000,
    boot_type="nonparametric",
    method="EBICglasso",
    n_cores=4,        # parallel bootstraps via joblib
    seed=1,
)

# Summary statistics for edges
print(boot.summary(statistic="edge").head(10))
#   node1  node2  sample   mean     sd  ci_lower  ci_upper

# Summary for a centrality measure
print(boot.summary(statistic="strength"))

# Plot edge weight accuracy (95% CIs)
psynet.plot_edge_accuracy(boot)

# Test whether edges differ significantly from each other
diff = boot.difference_test(statistic="edge")
psynet.plot_difference(boot, statistic="edge")
```

#### Centrality stability (case-dropping bootstrap)

Drop increasing proportions of cases and check how stable centrality rankings remain, quantified by the CS-coefficient.

```python
boot_case = psynet.bootnet(
    data,
    n_boots=1000,
    boot_type="case",
    method="EBICglasso",
    case_min=0.25,
    case_max=0.75,
    n_cores=4,
    seed=2,
)

# CS-coefficient: max proportion droppable while r ≥ 0.7 with original
print("CS(strength):", boot_case.cs_coefficient("strength"))

# Plot stability curves
psynet.plot_centrality_stability(boot_case)

# Difference test on centrality
psynet.plot_difference(boot_case, statistic="strength")
```

A CS-coefficient ≥ 0.5 is considered acceptable and ≥ 0.7 is good (see [Epskamp, Borsboom & Fried, 2018](https://doi.org/10.3758/s13428-017-0862-1)).

### 5. Visualization

```python
import psynet

data = psynet.make_depression9(n=300, seed=123)
net = psynet.estimate_network(data)

# Network graph — positive edges are solid blue, negative edges are dashed red
fig = psynet.plot_network(
    net,
    layout="spring",          # "spring", "circular", or "kamada_kawai"
    node_size="strength",     # size nodes by centrality (or pass an int)
    node_color="#87CEEB",
    title="Depression Symptom Network",
)
fig.savefig("network.png", dpi=150, bbox_inches="tight")

# Centrality dot plot
psynet.plot_centrality(net, measures=["strength", "expectedInfluence"])

# Bootstrap plots (see §4 above)
boot = psynet.bootnet(data, n_boots=500, seed=1)
psynet.plot_edge_accuracy(boot)
```

### 6. Multi-group network estimation (Joint Graphical Lasso)

PsyNet supports estimating networks across multiple groups simultaneously using the Joint Graphical Lasso (JGL; [Danaher, Wang & Witten, 2014](https://doi.org/10.1111/rssb.12033)). This encourages shared structure across groups while allowing group-specific differences.

Two penalty types are available:

| Penalty | Description |
|---|---|
| `"fused"` | Penalizes differences in edge weights between groups — networks share similar edges |
| `"group"` | Penalizes joint non-zeros across groups — networks share the same sparsity pattern |

```python
import psynet

# Generate multi-group data (groups share ~70% of edges, differ in ~30%)
data = psynet.make_multigroup(n_per_group=200, n_groups=2, p=9, seed=42)

# Estimate group networks — lambda selection via EBIC by default
gn = psynet.estimate_group_network(data, group_col="group", penalty="fused")

# Or pass a list of DataFrames (one per group)
# gn = psynet.estimate_group_network([df_group1, df_group2])
```

The result is a `GroupNetwork` object:

```python
gn["Group1"]          # access individual Network objects by group label
gn.group_labels       # list of group names
gn.lambda1            # selected sparsity penalty
gn.lambda2            # selected similarity/fusion penalty

gn.centrality()       # long-form DataFrame with centrality per group
gn.compare_edges()    # long-form DataFrame comparing edges across groups
```

#### Tuning parameters

Lambda selection is automatic by default (EBIC), but you can control the search:

```python
gn = psynet.estimate_group_network(
    data,
    group_col="group",
    penalty="group",
    criterion="ebic",          # "ebic", "bic", or "aic"
    gamma=0.5,                 # EBIC sparsity tuning (higher = sparser)
    search="sequential",       # "sequential" or "simultaneous" grid search
    n_lambda1=30,              # grid size for λ₁
    n_lambda2=30,              # grid size for λ₂
)

# Or supply lambdas directly to skip selection
gn = psynet.estimate_group_network(data, group_col="group", lambda1=0.1, lambda2=0.05)
```

#### Visualization

```python
# Side-by-side network plots (shared layout for visual comparison)
gn.plot(layout="spring", shared_layout=True)

# Compare centrality across groups
psynet.plot_group_centrality_comparison(gn, statistic="strength")
```

#### Bootstrap for group networks

Assess edge accuracy with nonparametric bootstrap (resampling within each group):

```python
boot = psynet.bootnet_group(
    data,
    group_col="group",
    n_boots=500,
    n_cores=4,
    seed=1,
    penalty="fused",
)

# Summary with confidence intervals
print(boot.summary(statistic="edge"))
print(boot.summary(statistic="strength", group="Group1"))

# Faceted edge accuracy plot (one panel per group)
boot.plot_edge_accuracy()
```

### 7. Time-series network estimation (graphicalVAR)

PsyNet supports time-series network analysis for intensive longitudinal data (e.g., ESM/EMA). This decomposes dynamics into two networks:

| Network | Type | Description |
|---|---|---|
| **Temporal** | Directed | VAR(1) coefficients — how variables at *t-1* predict variables at *t* |
| **Contemporaneous** | Undirected | Partial correlations of VAR residuals — same-timepoint associations after accounting for temporal effects |

This mirrors R's [`graphicalVAR`](https://cran.r-project.org/package=graphicalVAR) package using a two-step approach: (1) sparse VAR via column-wise Lasso, (2) EBIC graphical lasso on residuals.

```python
import psynet

# Generate synthetic VAR(1) time-series data
data = psynet.make_var_data(n_timepoints=500, p=6, seed=42)

# Estimate temporal + contemporaneous networks
ts = psynet.estimate_var_network(data)
```

The result is a `TSNetwork` object:

```python
ts.temporal           # directed Network (VAR coefficients B)
ts.contemporaneous    # undirected Network (partial correlations of residuals)
ts.labels             # variable names
ts.n_observations     # effective observations after lagging
ts.n_timepoints       # original number of timepoints
ts.n_nodes            # number of variables

# Access adjacency matrices
ts.temporal.adjacency_df       # labeled B matrix
ts.contemporaneous.edges_df    # non-zero contemporaneous edges
```

#### Centrality for both networks

```python
cent = ts.centrality()
# Returns DataFrame with columns: strength, closeness, betweenness,
# expectedInfluence, network, node
# 'network' column is "temporal" or "contemporaneous"
```

#### Visualization

```python
# Side-by-side plot: temporal (directed, with arrows) and contemporaneous (undirected)
ts.plot(layout="spring")

# Or call directly
psynet.plot_ts_networks(ts, layout="circular")
```

#### ESM gap handling (beep/day)

For experience sampling data with gaps between days, pass `beep` and `day` columns to exclude invalid lag pairs:

```python
# Data with beep (measurement within day) and day columns
ts = psynet.estimate_var_network(data, beep="beep", day="day")
# Only consecutive beeps within the same day are used for lagging
```

### 8. Multilevel VAR network estimation (mlVAR)

For multi-subject ESM/EMA data, PsyNet estimates multilevel VAR networks that decompose dynamics into three layers — mirroring R's [`mlVAR`](https://cran.r-project.org/package=mlVAR) package:

| Network | Type | Description |
|---|---|---|
| **Temporal** | Directed | Fixed-effect VAR(1) coefficients from mixed-effects models |
| **Contemporaneous** | Undirected | EBIC-glasso on pooled residuals across subjects |
| **Between-subjects** | Undirected | EBIC-glasso on subject-level means |

```python
import psynet

# Generate synthetic multilevel VAR data (20 subjects, 50 timepoints each)
data = psynet.make_mlvar_data(n_subjects=20, n_timepoints=50, p=4, seed=42)

# Estimate all three networks
mlvar = psynet.estimate_mlvar_network(data, subject="subject", beep="beep")
```

The result is an `MLVARNetwork` object:

```python
mlvar.temporal            # directed Network (fixed-effect VAR coefficients)
mlvar.contemporaneous     # undirected Network (partial correlations of residuals)
mlvar.between_subjects    # undirected Network (between-subject partial correlations)
mlvar.subject_temporal    # dict of per-subject directed Networks
mlvar.pvalues             # p-values for temporal fixed effects
mlvar.labels              # variable names
mlvar.subject_ids         # list of subject identifiers

# Centrality across all three networks
cent = mlvar.centrality()
```

#### Visualization

```python
# Three-panel plot: temporal, contemporaneous, between-subjects
mlvar.plot(layout="spring")

# Or call directly
psynet.plot_mlvar_networks(mlvar, layout="circular")
```

#### ESM gap handling

For data with day boundaries, pass `beep` and `day` columns to exclude invalid cross-day lag pairs:

```python
mlvar = psynet.estimate_mlvar_network(data, subject="subject", beep="beep", day="day")
```

**Note:** mlVAR requires `statsmodels` (optional dependency) for mixed-effects model estimation.

### 9. Synthetic datasets

PsyNet ships with data generators useful for demos and testing:

```python
# Big Five personality (25 items: O1-O5, C1-C5, E1-E5, A1-A5, N1-N5)
bfi = psynet.make_bfi25(n=500, seed=42)    # 1–6 Likert scale

# PHQ-9-like depression symptoms (dep1-dep9, three symptom clusters)
phq = psynet.make_depression9(n=300, seed=123)  # 0–3 Likert scale

# Multi-group data (groups with shared + differential edges)
mg = psynet.make_multigroup(n_per_group=200, n_groups=2, p=9, seed=42)

# VAR(1) time-series data (stationary process with sparse dynamics)
ts_data = psynet.make_var_data(n_timepoints=500, p=6, seed=42)

# Multilevel VAR data (multiple subjects with random effects on B matrix)
mlvar_data = psynet.make_mlvar_data(n_subjects=20, n_timepoints=50, p=4, seed=42)
```

### 10. Extending PsyNet with custom estimators

Add your own estimation method by decorating a class in `src/psynet/estimation/`:

```python
from psynet.estimation._registry import register
from psynet.network import Network

@register("my_method")
class MyEstimator:
    name = "my_method"

    def estimate(self, data, **kwargs):
        adj = ...  # compute your p × p adjacency matrix
        return Network(
            adjacency=adj,
            labels=list(data.columns),
            method=self.name,
            n_observations=len(data),
        )
```

Then use it like any built-in method:

```python
net = psynet.estimate_network(data, method="my_method")
```

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
