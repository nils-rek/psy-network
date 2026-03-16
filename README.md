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

### 3. Bootstrap analysis

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

### 4. Visualization

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

# Bootstrap plots (see §3 above)
boot = psynet.bootnet(data, n_boots=500, seed=1)
psynet.plot_edge_accuracy(boot)
```

### 5. Multi-group network estimation (Joint Graphical Lasso)

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

### 6. Synthetic datasets

PsyNet ships with data generators useful for demos and testing:

```python
# Big Five personality (25 items: O1-O5, C1-C5, E1-E5, A1-A5, N1-N5)
bfi = psynet.make_bfi25(n=500, seed=42)    # 1–6 Likert scale

# PHQ-9-like depression symptoms (dep1-dep9, three symptom clusters)
phq = psynet.make_depression9(n=300, seed=123)  # 0–3 Likert scale

# Multi-group data (groups with shared + differential edges)
mg = psynet.make_multigroup(n_per_group=200, n_groups=2, p=9, seed=42)
```

### 7. Extending PsyNet with custom estimators

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
