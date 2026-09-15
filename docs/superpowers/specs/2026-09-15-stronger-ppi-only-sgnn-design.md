# Stronger PPI-only EncoderSGNN design

## Brief summary

This spec proposes a **PPI-only upgrade path** for `EncoderSGNN`.

The goal is to make the current graph encoder substantially stronger before
adding pathway fusion or a new unified encoder. The new direction keeps the
encoder focused on one biological prior:

1. the PPI graph

but uses that prior more effectively than the current implementation.

The recommended first target is a stronger `EncoderSGNN` with:

1. **isolated-gene retention**
2. **weighted message passing**
3. **sample-conditioned graph pooling**, where the pooling queries adapt to
   each sample instead of staying fixed across all samples
4. **graph-specific regularization**
5. **clearer graph diagnostics and ablations**

This should remain compatible with the current VAE encoder contract and should
be evaluated directly against:

1. `EncoderMLP`
2. `EncoderPathNet`
3. current `EncoderSGNN`

## Goal

Improve `EncoderSGNN` so that it uses the PPI graph more effectively and has a
real chance to outperform `EncoderMLP` as a standalone encoder.

The upgraded encoder should:

1. keep more informative genes in the effective input space
2. use edge strength instead of binary-only connectivity
3. learn stronger sample-specific graph summaries
4. regularize graph learning more appropriately
5. preserve the current VAE output contract

## Scope

This spec covers only:

1. `EncoderSGNN`
2. its internal `PPIEncoder`
3. the config and diagnostics needed to support the stronger PPI-only design

It does not cover:

1. pathway fusion
2. new hybrid encoder classes
3. decoder redesign
4. multi-encoder fusion redesign

## Why a PPI-only iteration first

This is the right first step for three reasons.

### 1. It isolates the value of the graph prior

If we combine PPI and pathway logic too early, it becomes harder to tell
whether the improvement came from:

1. the graph
2. the pathway summary
3. the fusion design

A PPI-only upgrade gives a cleaner answer.

### 2. The current SGNN is still underpowered as a graph model

The current encoder uses the PPI graph, but in a fairly basic way:

1. genes outside retained edges are dropped
2. retained edges become binary
3. graph pooling uses fixed global queries
4. graph-specific regularization is minimal

So there is still room to improve the graph encoder itself before moving to a
larger multi-branch architecture.

### 3. It gives a fairer baseline for later graph+pathway work

If a future unified graph+pathway encoder is added, it should be compared
against a strong SGNN baseline, not only the current simple one.

## Current SGNN limitations

The current `EncoderSGNN` has five main weaknesses that matter for a stronger
PPI encoder.

### 1. Silent dropping of non-edge genes

Genes not present in the retained edge list disappear from the graph input.
That is risky because PPI is incomplete and isolated genes may still carry
useful expression signal.
The stronger SGNN should count and log these genes for every training run so
that we can see how much of the input space is connected versus isolated under
the chosen graph cutoff.

### 2. Edge weights are discarded

The graph currently retains topology after thresholding, but discards the
strength information in `conn`.

At the moment, the threshold itself is controlled by the shared constant
`NETWORK_CUTOFF = 0.5`. The stronger SGNN design should stop relying on that
implicit constant and instead make the cutoff an explicit model parameter.

### 3. Pooling is too rigid

The learned attention queries are fixed across samples, which makes the model
less adaptive to different bulk-expression states.

### 4. No graph-specific regularization

The encoder uses generic dropout, but no explicit graph-aware regularizers such
as DropEdge or attention stabilization.

### 5. `mu_mean` is not internally consistent under positional encoding

This is a correctness issue and should be fixed as part of the same upgrade.

## Chosen design

This spec keeps only the stronger PPI-only SGNN path.

It includes the low-risk fixes plus a larger graph-specific upgrade:

1. sample-conditioned graph pooling
2. stronger graph diagnostics
3. a cleaner sparse-attention design

Advantages:

1. better use of the PPI graph
2. more likely to materially improve reconstruction quality
3. still remains a single-encoder design

Tradeoff:

1. somewhat more ambitious than a cleanup-only pass

This gives a stronger and still interpretable PPI-only encoder before any
graph+pathway fusion work.

## Proposed stronger PPI-only architecture

The upgraded SGNN should still have the same high-level structure:

1. gene-level graph input
2. graph message passing
3. graph pooling
4. shared prediction heads

But each of those stages should be improved.

### 1. Graph input: retain isolated genes

The encoder should no longer keep only genes that appear in retained edges.

#### Proposed behavior

Keep all genes from the configured input gene list that can be aligned to:

1. the encoder input tensor
2. the prior gene-feature table

Genes with no retained graph neighbors should remain as isolated nodes.

#### Why this matters

This allows the encoder to still use:

1. raw expression
2. prior gene features
3. self/residual paths

for genes that are missing from or weakly covered by the PPI network.

#### Config

```yaml
model:
  gnn_gene_retention_mode: "keep_isolated"
```

Supported values:

1. `keep_isolated` (recommended)
2. `drop_isolated` (legacy mode)

### 2. Message passing: preserve weighted edges

The current graph throws away `conn` after thresholding.

#### Proposed behavior

After thresholding by a configurable graph cutoff, retain the surviving `conn`
values and use them in weighted aggregation:

```text
For gene i:
neigh_i = sum_j (w_ij * h_j) / sum_j w_ij
```

Here:

1. `h_j` is the embedding of neighbor gene `j`
2. `w_ij` is the retained PPI weight between genes `i` and `j`
3. `neigh_i` is the aggregated neighbor message for gene `i`

In plain language, each gene receives a weighted average of its neighbors'
embeddings, so stronger PPI edges contribute more than weaker ones. Dividing by
the sum of retained edge weights keeps the activation scale stable.

This should become the default graph aggregation mode.

#### Why this matters

A weighted graph lets the encoder distinguish:

1. weak biological support
2. strong biological support

instead of treating all retained edges identically.

#### Config

```yaml
model:
  gnn_network_cutoff: 0.5
  gnn_edge_weight_mode: "weighted_mean"
```

Supported values for the first implementation:

1. `binary_mean`
2. `weighted_mean`

Notes:

1. `gnn_network_cutoff` replaces the current implicit reliance on the shared
   `NETWORK_CUTOFF` constant
2. the default should remain `0.5` for backward compatibility

### 3. Pooling: sample-conditioned graph queries

This is the main representational upgrade.

The current learned query bank is global:

```text
Q = Q_base
```

That means every sample uses the same query template.

#### Proposed stronger behavior

Condition the graph pooling queries on a coarse sample summary derived from the
graph branch itself:

```text
Q_sample = Q_base + Delta(summary_graph)
```

where:

1. `Q_base` is the current learned query bank
2. `summary_graph` is a simple sample-level graph summary
3. `Delta(...)` is a small projection MLP

#### Recommended summary source

For the first implementation, define:

1. `summary_graph = mean(gene_embeddings, dim=nodes)`

This is simple and stable.

#### Why this matters

This lets the same SGNN focus on different graph regions for different bulk
samples instead of using one fixed global attention pattern.

#### Config

```yaml
model:
  gnn_query_mode: "sample_conditioned"
```

Supported values:

1. `fixed`
2. `sample_conditioned`

### 4. Sparse attention: keep fixed `topk`, but treat it as a first-class knob

The current sparse pooling already uses `topk`, and that should remain fixed in
the first stronger SGNN implementation.

The change here is not to make it adaptive yet, but to treat it as an explicit
ablation parameter.

#### Config

```yaml
model:
  gnn_topk_attention: 1024
```

Recommended ablation range:

1. `128`
2. `256`
3. `512`
4. `1024`

This keeps the first implementation manageable.

### 5. Graph-specific regularization

The first stronger SGNN should add one graph-specific regularizer and one
attention regularizer.

#### 5.1 DropEdge

Add:

```yaml
model:
  gnn_dropedge_rate: 0.0
```

Behavior:

1. apply during training only
2. randomly remove a fraction of retained edges before aggregation

#### 5.2 Attention dropout

Add:

```yaml
model:
  gnn_attention_dropout_rate: 0.0
```

Behavior:

1. apply dropout to attention weights after top-k softmax

This is the simplest way to regularize sparse pooling without redesigning the
entire attention mechanism.

## Required correctness fixes

These should be implemented even if they are not the main source of gain.

### 1. Recompute `mu_mean`

After positional encoding modifies `mu_all_types`, recompute:

1. `mu_mean`

So that the returned encoder output is internally consistent.

### 2. Add graph diagnostics

At initialization, log:

1. total configured genes
2. retained graph genes
3. isolated genes
4. retained edge count
5. effective `topk`
6. query mode
7. edge-weight mode

During training, also surface the connected-versus-isolated gene counts in the
run logs so that graph coverage stays visible for each experiment. These
diagnostics are important for reproducibility and debugging.

## Output contract

The stronger SGNN should continue to expose the same main output fields used by
the current VAE flow:

1. `mu_all_types`
2. `logvar_all_types`
3. `mu_mean`
4. `logvar_mean`
5. `cell_prop`
6. `cell_prop_feature`
7. `cell_type_existed`

This allows the new version to replace the current SGNN without forcing a VAE
integration redesign.

## Configuration additions

Recommended new or clarified config fields:

```yaml
model:
  gnn_gene_retention_mode: "keep_isolated"
  gnn_network_cutoff: 0.5
  gnn_edge_weight_mode: "weighted_mean"
  gnn_query_mode: "sample_conditioned"
  gnn_dropedge_rate: 0.0
  gnn_attention_dropout_rate: 0.0
  gnn_topk_attention: 1024
```

## Validation plan

The stronger SGNN should be validated in three layers.

### 1. Correctness tests

Add focused tests for:

1. isolated genes are retained under `keep_isolated`
2. weighted aggregation uses retained edge weights correctly
3. `mu_mean == mu_all_types.mean(dim=-1)` after positional encoding
4. sample-conditioned queries preserve expected tensor shapes

### 2. Baseline comparisons

Compare:

1. `EncoderMLP`
2. `EncoderPathNet`
3. current `EncoderSGNN`
4. stronger `EncoderSGNN`

This is the key result. The stronger SGNN needs to beat the current SGNN and
ideally improve on `EncoderMLP` in reconstruction-focused metrics.

### 3. Internal ablations

Within the stronger SGNN, compare:

1. binary vs weighted edges
2. fixed vs sample-conditioned queries
3. no DropEdge vs DropEdge
4. different `topk` values

## Metrics to track

Track at least:

1. reconstruction CCC
2. cell-proportion prediction accuracy
3. training stability
4. GPU memory usage
5. runtime per batch

## Recommendation

The first stronger PPI-only SGNN implementation should include:

1. isolated-gene retention
2. weighted message passing
3. sample-conditioned graph pooling
4. DropEdge
5. attention dropout
6. `mu_mean` consistency fix

This is the most promising PPI-only upgrade path before trying a larger
graph+pathway encoder.
