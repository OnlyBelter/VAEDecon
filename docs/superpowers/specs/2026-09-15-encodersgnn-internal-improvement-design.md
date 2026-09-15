# Unified PPI-pathway encoder design

## Brief summary

This spec replaces the previous long `EncoderSGNN` improvement menu with one
clear recommendation:

> add a new encoder that combines **gene-level PPI message passing** and
> **pathway-level summarization** inside a single architecture.

The current `EncoderSGNN` should remain as the PPI-only baseline, and
`EncoderPathNet` should remain as the pathway-only baseline. The new encoder
becomes the stronger experimental model.

This design aims to outperform `EncoderMLP` by using both:

1. **local graph structure** across genes
2. **higher-level pathway structure** across gene sets

while keeping the same VAE integration contract used by the current encoders.

## Goal

Design a stronger encoder that uses PPI information better than the current
`EncoderSGNN`, while also leveraging the pathway representation already used by
`EncoderPathNet`.

The new encoder should:

1. preserve more of the useful structure in the PPI graph
2. avoid silently discarding informative genes
3. exploit pathway-level organization in addition to gene-level interactions
4. output the same main fields expected by the current VAE path:
   - `mu_all_types`
   - `logvar_all_types`
   - `mu_mean`
   - `logvar_mean`
   - `cell_prop`
   - `cell_prop_feature`

## Main decision

Instead of continuing to enlarge `EncoderSGNN`, add a **new encoder**.

Recommended class name:

- `EncoderGraphPathway`

Alternative acceptable name:

- `EncoderPPIPathNet`

Recommendation:

- prefer `EncoderGraphPathway` because it describes the architecture more
  clearly

## Why a new encoder is better than modifying `EncoderSGNN`

`EncoderSGNN` currently represents one simple and interpretable idea:

1. filter genes to a PPI-supported set
2. run graph message passing
3. pool the gene graph into one sample representation

That is useful as a baseline.

If we force pathway handling, stronger regularization, adaptive pooling, and
graph-quality fixes all into that same class, the result becomes harder to:

1. understand
2. debug
3. ablate
4. compare fairly against current baselines

Adding a new encoder is cleaner because it keeps:

1. `EncoderMLP` as the non-graph baseline
2. `EncoderPathNet` as the pathway-only baseline
3. `EncoderSGNN` as the PPI-only baseline
4. `EncoderGraphPathway` as the stronger unified model

That makes model comparisons much easier to interpret.

## Current limitations that the new encoder should fix

The current `EncoderSGNN` has several limitations that motivate a new design.

### 1. Silent dropping of non-edge genes

Genes not present in the retained PPI edge list are removed from the effective
encoder input. That can discard useful signal because PPI is incomplete.

### 2. Edge weights are discarded

After thresholding, retained edges are treated as binary connections. This
throws away confidence or strength information from `conn`.

### 3. No pathway-level structure

The current SGNN works only at gene-node level. It does not explicitly model
functional modules or pathway summaries.

### 4. Pooling is too rigid

The current learned attention queries are global and fixed across samples. That
can be limiting for heterogeneous bulk expression patterns.

### 5. No graph-specific regularization

The current encoder uses generic dropout, but not graph-specific tools such as
DropEdge or attention regularization.

### 6. `mu_mean` output is internally inconsistent under positional encoding

This is a correctness issue and should still be fixed even if a new encoder is
added.

## Proposed architecture

The new encoder has three parts:

1. a **gene graph branch**
2. a **pathway branch**
3. a **fusion block**

### 1. Gene graph branch

This branch starts from gene-level input and uses the PPI graph directly.

#### Inputs

For each gene:

1. expression value
2. prior gene features such as mean/std statistics

#### Core design

The graph branch should:

1. retain isolated genes instead of silently dropping them
2. use weighted edges after thresholding
3. run multi-layer message passing
4. support graph-specific regularization
5. produce contextualized gene embeddings

#### Output

The branch output is:

1. per-gene embeddings
2. one graph-derived sample summary vector

### 2. Pathway branch

This branch reuses the main idea of `EncoderPathNet`.

#### Inputs

1. raw bulk gene expression
2. pathway mask derived from the configured pathway files

#### Core design

The pathway branch should:

1. project genes into pathway profiles
2. encode pathway profiles with a lightweight MLP or pathway-token block
3. produce one pathway-derived sample summary vector

This should stay simpler than the graph branch. Its job is to provide
higher-level biological organization, not to duplicate the full graph encoder.

#### Output

The branch output is:

1. one pathway summary vector for each sample

### 3. Fusion block

The graph summary and pathway summary should then be fused before the final
prediction heads.

Recommended first implementation:

1. concatenate graph summary and pathway summary
2. pass through a small fusion MLP
3. use the fused representation for:
   - `mu/logvar`
   - optional cell proportion prediction

This keeps the first implementation simple.

## How pathway information should help the graph encoder

The pathway branch is not just extra input. It should improve the graph branch
in one concrete way.

Recommended first mechanism:

### Pathway-conditioned graph pooling

Use the pathway summary vector to condition the graph pooling step.

Concretely:

1. compute pathway summary from pathway profiles
2. project pathway summary into a small conditioning vector
3. use that vector to modulate or offset the graph pooling queries

This is better than using only fixed global queries because it lets the graph
branch focus on different gene regions for different bulk samples.

Form:

```text
Q_sample = Q_base + Delta(pathway_summary)
```

where:

1. `Q_base` is the learned base query bank
2. `Delta(...)` is a small learned projection from pathway summary to query
   offsets

This is the main mechanism by which pathway structure should guide graph
pooling in the first implementation.

## Recommended message-passing behavior

The graph branch should use:

### 1. Weighted aggregation

After thresholding, preserve retained `conn` values and normalize by weighted
degree:

```text
neigh = (A_weighted @ h) / weighted_degree
```

### 2. Isolated-gene retention

Genes with no retained neighbors should still stay in the graph branch as
isolated nodes. They should contribute through:

1. their own expression signal
2. their prior gene features
3. the residual/self path

### 3. One graph-specific regularizer

For the first implementation, add:

- `DropEdge`

This is enough for a first graph-aware regularization mechanism without making
the design too large.

## What should stay simple in the first version

The first version should not try to solve everything at once.

Keep the following simple:

1. one pathway summary vector, not a large pathway-token transformer
2. one fusion MLP, not multiple cross-branch attention blocks
3. one pathway-conditioning path for graph pooling
4. fixed `topk` as a tunable config, not a learned adaptive sparsity system

## Required correctness fixes

Even though the main recommendation is a new encoder, the following correctness
fixes should still be applied either to the new encoder implementation or to
the existing SGNN code reused by it.

### 1. Recompute `mu_mean` after positional encoding

Ensure:

```text
mu_mean == mu_all_types.mean(dim=-1)
```

at encoder return time.

### 2. Make retained gene counts visible

At initialization, log:

1. total configured genes
2. genes retained in graph branch
3. isolated retained genes
4. retained edge count

This is important for experiment interpretation.

## Configuration additions

The new encoder should introduce a small number of explicit config fields.

Recommended fields:

```yaml
model:
  encoders: ["EncoderGraphPathway"]
  gnn_gene_retention_mode: "keep_isolated"
  gnn_edge_weight_mode: "weighted_mean"
  gnn_dropedge_rate: 0.0
  gnn_topk_attention: 1024
  graph_pathway_fusion_dim: 256
  graph_pathway_query_conditioning: true
```

Notes:

1. `gnn_topk_attention` stays fixed but explicitly tunable
2. `graph_pathway_query_conditioning` enables pathway-conditioned graph pooling
3. `keep_isolated` should be the recommended default

## Validation plan

The design should be validated in three steps.

### Step 1: Correctness

Add tests for:

1. isolated genes are retained
2. weighted edges are used correctly
3. `mu_mean` is recomputed consistently
4. the fused encoder still returns the standard output contract

### Step 2: Ablation against existing baselines

Compare:

1. `EncoderMLP`
2. `EncoderPathNet`
3. `EncoderSGNN`
4. `EncoderGraphPathway`

This is critical. The new encoder only earns its complexity if it improves on
these baselines.

### Step 3: Internal ablations

For the new encoder, compare:

1. graph branch only
2. pathway branch only
3. graph + pathway fusion
4. graph + pathway-conditioned graph pooling

This will tell us whether the unified design is helping for the right reason.

## Metrics to track

Track at least:

1. reconstruction CCC
2. cell-proportion prediction accuracy
3. training stability
4. GPU memory usage
5. runtime per batch

If the new encoder is truly better than `EncoderMLP`, it should show a clear
gain in at least the reconstruction metrics without becoming too unstable or
too expensive.

## Recommendation

The recommended implementation path is:

1. keep `EncoderSGNN` and `EncoderPathNet` unchanged as baselines
2. add a new encoder class: `EncoderGraphPathway`
3. build it from:
   - a weighted PPI graph branch
   - a simple pathway summary branch
   - a fusion MLP
   - pathway-conditioned graph pooling
4. keep the first version small and easy to ablate

This is the cleanest way to build a stronger PPI-based encoder that has a real
chance to beat `EncoderMLP` while still being interpretable and testable.
