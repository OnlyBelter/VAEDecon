# Transformer Encoder for Cell Proportions, Bulk Context, and Cell-Type Representations

## Goal

Evaluate and formalize a Transformer-based encoder design that can be used in
two related ways:

1. as an independent model for predicting cell proportions directly from bulk
   gene expression
2. as an encoder inside the current VAE workflow, where it predicts cell
   proportions while also producing:
   - a sample-level bulk-context summary
   - cell-type-specific latent representations for cell-type-specific GEP
     decoding

The design should make it clear what already exists in the current codebase,
what should be improved, how the component fits into the end-to-end VAE
workflow, and what its advantages and tradeoffs are relative to `EncoderMLP`.

This document is the **system-level** spec for how the Transformer encoder is
used:

1. as a standalone cell-proportion model
2. as a VAE encoder
3. as part of hybrid and multi-encoder workflows

The detailed **encoder-internal** redesign is specified separately in:

- `2026-08-18-gene-transformer-encoder-internal-improvement-design.md`

## Why This Spec

The current project now has two relevant encoder directions:

1. the existing `EncoderMLP` / `EncoderResMLP` workflow, which is stable and
   already integrated into the full VAE pipeline
2. the existing `GeneTransformerEncoder`, which already contains the core idea
   of a global sample token plus per-cell-type tokens, but is not yet written
   as a fully explicit multitask design for:
   - direct cell-proportion prediction
   - bulk-context summarization
   - cell-type-specific representation learning

This spec turns that Transformer direction into a clearer design target and
defines how to test it both independently and inside the VAE.

The internal Transformer redesign itself is intentionally not repeated in full
here. That design now lives in the companion encoder-focused spec so this
document can stay focused on system integration and experimental positioning.

At the same time, the design must remain compatible with the broader
multi-encoder setting already present in the project, including:

1. `EncoderMLP`
2. `EncoderResMLP`
3. `EncoderPathNet`
4. `EncoderSGNN`
5. `GeneTransformerEncoder`

The Transformer design in this spec is therefore not a two-encoder-only idea.
It must remain compatible with three-encoder and four-encoder fusion settings.

## Current Workflow

### Existing `EncoderMLP`

The current `EncoderMLP` takes one bulk sample vector `x` and maps it through a
shared MLP trunk. From the resulting sample-level hidden feature:

1. one head predicts all cell-type-specific latent parameters together
   (`mu_all_types`, `logvar_all_types`)
2. another head predicts cell proportions when `predict_cell_prop=True`
3. the trunk feature is returned as `cell_prop_feature`, which the VAE can
   reuse for:
   - a shared cell-proportion head
   - conditioned-decoder bulk context

This workflow already supports the current VAE training objectives:

1. bulk reconstruction
2. cell proportion prediction
3. cell-type-specific GEP reconstruction

Its main strengths are:

1. simplicity
2. stable integration with the current VAE code
3. efficient training on vectorized bulk inputs

Its main limitation is that the same sample-level feature must implicitly
support all tasks at once. In particular, it does not explicitly separate:

1. a global summary of the mixed bulk sample
2. per-cell-type sample-specific representations

### Existing `GeneTransformerEncoder`

The current `GeneTransformerEncoder` already follows a more structured idea.
Each gene is treated as an input token through:

$$
\text{gene\_kv}_g = \text{gene\_id\_embedding}_g + \text{value\_projector}(x_g)
$$

It then uses a small bank of learned latent queries:

1. one global query when `predict_cell_prop=True`
2. one query per cell type

These queries attend to all genes using cross-attention, producing a compact
set of latent tokens. A Transformer encoder then lets those latent tokens
interact before they are mapped to:

1. `global_out -> cell_prop`
2. `type_out -> mu/logvar`

This is already conceptually close to the desired multitask design. However,
the current implementation still has some limitations:

1. the encoder-internal token roles and task branches are still more implicit
   than ideal
2. the returned feature interface is still optimized around the older
   `cell_prop_feature` convention
3. there is no explicit decoder-side use of per-cell-type context tokens
4. the standalone direct cell-proportion use case is not documented as a
   first-class mode

The detailed internal redesign for these encoder-side issues is described in
the companion spec:

- `2026-08-18-gene-transformer-encoder-internal-improvement-design.md`

## Main Design Idea

The improved Transformer encoder should be treated as a multitask feature
extractor with three distinct outputs:

1. a **global bulk-context summary** for the whole sample
2. a set of **cell-type-specific sample representations**
3. a **cell proportion prediction** head

Formally, for sample $s$ and cell type $c$:

- bulk sample input: $x^{(s)} \in \mathbb{R}^{G}$
- global sample representation: $h_{\mathrm{bulk}}^{(s)}$
- cell-type-specific representation: $h_c^{(s)}$
- cell proportion prediction: $\hat{p}^{(s)}$

These outputs can then be used in two modes:

1. standalone cell-proportion prediction
2. joint VAE training with conditioned decoding

## Why Improve the Current Transformer Workflow

The current project objective is not only to predict cell proportions. It is to
infer:

1. accurate cell proportions
2. informative sample-level context
3. useful cell-type-specific latent structure for SCT-GEP reconstruction

The Transformer encoder is interesting because it matches that structure more
naturally than a plain MLP:

1. a global token can summarize the full mixture for cell proportions
2. per-cell-type tokens can specialize into cell-type-specific sample
   representations
3. cross-attention lets those tokens gather information from all genes while
   keeping runtime closer to linear in gene count

In other words, the Transformer is attractive not merely because it is more
powerful, but because its token structure matches the biological decomposition
goal better.

## System-level Transformer contract

This document assumes the Transformer encoder exposes three core outputs at the
system level:

1. a global bulk-context summary
2. cell-type-specific latent outputs
3. a cell-proportion prediction path

In notation:

- bulk sample input: $x^{(s)} \in \mathbb{R}^{G}$
- global sample representation: $h_{\mathrm{bulk}}^{(s)}$
- cell-type-specific representation: $h_c^{(s)}$
- cell proportion prediction: $\hat{p}^{(s)}$

At the VAE interface level, the relevant outputs are:

1. `cell_prop`
2. `mu_all_types`
3. `logvar_all_types`
4. `bulk_context_feature`
5. `cell_type_context_features`
6. `cell_prop_feature` as a backward-compatible alias of
   `bulk_context_feature`

The detailed internal architecture that produces these outputs is specified in:

- `2026-08-18-gene-transformer-encoder-internal-improvement-design.md`

## How Can I Use It to Predict Cell Proportions Directly

This component can be used as a standalone model for cell-proportion
prediction, without the VAE decoder.

### Standalone Workflow

The direct-use workflow is:

1. input bulk gene expression vector
2. encode the sample into a global bulk-context representation
3. predict cell proportions from that global representation

In this mode, the main prediction path is:

$$
x^{(s)} \rightarrow h_{\mathrm{bulk}}^{(s)} \rightarrow \hat{p}^{(s)}
$$

The cell-type tokens are still useful even in this direct mode because they can
act as structured auxiliary latent slots that help the model organize the bulk
signal by cell type, even if the final loss is only on cell proportions.

### Why This Direct Mode Is Useful

This mode gives a clean benchmark for answering:

1. does the Transformer architecture improve cell-proportion prediction by
   itself?
2. does the query-token formulation capture cross-gene structure better than
   `EncoderMLP`?
3. is any gain coming from the encoder alone, or only from joint VAE training?

This should be treated as an explicit ablation pathway rather than a side
effect of the joint model.

### Recommended Direct-Mode Output Interface

For the standalone mode, the component should expose:

1. predicted cell proportions
2. the global bulk-context feature
3. optional cell-type token features for analysis

The minimum required output is:

- `cell_prop`
- `bulk_context_feature`

Optionally return:

- `cell_type_context_features`

for downstream analysis, visualization, or transfer into the joint model.

## Joint Integration with the Whole VAE Workflow

### Current VAE Integration Pattern

The current VAE expects encoders to provide:

1. `mu_all_types`
2. `logvar_all_types`
3. `mu_mean`
4. `logvar_mean`
5. `cell_prop`
6. `cell_prop_feature`

The current implementation already supports multiple encoders in parallel. When
more than one encoder is active, the VAE can:

1. fuse encoder posteriors across encoders
2. fuse cell-proportion features for a shared prediction head
3. fuse encoder-derived context features for conditioned decoding

The VAE then:

1. resolves cell proportions
2. samples per-cell-type latent vectors
3. decodes them into per-cell-type GEPs
4. mixes them back into bulk reconstruction
5. computes the multitask losses

When the decoder supports conditioning, the VAE also projects
`cell_prop_feature` into a decoder-side bulk-context vector.

### Proposed Joint Integration

The improved Transformer encoder should fit this workflow while making the
roles cleaner.

The joint mode should expose:

1. `bulk_context_feature`
2. `cell_type_context_features`
3. `cell_prop`
4. `mu_all_types`
5. `logvar_all_types`

The VAE should then use these as follows:

1. `cell_prop`:
   - for cell-proportion supervision
   - for downstream cell-type-existence handling
2. `mu_all_types`, `logvar_all_types`:
   - for sampling one latent vector per cell type
3. `bulk_context_feature`:
   - as decoder bulk context for conditioned decoding
4. `cell_type_context_features`:
   - optional extra conditioning signal for a future decoder extension

The initial integration should remain conservative:

1. keep the existing VAE latent sampling and loss structure
2. use `bulk_context_feature` as the main decoder conditioning input
3. defer direct decoder consumption of `cell_type_context_features` to a later
   extension if needed

This keeps the first Transformer integration aligned with the already-defined
conditioned-decoder workflow.

### Keeping `EncoderMLP` and `GeneTransformerEncoder` combinable

The Transformer path must not require replacing `EncoderMLP`. The current VAE
already supports multi-encoder workflows, so the revised design must preserve a
hybrid option where both encoders run in parallel.

Under this hybrid setting:

1. `EncoderMLP` provides a strong and stable vector-space baseline
2. `GeneTransformerEncoder` provides a structured global token and
   cell-type-specific token features
3. the VAE fuses their posteriors using the existing multi-encoder posterior
   fusion logic
4. the cell-proportion pathway can use the existing shared feature fusion
   mechanism
5. conditioned decoding can consume a fused bulk-context feature aggregated
   across both encoders

This hybrid path is important for two reasons:

1. it lets you test whether the Transformer adds value beyond `EncoderMLP`
   instead of forcing a full replacement
2. it reduces the risk of losing the robust behavior of the current MLP branch
   while exploring the richer context modeling of the Transformer branch

Therefore, the Transformer design in this spec must remain compatible with:

1. standalone Transformer-only training
2. Transformer-only VAE integration
3. joint `EncoderMLP` + `GeneTransformerEncoder` VAE integration
4. joint multi-encoder VAE integration with three or more encoders, including
   `EncoderPathNet` and `EncoderSGNN`

### Task-specialized hybrid option

In addition to symmetric multi-encoder fusion, the spec should explicitly
support a task-specialized hybrid mode.

In this mode:

1. `GeneTransformerEncoder` is the preferred branch for predicting cell
   proportions
2. `GeneTransformerEncoder` also provides the main bulk-context summary for the
   conditioned decoder
3. `EncoderMLP` remains the preferred branch for latent parameter prediction
   and downstream cell-type-specific GEP reconstruction

The motivation is straightforward:

1. if the Transformer predicts cell proportions better than `EncoderMLP`, the
   final cell-proportion output should come directly from the Transformer
   instead of being diluted by a fused or averaged head
2. if `EncoderMLP` remains strong for bulk-to-latent mapping, it can focus more
   of its capacity on cell-type-specific representation learning
3. this creates a more purposeful collaboration between the two encoders,
   rather than forcing them to contribute equally to every task

Under this task-specialized hybrid design, the VAE workflow becomes:

1. Transformer branch:
   - predicts `cell_prop`
   - provides `bulk_context_feature`
2. MLP branch:
   - provides `mu_all_types`
   - provides `logvar_all_types`
3. decoder:
   - uses the Transformer-derived bulk context
   - uses the MLP-derived latent variables for cell-type-specific decoding

This gives the model a cleaner division of labor:

1. Transformer branch: infer sample composition
2. MLP branch: encode reconstruction-relevant cell-type latent structure

The first implementation of this task-specialized hybrid should remain
conservative:

1. do not force symmetric posterior fusion between the MLP and Transformer
   branches
2. do not average the Transformer cell-proportion head with the MLP head if the
   goal is to test whether the Transformer is the stronger cell-proportion
   predictor
3. treat this as a dedicated ablation against both:
   - MLP-only
   - Transformer-only
   - symmetric hybrid fusion

This is the cleanest way to test the hypothesis that the Transformer is better
for composition prediction, while the MLP remains useful for latent/GEP
modeling.

### Compatibility with `EncoderPathNet` and `EncoderSGNN`

This spec must also preserve compatibility with other existing encoder families
that already fit the current VAE output contract.

Relevant existing encoders include:

1. `EncoderPathNet`
   - pathway-aware encoder built on top of the MLP encoder pattern
   - already compatible with the current VAE latent and feature interface
2. `EncoderSGNN`
   - graph-based encoder using PPI structure
   - already exposes:
     - `mu_all_types`
     - `logvar_all_types`
     - `mu_mean`
     - `logvar_mean`
     - `cell_prop`
     - `cell_prop_feature`

This means the revised routing and fusion design must not assume there are only
two encoder families in the system. It must support:

1. Transformer + MLP
2. Transformer + PathNet
3. Transformer + SGNN
4. MLP + PathNet + Transformer
5. MLP + PathNet + SGNN + Transformer

For the first implementation, the important requirement is not that every
encoder must become Transformer-aware. The requirement is that the routing and
fusion design remains generic enough to include these encoders without breaking
the current VAE workflow.

## Relationship to the Conditioned Decoder

The conditioned decoder spec already introduced the idea that decoding should
depend on:

1. cell-type identity
2. bulk-context summary
3. per-cell-type latent code

The Transformer encoder strengthens that design because it can provide a more
structured `bulk_context_feature` than `EncoderMLP`.

Under the joint design, the decoder path becomes:

$$
z_c^{(s)}, \; c, \; h_{\mathrm{bulk}}^{(s)}
\rightarrow \mathrm{DecoderConditionalMLP}
\rightarrow \widehat{\mathrm{GEP}}_c^{(s)}
$$

A later extension may also allow:

$$
z_c^{(s)}, \; c, \; h_{\mathrm{bulk}}^{(s)}, \; h_c^{(s)}
\rightarrow \mathrm{DecoderConditionalMLP}
\rightarrow \widehat{\mathrm{GEP}}_c^{(s)}
$$

but that should not be required for the first integration.

## Advantages Compared to `EncoderMLP`

### 1. Better role separation

`EncoderMLP` produces one shared sample feature and asks it to support all
tasks at once. The Transformer naturally separates:

1. a global sample summary
2. cell-type-specific token representations

This is a better fit for the multitask objective.

### 2. Richer global context for cell proportions

The global token is a natural summary for predicting cell proportions, whereas
`EncoderMLP` uses a trunk feature that was not explicitly designed as a global
mixture summary.

### 3. More structured cell-type representations

The per-cell-type query tokens provide a clearer route for learning
cell-type-specific sample information than the fully vectorized MLP head.

### 4. Better compatibility with conditioned decoding

The Transformer can provide a stronger bulk-context feature for
`DecoderConditionalMLP`, which may help reduce collapse in reconstructed
cell-type-specific GEPs.

### 5. More flexible as a standalone model

Unlike `EncoderMLP`, the Transformer design naturally supports both:

1. a direct cell-proportion predictor
2. a joint latent encoder for the VAE

## Tradeoffs Compared to `EncoderMLP`

### 1. Higher complexity

The Transformer has more moving parts:

1. gene embeddings
2. latent queries
3. cross-attention
4. token interaction layers

This makes debugging and ablation more demanding.

### 2. Greater overfitting risk

The Transformer is more expressive, which may be an advantage on large data but
can become a liability on smaller datasets.

### 3. Weaker built-in biological prior than pathway/graph encoders

The Transformer learns flexible gene interactions, but it does not explicitly
encode known biological structure such as pathways or PPI graphs.

### 4. Harder attribution

If performance improves, it may be less obvious whether the gain came from:

1. better cell-proportion prediction
2. better bulk-context summarization
3. better latent structure for decoding

This increases the importance of clean ablations.

## Recommended improvements over the current implementation

### 1. Improve the encoder internals in the companion spec

The detailed internal Transformer refactor now lives in:

- `2026-08-18-gene-transformer-encoder-internal-improvement-design.md`

That companion spec owns the encoder-specific decisions such as:

1. explicit token/query roles
2. task-specific branch split
3. internal output semantics

This document only depends on the resulting system-level outputs.

### 2. Return explicit context outputs

In addition to `cell_prop_feature`, the encoder should explicitly expose:

1. `bulk_context_feature`
2. `cell_type_context_features`

This makes integration with the VAE and conditioned decoder much cleaner.

### 3. Keep first VAE integration conservative

For the first integration:

1. use the Transformer bulk token as the decoder context
2. keep decoder consumption of per-cell-type context out of scope

This limits architectural churn while still testing the main idea.

### 4. Preserve the hybrid multi-encoder path

Do not redesign the Transformer integration in a way that breaks the current
multi-encoder fusion workflow.

The first implementation must keep it possible to:

1. run `EncoderMLP` alone
2. run `GeneTransformerEncoder` alone
3. run both encoders together and compare fused performance against either
   single-encoder baseline

This is important scientifically because it separates three questions:

1. is the Transformer better than `EncoderMLP` on its own?
2. does the Transformer add complementary information to `EncoderMLP`?
3. is any improvement coming from replacement or from combination?

## Config and Interface Expectations

The improved Transformer mode should remain configurable through model fields
such as:

- `transformer_d_model`
- `transformer_nhead`
- `transformer_num_layers`
- `transformer_dim_feedforward`
- `transformer_dropout`

The design should also define whether the component is running in:

1. **standalone mode**
2. **joint VAE mode**
3. **task-specialized hybrid mode**

This may be handled by:

1. a separate training workflow for the standalone predictor
2. a model flag that enables proportion-only output usage
3. a model flag or routing policy that selects:
   - which encoder provides the final `cell_prop`
   - which encoder provides the primary latent posterior
   - which encoder provides the decoder bulk context

The first implementation should prefer minimal disruption to the current config
surface.

### Recommended routing config

The spec is clearer if it names the routing policy explicitly. This lets you
test replacement, fusion, and task specialization without changing the high-
level model code each time.

For long-term compatibility with three-encoder and four-encoder settings, the
routing config should target **encoder aliases**, not only encoder families.

The recommended new model fields are:

```yaml
model:
  encoder_aliases:
    - mlp_main
    - transformer_main
  encoder_output_routing:
    cell_prop_source: fused
    latent_posterior_source: fused
    decoder_context_source: fused
```

Where:

- `cell_prop_source`
  - selects which encoder path provides the final `cell_prop`
- `latent_posterior_source`
  - selects which encoder path provides the primary `mu_all_types` and
    `logvar_all_types`
- `decoder_context_source`
  - selects which encoder path provides the decoder bulk-context feature

For the first implementation, the allowed values should be:

- any active encoder alias, for example:
  - `mlp_main`
  - `transformer_main`
  - `pathway_main`
  - `sgnn_main`
- `fused`

where:

- an encoder alias means use that specific encoder branch directly
- `fused` means use the current VAE multi-encoder fusion pathway

If the first implementation does not yet support configurable aliases in code,
the spec may use fixed family names as a temporary shortcut for two-encoder
experiments. However, the design target should remain alias-based so it scales
cleanly to three encoders or four encoders.

### Routing semantics

This routing contract must stay aligned with the current VAE workflow.

The recommended semantics are:

1. if `cell_prop_source` names a specific encoder alias, use that branch
   prediction as the final `pred_cell_prop`
2. if `cell_prop_source: fused`, use the current shared feature fusion head or
   legacy output averaging path across all active encoders
3. if `latent_posterior_source` names a specific encoder alias, use that
   branch `mu_all_types/logvar_all_types`
4. if `latent_posterior_source: fused`, use the existing posterior fusion logic
   across all active encoders
5. if `decoder_context_source` names a specific encoder alias, build the
   decoder context from that branch feature
6. if `decoder_context_source: fused`, build the decoder context from the
   existing fused encoder feature projection path across all active encoders

This generalization is important because it keeps the routing logic valid for:

1. two encoders
2. three encoders
3. four encoders

without introducing special-case logic for each encoder family.

### Recommended mode presets

The following presets make the experimental modes easier to reproduce and
compare.

**MLP-only baseline**

```yaml
model:
  encoders: ['EncoderMLP']
  encoder_aliases: ['mlp_main']
  encoder_output_routing:
    cell_prop_source: mlp_main
    latent_posterior_source: mlp_main
    decoder_context_source: mlp_main
```

**Transformer-only baseline**

```yaml
model:
  encoders: ['GeneTransformerEncoder']
  encoder_aliases: ['transformer_main']
  encoder_output_routing:
    cell_prop_source: transformer_main
    latent_posterior_source: transformer_main
    decoder_context_source: transformer_main
```

**Symmetric hybrid**

```yaml
model:
  encoders: ['EncoderMLP', 'GeneTransformerEncoder']
  encoder_aliases: ['mlp_main', 'transformer_main']
  encoder_output_routing:
    cell_prop_source: fused
    latent_posterior_source: fused
    decoder_context_source: fused
```

**Task-specialized hybrid**

```yaml
model:
  encoders: ['EncoderMLP', 'GeneTransformerEncoder']
  encoder_aliases: ['mlp_main', 'transformer_main']
  encoder_output_routing:
    cell_prop_source: transformer_main
    latent_posterior_source: mlp_main
    decoder_context_source: transformer_main
```

This last preset directly encodes the main hypothesis of the current design
discussion:

1. the Transformer may be better for cell composition prediction
2. the MLP may remain better or more stable for latent/GEP modeling
3. the Transformer bulk token may be the best decoder conditioning source

**Three-encoder example**

```yaml
model:
  encoders: ['EncoderMLP', 'EncoderPathNet', 'GeneTransformerEncoder']
  encoder_aliases: ['mlp_main', 'pathway_main', 'transformer_main']
  encoder_output_routing:
    cell_prop_source: transformer_main
    latent_posterior_source: fused
    decoder_context_source: fused
```

**Four-encoder example**

```yaml
model:
  encoders:
    ['EncoderMLP', 'EncoderPathNet', 'EncoderSGNN', 'GeneTransformerEncoder']
  encoder_aliases:
    ['mlp_main', 'pathway_main', 'sgnn_main', 'transformer_main']
  encoder_output_routing:
    cell_prop_source: transformer_main
    latent_posterior_source: fused
    decoder_context_source: fused
```

These examples show the intended scaling behavior: the Transformer can be used
as a specialist branch without preventing larger multi-encoder fusion setups.

### Validation rules

The config validation must reject routing choices that do not match the active
encoders.

For example:

1. if `cell_prop_source` names a specific alias, that alias must be present in
   the active encoder list
2. if `latent_posterior_source` names a specific alias, that alias must expose
   valid latent posterior outputs
3. if any source is set to `fused`, then at least two encoders must be active
4. if `decoder_context_source` is not `fused`, the selected encoder must expose
   a valid context feature for the conditioned decoder path
5. `encoder_aliases` must have the same length and order as `encoders`

These checks are important because they make the ablations explicit and prevent
silent fallback behavior.

## Implementation checklist

This section translates the design into a concrete implementation plan. The
goal is to keep the first version small, testable, and compatible with the
current VAE workflow.

### 1. Config changes

Start by adding the minimum config surface needed for routing and validation.

Implementation tasks:

1. add a new `encoder_output_routing` config block under `model`
2. add fields:
   - `cell_prop_source`
   - `latent_posterior_source`
   - `decoder_context_source`
3. support values:
   - encoder aliases
   - `fused`
4. add validation rules that check routing choices against the active encoder
   list
5. preserve backward compatibility by defaulting to the current behavior when
   no routing block is provided

Recommended first default:

```yaml
model:
  encoder_aliases:
    - encoder_0
  encoder_output_routing:
    cell_prop_source: fused
    latent_posterior_source: fused
    decoder_context_source: fused
```

### 2. Transformer encoder interface changes

The detailed internal Transformer refactor is specified in the companion
encoder-focused spec. From the perspective of this system-level document, the
important requirement is the output contract.

Implementation tasks:

1. keep the current `cell_prop` output
2. keep the current `mu_all_types` and `logvar_all_types` outputs
3. expose explicit:
   - `bulk_context_feature`
   - `cell_type_context_features`
4. keep `cell_prop_feature` for backward compatibility in the first
   implementation, mapping it to `bulk_context_feature`

This keeps the new design compatible with current VAE assumptions while making
the Transformer outputs more interpretable.

### 3. VAE routing changes

The main implementation work is in the VAE forward path, where encoder outputs
are currently fused in a symmetric way by default.

Implementation tasks:

1. collect encoder outputs as separate branch-specific objects
2. identify encoder type or encoder role for each active encoder
3. route `pred_cell_prop` according to `cell_prop_source`
4. route `mu_all_types` and `logvar_all_types` according to
   `latent_posterior_source`
5. route decoder context according to `decoder_context_source`
6. preserve the existing `fused` path by reusing:
   - posterior fusion
   - shared feature fusion for `cell_prop`
   - fused context projection for conditioned decoding
7. ensure the routed outputs still feed the existing:
   - effective cell proportion resolution
   - latent sampling
   - reconstruction
   - loss computation

The routing implementation must work for:

1. one encoder
2. two encoders
3. three encoders
4. four encoders

without assuming that only one MLP-like branch and one Transformer branch are
present.

The first version must avoid silent mixing between routing modes. If the config
selects `transformer` for `cell_prop_source`, the final `pred_cell_prop` must
come from the Transformer branch only.

### 4. Task-specialized hybrid support

The task-specialized hybrid is one of the main goals of this spec, so it must
be treated as a first-class supported mode.

Implementation tasks:

1. support the routing preset:

```yaml
model:
  encoders: ['EncoderMLP', 'GeneTransformerEncoder']
  encoder_aliases: ['mlp_main', 'transformer_main']
  encoder_output_routing:
    cell_prop_source: transformer_main
    latent_posterior_source: mlp_main
    decoder_context_source: transformer_main
```

2. verify that:
   - the Transformer branch provides the final cell proportions
   - the MLP branch provides the latent posterior
   - the Transformer branch provides the conditioned-decoder context
3. keep the existing symmetric hybrid pathway available as a separate ablation

This is the minimum implementation needed to test the main hypothesis behind
the hybrid design.

The same routing mechanism must also remain valid when additional encoders such
as `EncoderPathNet` or `EncoderSGNN` are active in the model.

### 5. Conditioned decoder compatibility

The spec already assumes compatibility with `DecoderConditionalMLP`, so the
first implementation must preserve that path.

Implementation tasks:

1. ensure `decoder_context_source` can feed the conditioned decoder path
2. verify that Transformer-derived `bulk_context_feature` can be projected by
   the existing decoder-context projection blocks
3. keep direct use of `cell_type_context_features` out of scope for the first
   implementation

This keeps the first implementation focused on the highest-value context signal
without expanding the decoder contract too early.

### 6. Tests

The implementation must be protected by focused regression tests. The goal is
to validate routing behavior, not only tensor shapes.

Implementation tasks:

1. add config tests for:
   - valid routing combinations
   - invalid routing combinations
   - backward-compatible defaults
2. add Transformer interface tests for:
   - `bulk_context_feature`
   - `cell_type_context_features`
   - direct cell-proportion output
3. add VAE routing tests for:
   - `cell_prop_source` from a named encoder alias
   - `cell_prop_source: fused`
   - `latent_posterior_source` from a named encoder alias
   - `latent_posterior_source: fused`
   - `decoder_context_source` from a named encoder alias
   - `decoder_context_source: fused`
4. add at least one task-specialized hybrid test that verifies the selected
   outputs come from the intended encoder branches
5. add a conditioned-decoder regression test using Transformer-derived decoder
   context
6. add at least one multi-encoder regression test with three encoders
7. add at least one multi-encoder regression test with four encoders if test
   fixtures can support it cheaply

### 7. Recommended implementation order

Implement the feature in the following order:

1. add config fields and validation
2. update `GeneTransformerEncoder` outputs
3. add VAE routing logic for `cell_prop_source`
4. add VAE routing logic for `latent_posterior_source`
5. add VAE routing logic for `decoder_context_source`
6. verify symmetric fusion remains unchanged
7. add tests for standalone, fused, and task-specialized modes

This order reduces the chance of breaking the current pipeline while the new
routing logic is still under construction.

## Testing Plan

### Standalone tests

1. verify direct cell-proportion prediction shape and loss behavior
2. compare proportion metrics against `EncoderMLP`
3. run small-subset overfit tests to confirm optimization sanity

### Joint VAE tests

1. verify VAE forward compatibility with Transformer encoder outputs
2. verify conditioned decoder can consume Transformer bulk context
3. compare:
   - bulk reconstruction
   - cell proportion performance
   - cell-type-specific GEP metrics

### Ablations

Recommended ablation order:

1. `EncoderMLP` baseline
2. standalone Transformer for cell proportions only
3. Transformer-only VAE with current decoder
4. symmetric `EncoderMLP` + `GeneTransformerEncoder` VAE with current decoder
5. task-specialized hybrid:
   - Transformer for `cell_prop`
   - MLP for latent posterior
   - Transformer for bulk decoder context
6. Transformer-only VAE with `DecoderConditionalMLP`
7. symmetric `EncoderMLP` + `GeneTransformerEncoder` VAE with
   `DecoderConditionalMLP`
8. task-specialized hybrid with `DecoderConditionalMLP`

This will isolate whether the benefit comes from:

1. the encoder itself
2. encoder complementarity in the hybrid setting
3. task specialization across encoders
4. the encoder-decoder pairing
5. better bulk-context conditioning

## Recommended Positioning

The Transformer encoder should not immediately replace `EncoderMLP` as the
default production choice.

Instead, it should be positioned as:

1. a **structured experimental encoder**
2. a **standalone cell-proportion prediction model**
3. a **joint VAE encoder candidate** for improving bulk-context summarization
   and cell-type-specific representation learning
4. a **hybrid companion encoder** that can be combined with `EncoderMLP`
   rather than only replacing it
5. a **specialist cell-proportion branch** in a task-specialized hybrid VAE
   design

This is the most scientifically interpretable way to evaluate it without
confounding too many architectural changes at once.

## Summary

The improved Transformer encoder is promising because it matches the structure
of the task better than `EncoderMLP`:

1. one global token for cell proportions and bulk context
2. one token per cell type for cell-type-specific representation learning
3. one shared encoder that can support both direct prediction and joint VAE
   inference

Its main advantage is not merely higher model capacity. Its main advantage is a
cleaner decomposition of the multitask problem.

Its main cost is added complexity and a higher need for careful ablation.

Therefore, the recommended strategy is:

1. benchmark it first as a standalone cell-proportion model
2. then integrate it into the VAE as a bulk-context-aware encoder
3. keep the `EncoderMLP` + `GeneTransformerEncoder` combination available as a
   core ablation
4. explicitly test the task-specialized hybrid where the Transformer leads
   `cell_prop` prediction and the MLP leads latent/GEP modeling
5. test it together with the conditioned decoder rather than replacing the
   whole architecture at once
