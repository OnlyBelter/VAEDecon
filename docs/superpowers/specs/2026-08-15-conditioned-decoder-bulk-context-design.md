# Cell-Type-Conditioned Decoder with Bulk-Context Injection Design

## Goal

Reduce within-cell-type GEP collapse in VAEDecon by implementing the two
highest-priority model improvements identified in the collapse diagnosis:

1. redesign the decoder so it is explicitly conditioned on cell type
2. feed sample-level bulk-context features into the decoder

The first implementation should be strong enough to address the current
decoder-identifiability bottleneck, while remaining surgically compatible with
the existing workflow and existing decoder options.

## Problem Summary

The current `DecoderMLP` only receives the sample-specific latent code
`z_c^(s)` for each cell type and sample. It does **not** receive:

1. an explicit cell-type identity signal at decode time
2. a sample-level bulk-context feature that summarizes the full mixed bulk
   input

This makes it easy for the decoder to learn a near-centroid mapping for each
cell type and ignore much of the within-cell-type sample variation that should
be preserved in the reconstructed cell-type-specific GEPs.

The debug-overfit result strengthens this diagnosis:

- cell proportions can be fitted very well on the selected training subset
- but within-cell-type GEP diversity still collapses on that same subset

That pattern is more consistent with a decoder conditioning bottleneck than
with a pure parameter-count bottleneck.

## Design Overview

The first implementation will introduce a new decoder class:

- `DecoderConditionalMLP`

This decoder will keep a shared MLP trunk, but each decoder block will be
conditioned using a FiLM-style transformation driven by:

1. a learned cell-type embedding `e_c`
2. a sample-level bulk-context feature `h_bulk^(s)`

Formally, the target decoding form becomes:

$$
\text{GEP}_{c}^{(s)} = f\left(z_{c}^{(s)}; e_c, h_{\text{bulk}}^{(s)}\right)
$$

and under residual mode:

$$
\text{GEP}_{c}^{(s)} = \text{prototype}_{c} + f\left(z_{c}^{(s)}; e_c, h_{\text{bulk}}^{(s)}\right)
$$

The new decoder is additive to the current system:

- existing `DecoderMLP` behavior remains unchanged
- existing `DecoderResMLP` behavior remains unchanged
- users opt in by selecting the new decoder name in config

## Why FiLM for the first implementation

Among the candidate designs, the first implementation will use a FiLM-style
conditioned shared trunk because it has the best balance of:

1. explicit conditioning strength
2. moderate implementation complexity
3. preservation of parameter sharing across cell types

Compared with concatenating all context only at decoder input, FiLM gives each
decoder block a direct route to change its behavior according to:

- which cell type is being decoded
- which bulk sample the reconstruction came from

Compared with fully separate decoders, it keeps the model smaller and more
interpretable.

## New Decoder Path

### Decoder inputs

`DecoderConditionalMLP` will receive:

1. `z`: flattened latent vectors with shape `(B * C, L)`
2. `cell_type_indices`: flattened cell-type indices with shape `(B * C,)`
3. `bulk_context`: repeated sample-level context vectors with shape
   `(B * C, D_ctx)`

where:

- `B` is batch size
- `C` is number of cell types
- `L` is latent dimension
- `D_ctx` is the projected bulk-context dimension

Here, `cell_type_indices` means the flattened integer cell-type IDs, not the
embedding vectors themselves. Its dtype should be integer / `torch.long`, and
each element should lie in `{0, 1, ..., C-1}`.

For example, if:

- `B = 2`
- `C = 3`

then flattening the per-sample per-cell-type decode order gives:

1. sample 1, cell type 0
2. sample 1, cell type 1
3. sample 1, cell type 2
4. sample 2, cell type 0
5. sample 2, cell type 1
6. sample 2, cell type 2

so:

```python
cell_type_indices = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
```

The actual cell-type embeddings are then looked up **inside** the decoder via
its decoder-side embedding table. In other words:

$$
e_c = \mathrm{Embedding}(\text{cell\_type\_indices})
$$

So the decoder input interface uses symbolic IDs, while the decoder itself
owns the learnable cell-type embedding parameters.

This is intentional because it:

1. keeps the VAE-side interface simpler
2. keeps decoder-side embedding dimensionality configurable inside the decoder
3. lets the conditioned decoder evolve later without changing the upstream VAE
   calling convention

### Conditioning pathway

The decoder will learn:

1. a cell-type embedding table
2. a conditioning MLP that maps `[e_c || h_bulk^(s)]` to FiLM parameters

For each decoder block hidden state `h`, FiLM applies:

$$
\mathrm{FiLM}(h) = (1 + \gamma) \odot h + \beta
$$

where `gamma` and `beta` are learned functions of the conditioning vector.

Using `(1 + gamma)` instead of `gamma` directly keeps the initialization closer
to an identity modulation.

### Output semantics

The new decoder will keep the same output semantics as the current
`DecoderMLP`:

- same gene-level output shape
- same activation family at the output layer
- no change to downstream residual or non-residual reconstruction logic

This is important so that:

- `learn_gep_residual` continues to work without further semantic changes
- existing loss code remains valid

## Bulk-Context Feature Design

The decoder needs a per-sample context vector that summarizes the mixed bulk
input before per-cell-type decoding.

The first implementation will reuse the existing encoder-side
`cell_prop_feature` tensors as the source of sample-level bulk context. This is
the most natural choice because:

1. these features already summarize the bulk sample at the encoder trunk level
2. they are already exposed consistently by the active encoders used in current
   ablations
3. reusing them avoids inventing a second parallel feature-extraction path

### Context fusion across multiple encoders

For the first implementation:

- each encoder feature will be projected to a shared `decoder_context_dim`
- projected features will be averaged across encoders

This keeps the first version simple and matches the current main use case,
which is a single `EncoderMLP`.

No separate gated fusion for decoder context will be added in the first
implementation. If needed later, that can be a follow-up ablation.

## Config Changes

### New decoder option

Extend `model.decoders` so the following value is supported:

- `DecoderConditionalMLP`

### New model fields

Add the following optional fields under `model`:

```yaml
model:
  conditional_decoder_cell_type_emb_dim: 64
  conditional_decoder_context_dim: 256
  conditional_decoder_dropout_rate: 0.1
```

Definitions:

- `conditional_decoder_cell_type_emb_dim`
  - embedding dimension for decoder-side cell-type identity
- `conditional_decoder_context_dim`
  - projected bulk-context dimension used by the conditioned decoder
- `conditional_decoder_dropout_rate`
  - dropout used inside the conditioning MLP / FiLM blocks

Defaults should preserve backward compatibility:

- existing configs keep working unchanged
- the new fields only matter when `decoders: ['DecoderConditionalMLP']`

## Code Changes

### 1. `vaedecon/models/nn/mlp.py`

Add:

1. a reusable FiLM-conditioned dense block for the decoder path
2. a new `DecoderConditionalMLP` class

Behavior:

- shared decoder trunk
- learned cell-type embedding
- FiLM modulation at each block using `[cell_type_embedding || bulk_context]`
- same final output shape as `DecoderMLP`

### 2. `vaedecon/models/nn/__init__.py`

Export `DecoderConditionalMLP` so it can be selected through config.

### 3. `vaedecon/workflow/workflow.py`

Extend decoder construction logic so:

- `"DecoderConditionalMLP"` maps to the new decoder class

No changes to existing decoder-name behavior.

### 4. `vaedecon/configs/default_config.py`

Add the new config fields and validation:

- positive integer checks for embedding/context dims
- dropout range check through existing float constraints

### 5. `vaedecon/models/vae/vae_model.py`

Add a decoder-context pathway in the VAE:

1. collect encoder-side `cell_prop_feature` tensors
2. project them into a shared `conditional_decoder_context_dim`
3. average across encoders
4. repeat the resulting per-sample context across cell types
5. pass `z`, `cell_type_indices`, and `bulk_context` into
   `DecoderConditionalMLP`

Implementation detail:

- detect conditioned-decoder capability via a decoder attribute such as
  `supports_conditioning = True`
- if the decoder does not support conditioning, keep the current decode path
  unchanged

This avoids changing the interface contract for existing decoder classes.

## Non-goals for the first implementation

The following are intentionally out of scope for this first decoder redesign:

1. fully separate decoders for each cell type
2. low-rank residual factorization
3. sample-level contrastive or triplet losses
4. learned prototype correction around the SCT anchor
5. decoder-context gated fusion across multiple encoders

These may still be useful later, but they are not required for the first test
of the decoder-conditioning hypothesis.

## Testing Plan

Add focused tests for:

1. config loading with the new conditioned-decoder fields
2. workflow decoder selection for `DecoderConditionalMLP`
3. forward-shape behavior of `DecoderConditionalMLP`
4. VAE forward compatibility when the decoder supports conditioning
5. regression that `DecoderMLP` behavior remains unchanged

At minimum, verify:

- the conditioned decoder accepts `(z, cell_type_indices, bulk_context)`
- output shapes match existing decoder expectations
- old decoders still run through the previous path

## Expected Outcomes

If the decoder-conditioning hypothesis is correct, the first expected changes
are:

1. lower `recon_vs_recon` CCC within the collapsed cell types
2. smaller collapse gap relative to `true_vs_true`
3. better preservation of within-cell-type variation on the debug-overfit
   subset

The key success criterion is **not** only better average correlation. It is:

> reconstructed same-cell-type samples should stop collapsing toward a single
> centroid when the true same-cell-type samples are more diverse.

## Recommended First Ablation After Implementation

Create the first post-implementation architecture ablation by replacing:

```yaml
model:
  decoders: ['DecoderMLP']
```

with:

```yaml
model:
  decoders: ['DecoderConditionalMLP']
  conditional_decoder_cell_type_emb_dim: 64
  conditional_decoder_context_dim: 256
  conditional_decoder_dropout_rate: 0.1
```

and keep the rest of the current Ablation 5 residual configuration unchanged.

This gives the cleanest before/after test of whether explicit decoder
conditioning reduces the within-cell-type GEP collapse.
