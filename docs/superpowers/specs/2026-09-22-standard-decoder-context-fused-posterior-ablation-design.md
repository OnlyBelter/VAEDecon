# Standard Decoder with Context-Fused Posterior Ablation Design

## Goal

Add an ablation path that can be selected by configuring:

```yaml
decoders: ["DecoderMLP"]
```

The ablation is intended to test whether the cell-type-conditioned decoder
provides value beyond a standard decoder. It will remove decoder-side
cell-type embeddings, conditioning vectors, and FiLM modulation while
retaining the DeSide branch for cell-proportion prediction and bulk-context
features.

The ablation will concatenate the raw `EncoderMLP` feature with the projected
DeSide bulk context before computing the latent posterior parameters.

## Current-Dimension Verification

For the reference configuration:

```yaml
encoder_hidden_dims: [512, 2048, 2048, 2048, 2048, 512]
fusion_hidden_dims: [512, 256]
conditional_decoder_context_dim: 128
```

The raw `EncoderMLP` feature is the output of the final encoder body layer:

```text
EncoderMLP feature: B x 512
```

With the reference configuration's single `EncoderMLP`, the
`fusion_hidden_dims: [512, 256]` setting is not used by the active encoder or
by the proposed posterior-fusion path. It is associated with other
fusion-capable model components and does not change the final feature emitted
by `EncoderMLP`. Therefore, the `256` value is not the raw `EncoderMLP`
output dimension.

The DeSide bulk context will be projected to:

```text
DeSide context: B x 128
```

The ablation posterior input is therefore:

```text
B x 512 || B x 128 = B x 640
```

No `512 -> 256` projection will be inserted.

## Design Overview

### Conditional decoder path

When `DecoderConditionalMLP` is selected, the existing behavior remains
unchanged:

```text
EncoderMLP
  -> latent posterior
  -> z

DeSide
  -> cell proportions
  -> bulk context

z + cell-type embedding + bulk context
  -> DecoderConditionalMLP
  -> FiLM-conditioned cell-type GEPs
```

This path continues to use the decoder-side cell-type embedding, projected
conditioning vector, and FiLM blocks.

### Standard decoder ablation path

When `DecoderMLP` is selected, the model will use:

```text
Bulk GEP x: B x G
  |
  +-> EncoderMLP body -> encoder feature: B x 512
  |
  +-> DeSide predictor -> bulk context projector -> B x 128

Concatenate:
  B x 512 || B x 128 -> B x 640

Posterior head:
  B x 640 -> B x (C x L x 2)
           -> mu/logvar: B x L x C
           -> sampled z: B x L x C

Standard DecoderMLP:
  z -> cell-type GEPs: B x G x C

DeSide proportions:
  B x C

Cell-type GEPs + DeSide proportions
  -> reconstructed bulk GEP: B x G
```

For the reference configuration:

```text
B = 128
G = 9028
C = 16
L = 48

posterior head output = 16 x 48 x 2 = 1536
```

The posterior tensors retain the existing public shapes:

```text
mu_all_types:     B x 48 x 16
logvar_all_types: B x 48 x 16
z_types:          B x 48 x 16
```

The standard decoder continues to receive only flattened latent vectors:

```text
z_types_flat: (B x C) x L
```

It will not receive cell-type indices, a decoder conditioning vector, or FiLM
parameters.

## Posterior-Fusion Implementation

The existing `EncoderMLP` computes latent posterior heads from its final
512-dimensional feature. For this ablation, those encoder-local posterior
outputs will be bypassed when routing the latent posterior.

The VAE will construct a dedicated posterior head for the standard-decoder
ablation:

```text
fused_posterior_input: B x 640
fused_posterior_head:  Linear(640, C x L x 2)
```

The output will be reshaped and split using the same convention as the
existing encoder:

```text
B x (C x 2L)
-> B x C x 2L
-> mu_raw, logvar_raw: B x C x L
-> mu_all_types, logvar_all_types: B x L x C
```

The existing numerical protections remain active:

- clamp `logvar` to the package-level bounds
  `LOGVAR_CLAMP_MIN = -15` and `LOGVAR_CLAMP_MAX = 15`
- apply cell-type existence shifts, when configured
- use the existing Gaussian reparameterization function
- preserve the existing downstream KL and reconstruction interfaces

The standard-decoder posterior fusion source will use the selected encoder
feature and the routed DeSide bulk context. Multi-encoder averaging or gated
fusion is outside the scope of this ablation unless already required by the
existing routing configuration.

## Context Projection

The DeSide predictor currently exposes its terminal summary as
`bulk_context_feature`. For the standard-decoder ablation, this feature will
be projected to `conditional_decoder_context_dim`:

```text
DeSide bulk_context_feature: B x D_deside
Context projector:           D_deside -> 128
Projected context:           B x 128
```

The projection should use the existing context-projector building pattern so
that the standard and conditional decoder experiments use the same context
representation:

```text
Linear -> LayerNorm -> GELU -> Dropout
```

The context is fused once per sample, before latent sampling. It is not
repeated over cell types until the latent tensors are decoded.

## Removed Components in the Ablation

The following components are not used by the `DecoderMLP` path:

1. decoder-side cell-type embedding lookup
2. concatenation of cell-type embedding and bulk context
3. decoder conditioning vector `q`
4. FiLM networks
5. FiLM scale and shift operations
6. `cell_type_indices` passed to the decoder
7. `bulk_context` passed to the decoder

The DeSide cell-proportion output is retained because it is still required to
mix the decoded cell-type GEPs into a reconstructed bulk GEP.

## Decoder Output Compatibility

`DecoderMLP` itself will retain its existing input and output contract:

```text
input:  (B x C) x L
output: (B x C) x G
```

The downstream reshaping, residual reconstruction, CPM conversion, and
proportion-weighted mixing remain unchanged. This keeps the ablation focused
on the posterior/context fusion and removes conditional decoder operations
without changing the reconstruction target or loss interfaces.

## Configuration Behavior

The decoder class selected by `model.decoders` determines the route:

| Decoder selection | Posterior input | Decoder inputs | FiLM |
| --- | --- | --- | --- |
| `DecoderConditionalMLP` | Existing encoder posterior | `z`, cell-type IDs, bulk context | Enabled |
| `DecoderMLP` | Encoder feature + projected DeSide context | `z` only | Disabled |

The reference configuration can activate the ablation by changing:

```yaml
decoders: ["DecoderMLP"]
```

No change to `DecoderConditionalMLP` behavior is expected when the conditional
decoder remains selected.

## Testing Plan

### Unit tests

Add tests that verify:

1. The raw `EncoderMLP` feature has the expected final dimension.
2. The context-fused posterior input has shape `B x 640` for the reference
   dimensions.
3. The fused posterior head returns `mu_all_types` and `logvar_all_types`
   with shape `B x 48 x 16`.
4. The standard decoder path calls `DecoderMLP` with latent vectors only.
5. The standard decoder path does not construct or pass cell-type indices.
6. The standard decoder path does not invoke FiLM conditioning.
7. The conditional decoder path retains its current required inputs and
   output shape.
8. The final reconstructed bulk GEP retains shape `B x G`.

### Regression tests

Run the existing VAE, decoder, configuration, cell-proportion, and numerical
stability tests. In particular, verify that:

- `DecoderMLP` output shape behavior is unchanged
- `DecoderConditionalMLP` still requires both conditioning inputs
- cell-proportion routing remains unchanged
- residual-mode output semantics remain unchanged

## Non-Goals

This change will not:

1. change the `EncoderMLP` body dimensions
2. reinterpret `fusion_hidden_dims` as the EncoderMLP output dimension
3. add a `512 -> 256` projection
4. change the standard `DecoderMLP` implementation
5. remove DeSide cell-proportion prediction
6. alter the conditional decoder architecture
7. introduce a new decoder class
8. change reconstruction losses or evaluation metrics

## Acceptance Criteria

The implementation is complete when:

1. Selecting `DecoderMLP` produces the `B x 640` posterior fusion path.
2. The resulting posterior tensors have the existing expected shapes.
3. `DecoderMLP` receives only sampled latent vectors.
4. Cell-type embeddings and FiLM are absent from the standard decoder path.
5. Selecting `DecoderConditionalMLP` preserves existing behavior.
6. The focused tests and relevant regression tests pass.
