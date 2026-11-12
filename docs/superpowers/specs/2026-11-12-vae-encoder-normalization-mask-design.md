# VAE encoder normalization mask

## Goal

Allow per-stage control of LayerNorm in the VAE's `EncoderMLP` and
`EncoderResMLP`, using an input-plus-hidden-stage mask analogous to
`deside_normalization_layer`. Preserve existing encoder behavior when the new
setting is omitted.

## Configuration

Add `model.encoder_normalization_layer`, a list containing only `0` and `1`.
Its required length is `len(model.encoder_hidden_dims) + 1`:

- The first entry controls LayerNorm on the flattened raw encoder input.
- Each following entry controls normalization for the corresponding hidden
  stage.

When omitted, resolve it to `[0] + [1] * len(encoder_hidden_dims)`. This keeps
raw-input normalization disabled and retains all currently enabled hidden
LayerNorm modules. For the six hidden layers in the DeSide reproduction
configuration, the default is `[0, 1, 1, 1, 1, 1, 1]`.

Reject masks with invalid values or a length that does not match the encoder
dimensions. The new field is independent of `deside_normalization_layer`;
changing it must not affect the DeSide predictor or either decoder.

## Encoder behavior

For `EncoderMLP`, apply the first mask entry before the first projection and
each hidden-stage entry after that stage's `Linear`, before `GELU` and
dropout. A disabled normalization slot should be an identity operation.

For `EncoderResMLP`, apply the first entry to the flattened raw input. Each
hidden-stage entry controls all LayerNorm modules in that stage: the projection
LayerNorm and both LayerNorms inside its residual block. Linear, activation,
dropout, and residual-addition order otherwise remain unchanged.

The mask controls LayerNorm placement only. It does not add BatchNorm or
change normalization axes: each enabled LayerNorm operates on features within
each sample.

## Compatibility

Configurations without `encoder_normalization_layer` retain the existing
architecture and checkpoint parameter names. With the default mask, no new
trainable normalization parameters are introduced. Explicit masks that disable
existing normalization layers define a different architecture and require a
checkpoint trained with that same mask.

Add the explicit preserving-default mask to the packaged example configuration.

## Tests

- Verify omitted masks resolve to input normalization off and all existing
  hidden normalization on for both encoder classes.
- Verify enabled input normalization is applied before the first linear
  projection.
- Verify each hidden-stage bit independently controls the intended LayerNorm
  modules, including all three normalization modules in each `EncoderResMLP`
  stage.
- Verify invalid mask values and lengths fail with actionable validation
  errors.
- Verify default-mask state-dict keys and forward behavior remain compatible
  with the pre-mask encoders.
- Parse the packaged example configuration and the DeSide-reproduction
  example with the explicit preserving-default mask.
