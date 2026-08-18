# GeneTransformerEncoder internal improvement design

## Goal

This spec focuses only on improving the internal design of
`GeneTransformerEncoder`.

The first implementation must stay narrow:

1. improve the encoder internals
2. keep the current VAE integration contract compatible
3. avoid redesigning the full multi-encoder routing system
4. avoid changing decoder behavior in this step

The main target is to make the current Transformer encoder more explicit and
more aligned with the multitask objective:

1. predict cell proportions
2. summarize bulk-sample context
3. produce cell-type-specific latent representations

This document is the **encoder-internal** companion to:

- `2026-08-18-transformer-encoder-cell-prop-and-bulk-context-design.md`

That system-level spec covers:

1. standalone usage for direct cell-proportion prediction
2. joint VAE integration
3. hybrid and multi-encoder routing
4. experimental positioning and ablation strategy

This document only covers how `GeneTransformerEncoder` itself should be
refactored internally.

## Why a smaller spec

The previous Transformer spec grew into a broader system-design document that
included:

1. standalone proportion prediction
2. VAE integration
3. hybrid routing
4. multi-encoder fusion across several encoder families

That larger design is still useful as a long-range reference, but it is too
large for the next implementation step.

This spec narrows the scope to one concrete question:

> How should `GeneTransformerEncoder` itself be improved so that its internal
> structure better matches the roles of global sample summary and
> cell-type-specific representation learning?

The broader questions of how to route Transformer outputs through the VAE, how
to combine it with other encoders, and how to position it experimentally are
handled in the companion system-level spec rather than repeated here.

## Current implementation summary

The current `GeneTransformerEncoder` in
`vaedecon/models/nn/transformer.py` works as follows:

1. project each gene scalar value into `d_model`
2. add a learned gene identity embedding
3. create one shared bank of learned latent queries:
   - one global query if `predict_cell_prop=True`
   - one query per cell type
4. run cross-attention from latent queries to gene tokens
5. run a shared Transformer encoder on the latent tokens
6. interpret:
   - the first token as `global_out`
   - the remaining tokens as `type_out`
7. predict:
   - cell proportions from `global_out`
   - `mu/logvar` from `type_out`

This already captures the right high-level intuition, but the implementation is
still more implicit than it needs to be.

## Current limitations

The current encoder has four main limitations.

### 1. Query roles are implicit

The current encoder stores all queries in one parameter:

```python
self.latents = nn.Parameter(torch.randn(num_queries, self.d_model))
```

That means the distinction between:

1. global query
2. cell-type queries

exists only by position convention, not by explicit structure.

This makes the code harder to read and weakens the architectural signal that
the global token and cell-type tokens serve different purposes.

### 2. Task-specific heads are only partially separated

The current encoder already uses:

1. `global_out -> cell_prop`
2. `type_out -> mu/logvar`

but both outputs come directly from the same shared latent-token processing
stack with no task-specific refinement after the shared Transformer.

That makes the roles cleaner than an MLP, but still leaves some unnecessary
interference between:

1. composition prediction
2. cell-type latent inference

### 3. The code does not make the intended semantics visible

The current logic relies on comments such as:

- index 0 is global
- indices 1..K are cell types

This is understandable, but it is not the cleanest form of implementation.
The structure we want the model to learn should be visible in the module
definition itself.

### 4. The output contract is now cleaner than the internal design

The encoder now exposes:

1. `bulk_context_feature`
2. `cell_type_context_features`
3. `cell_prop_feature`

but the internal token organization is still the older implicit form. That
means the public interface is now slightly ahead of the internal architecture.

## Main design decision

The first internal improvement should do two things together:

1. split the query bank into:
   - `global_query`
   - `cell_type_queries`
2. split the post-transformer head logic into:
   - a branch for cell proportion prediction
   - a branch for latent parameter prediction

This is the smallest improvement that gives the encoder a clearer internal
structure without turning it into a whole new model family.

## Design options considered

### Option 1: Query split only

In this option:

1. replace `self.latents` with explicit query parameters
2. keep the current shared downstream processing almost unchanged

Advantages:

1. very small change
2. easy to verify
3. preserves most current behavior

Limitations:

1. only partly fixes the architecture
2. still leaves the global and cell-type tasks sharing the same final token
   refinement path too tightly

### Option 2: Branch split only

In this option:

1. keep the current shared latent bank
2. add separate task-specific branches after the shared Transformer

Advantages:

1. may reduce task interference
2. relatively contained implementation

Limitations:

1. still leaves token roles implicit
2. code readability and architecture semantics remain weaker than needed

### Option 3: Query split + branch split

In this option:

1. make the token roles explicit
2. make the task branches explicit

Advantages:

1. best match to the intended decomposition
2. clearer code
3. cleaner inductive bias
4. still small enough to remain a focused encoder-only change

Tradeoff:

1. slightly larger change than options 1 or 2

### Recommendation

Use **Option 3**.

This gives the first meaningful internal upgrade without forcing broader VAE or
decoder changes.

## Proposed architecture

### 1. Gene token construction

This part remains unchanged in spirit.

For bulk sample $x^{(s)}$, gene token $g$ is:

$$
u_g^{(s)} = E_g + \mathrm{Proj}_{\mathrm{value}}(x_g^{(s)})
$$

where:

1. $E_g$ is the learned gene identity embedding
2. $\mathrm{Proj}_{\mathrm{value}}$ maps one scalar expression value into
   `d_model`

This produces the gene token sequence:

$$
U^{(s)} = [u_1^{(s)}, \dots, u_G^{(s)}]
$$

### 2. Explicit query parameters

Replace the current shared latent bank with:

1. `global_query`
2. `cell_type_queries`

Conceptually:

$$
Q =
[q_{\mathrm{global}}, q_1, q_2, \dots, q_C]
$$

where:

1. $q_{\mathrm{global}}$ is the learned query for sample-level composition and
   bulk context
2. $q_c$ is the learned query for cell type $c$

This makes the intended roles explicit in the code.

### 3. Shared cross-attention and shared token encoder

The shared early Transformer logic remains:

$$
H_0^{(s)} = \mathrm{CrossAttn}(Q, U^{(s)}, U^{(s)})
$$

followed by:

$$
H^{(s)} = \mathrm{TransformerEncoder}(H_0^{(s)})
$$

This shared stage still performs the main information extraction and lets the
global token and cell-type tokens interact.

### 4. Explicit token split

After the shared Transformer, split:

$$
h_{\mathrm{bulk}}^{(s)} = H^{(s)}_{\mathrm{global}}
$$

$$
h_c^{(s)} = H^{(s)}_{c}, \qquad c = 1, \dots, C
$$

These are the two semantic outputs of the shared encoder body:

1. one global bulk token
2. one token per cell type

### 5. Task-specific branch split

After the shared token encoder, add light branch-specific refinement.

#### Cell-proportion branch

Use a small branch operating on the global token:

$$
\tilde{h}_{\mathrm{bulk}}^{(s)} =
f_{\mathrm{prop\_branch}}\left(h_{\mathrm{bulk}}^{(s)}\right)
$$

Then predict:

$$
\hat{p}^{(s)} = g_{\mathrm{prop}}\left(\tilde{h}_{\mathrm{bulk}}^{(s)}\right)
$$

This branch is dedicated to mixture-composition prediction.

#### Latent branch

Use a small branch operating on the cell-type tokens:

$$
\tilde{h}_c^{(s)} = f_{\mathrm{latent\_branch}}\left(h_c^{(s)}\right)
$$

Then predict:

$$
[\mu_c^{(s)}, \log \sigma_c^{2(s)}]
=
g_{\mathrm{latent}}\left(\tilde{h}_c^{(s)}\right)
$$

This branch is dedicated to cell-type-specific latent inference.

### 6. Output interface

The first implementation must preserve compatibility with the current VAE
contract.

Therefore the encoder should continue to return:

1. `cell_prop`
2. `dd_alpha` when applicable
3. `mu_all_types`
4. `logvar_all_types`
5. `mu_mean`
6. `logvar_mean`
7. `cell_prop_feature`

And it must also keep the clearer explicit outputs:

1. `bulk_context_feature`
2. `cell_type_context_features`

For the first implementation:

- `cell_prop_feature` should remain an alias of `bulk_context_feature`

This keeps the VAE integration unchanged while making the encoder internals
cleaner.

## What stays out of scope

The following items are intentionally out of scope for this spec:

1. multi-encoder routing redesign
2. decoder redesign
3. explicit use of `cell_type_context_features` inside the decoder
4. hybrid fusion policy across MLP, PathNet, SGNN, and Transformer branches
5. standalone training workflow redesign

These may be valid next steps later, but this spec is only about improving
`GeneTransformerEncoder` itself.

## Expected advantages

The main expected gains are architectural clarity and cleaner task separation.

### 1. Stronger inductive bias

The model more explicitly encodes the idea that:

1. one token summarizes the whole mixture
2. one token per cell type represents cell-type-specific sample information

### 2. Reduced task interference

Adding light task-specific branches after the shared Transformer can reduce the
amount of conflict between:

1. cell-proportion prediction
2. latent-parameter inference

### 3. Better code readability

The implementation becomes easier to understand because the intended roles are
visible directly in the module definition, instead of only being implied by
token ordering.

### 4. Cleaner platform for later work

This change creates a better foundation for future experiments such as:

1. using Transformer bulk context in conditioned decoding
2. using cell-type token features as additional decoder context
3. comparing branch-specialized Transformer behavior against MLP baselines

## Risks and tradeoffs

This is still a modest change, but it has a few tradeoffs.

### 1. More parameters

The branch split adds a small amount of extra capacity.

This is acceptable because the branch modules can stay lightweight.

### 2. More implementation detail to maintain

The encoder becomes slightly more complex internally, although the public
interface stays almost the same.

### 3. Performance gains are not guaranteed

This change is motivated by cleaner decomposition, not by a guarantee of higher
accuracy. It still needs ablation.

## Implementation plan

### 1. Replace the shared query bank

Implementation tasks:

1. remove `self.latents`
2. add `self.global_query`
3. add `self.cell_type_queries`
4. build the combined query sequence in `forward`

### 2. Add lightweight branch modules

Implementation tasks:

1. add a small refinement module for the global token
2. add a small refinement module for the cell-type tokens
3. keep both branches lightweight, for example:
   - `LayerNorm`
   - `Linear`
   - `GELU`
   - optional dropout

The goal is not to build a second Transformer. The goal is only to give each
task a small private head before the final projection.

### 3. Keep output compatibility

Implementation tasks:

1. preserve all existing required output keys
2. keep `cell_prop_feature = bulk_context_feature`
3. keep tensor shapes unchanged for downstream VAE code

### 4. Keep current configuration surface minimal

For the first implementation, do not add a large new config surface.

At most, add only small optional controls if absolutely necessary, such as:

1. one branch hidden dimension
2. one branch dropout rate

If the current defaults are sufficient, prefer no new config fields in the
first implementation.

## Testing plan

The tests for this change should stay encoder-focused.

### Required tests

1. verify forward output shapes remain unchanged for:
   - `cell_prop`
   - `mu_all_types`
   - `logvar_all_types`
   - `bulk_context_feature`
   - `cell_type_context_features`
2. verify `bulk_context_feature` uses the global-token branch
3. verify `cell_type_context_features` uses the cell-type-token branch
4. verify backward compatibility with existing VAE expectations
5. verify no regression in the current `predict_cell_prop=False` path

### Recommended regression tests

1. compare output tensor shapes before and after the refactor
2. test that the encoder still works with positional encoding enabled
3. test that the encoder still works with each supported cell-proportion
   activation mode

## Recommended implementation order

Implement in this order:

1. refactor queries into explicit `global_query` and `cell_type_queries`
2. keep outputs unchanged and verify shapes
3. add lightweight global and latent branches
4. expose the same outputs through the refined branches
5. run focused encoder and VAE-compatibility tests

This order keeps the refactor easy to debug.

## Summary

The first internal improvement to `GeneTransformerEncoder` should stay small
and explicit:

1. replace the implicit shared query bank with:
   - `global_query`
   - `cell_type_queries`
2. add light task-specific branches after the shared Transformer:
   - one for cell proportions
   - one for latent prediction
3. preserve the current VAE-facing interface

This gives the encoder a clearer internal structure without turning the next
step into a full VAE redesign.
