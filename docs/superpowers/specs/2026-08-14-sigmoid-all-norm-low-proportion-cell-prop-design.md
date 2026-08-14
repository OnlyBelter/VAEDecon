# Sigmoid-All-Norm Cell-Proportion Prediction and Low-Proportion-Aware Loss Design

## Goal

Improve cell-proportion prediction accuracy in VAEDecon, especially in the low-proportion regime, by introducing:

1. a new cell-proportion activation mode that predicts **all** cell types independently with sigmoid and then normalizes them to sum to 1
2. an optional low-proportion-aware weighting scheme for supervised cell-proportion loss

This design is motivated by the current Ablation 1 result, where the global prediction metric is already strong but low-abundance cell types are still not predicted well enough.

## Problem Summary

The current `softmax` proportion head predicts all cell types jointly on the simplex. This is mathematically clean, but it tightly couples logits across cell types. In practice, dominant cell types can suppress the gradients available to low-abundance cell types.

VAEDecon already supports a `sigmoid` activation mode, but that implementation is not suitable for the new goal because it:

1. predicts only `n_cell_types - 1` outputs
2. treats cancer as a remainder term
3. supervises only the non-cancer subset

That design is useful for a special prior assumption, but it is not equivalent to:

> predict every cell type independently, including cancer, then normalize all predictions to sum to 1

## Design Overview

### 1. New activation mode: `sigmoid_all_norm`

Add a new `cell_prop_activation_function` option:

- `sigmoid_all_norm`

Behavior:

1. the head outputs one raw logit per cell type
2. apply elementwise sigmoid to every cell type
3. normalize the resulting positive vector to sum to 1

Formally, for raw logits $z \in \mathbb{R}^C$:

$$
a_c = \sigma(z_c / \tau)
$$

$$
p_c = \frac{a_c}{\sum_{j=1}^{C} a_j + \varepsilon}
$$

where:

- $C$ is the number of cell types
- $\tau$ is an optional temperature
- $\varepsilon$ is a small numerical constant

Default behavior for the first implementation:

- use a fixed temperature of `1.0`
- no learnable prevalence prior yet

### Why this may help

Compared with `softmax`, this gives each cell type a more independent pre-normalization activation. That can reduce excessive competition among cell types and may improve low-proportion recovery when several minor cell types co-exist in the same sample.

## Low-Proportion-Aware Loss Weighting

### Motivation

Even with a better activation, the supervised loss can still be dominated by large-fraction cell types. We want an optional way to increase the contribution from low but biologically meaningful true proportions.

### New option

Add an optional weighting mode inside the cell-proportion supervision loss.

Proposed config fields:

```yaml
model:
  cell_prop_loss_type: "l1_kl"
  cell_prop_loss_kl_weight: 0.5
  cell_prop_loss_weighting: "none"  # "none" or "low_prop_inverse"
  cell_prop_loss_low_prop_epsilon: 0.01
  cell_prop_loss_weight_clamp: [1.0, 5.0]
```

### Weighting definition

For each target cell proportion $y_c$, define:

$$
w_c = \mathrm{clip}\left(\frac{1}{y_c + \epsilon_{\text{low}}}, w_{\min}, w_{\max}\right)
$$

Then normalize weights within each sample so the mean weight stays near 1:

$$
\tilde{w}_c = \frac{w_c}{\frac{1}{C}\sum_{j=1}^{C} w_j}
$$

Use these weights inside the supervised cell-proportion loss.

Default first implementation:

- `cell_prop_loss_weighting: "none"`
- optional mode name: `"low_prop_inverse"`
- `cell_prop_loss_low_prop_epsilon: 0.01`
- `cell_prop_loss_weight_clamp: [1.0, 5.0]`

### Why normalize weights

Without normalization, the total loss scale would drift a lot across samples depending on how many small components are present. Keeping the sample-wise mean weight near 1 makes the weighting easier to tune and compare across runs.

## Supervised Loss Under the New Activation

The existing `l1_kl` loss should remain available and become the default first choice for `sigmoid_all_norm`.

For predicted proportions $\hat{p}$ and target proportions $y$:

$$
\mathcal{L}_{\text{cell-prop}} = \sum_c \tilde{w}_c |\hat{p}_c - y_c|
+ \lambda_{\mathrm{KL}} \sum_c \tilde{w}_c y_c (\log y_c - \log \hat{p}_c)
$$

with:

- $\lambda_{\mathrm{KL}} = 0.5$ by default
- $\tilde{w}_c = 1$ when weighting is disabled

Important:

1. for `sigmoid_all_norm`, supervise the **full** vector including cancer
2. do not drop cancer from the supervised target
3. keep numerical stabilization with `clamp_min(eps)` and renormalization before KL

## Config Changes

### Extend `cell_prop_activation_function`

Allowed values become:

- `softplus`
- `softmax`
- `sigmoid`
- `sigmoid_all_norm`

### Add weighting config fields

Under `model`:

```yaml
model:
  cell_prop_loss_weighting: "none"
  cell_prop_loss_low_prop_epsilon: 0.01
  cell_prop_loss_weight_clamp: [1.0, 5.0]
```

Defaults should preserve backward compatibility:

- weighting off by default
- existing configs keep current behavior

## Code Changes

### 1. `vaedecon/models/base/base_utils.py`

Update:

1. `get_cell_prop_head_output_dim(...)`
   - `sigmoid_all_norm` should return `n_cell_types`
2. `build_cell_prop_from_head_output(...)`
   - add a new branch for `sigmoid_all_norm`
   - compute sigmoid over all logits
   - normalize to sum to 1
   - return `(full_cell_prop, None)`

Do **not** modify the existing `sigmoid` semantics, because it is already a meaningful special mode with cancer-as-remainder behavior.

### 2. encoder files

Ensure all cell-proportion heads that depend on `get_cell_prop_head_output_dim(...)` work automatically once the helper is updated:

- `vaedecon/models/nn/mlp.py`
- `vaedecon/models/nn/res_mlp.py`
- `vaedecon/models/nn/transformer.py`
- `vaedecon/models/gnn/ppi_only_embedding.py`
- `vaedecon/models/nn/fused_mlp_gnn.py`
- any shared cell-proportion head path in `vae_model.py`

Because these already defer output dimension logic to the helper, this should remain a small change.

### 3. `vaedecon/models/vae/vae_model.py`

Update supervised cell-proportion loss selection:

1. `softmax`
   - unchanged
2. `softplus`
   - unchanged
3. `sigmoid`
   - unchanged, still supervise non-cancer subset only
4. `sigmoid_all_norm`
   - supervise the full vector including cancer

Also add the optional low-proportion-aware weighting path inside `_cell_prop_supervision_loss(...)`.

Recommended implementation shape:

1. compute `supervised_pred` and `target`
2. compute optional per-cell weights from target
3. apply weights consistently to the L1 and KL terms

### 4. config schema files

Update the model config schema in:

- `vaedecon/configs/default_config.py`

Add:

- new activation choice
- weighting mode and parameters

## First Recommended Experiment

After implementation, the first clean experiment should be:

```yaml
model:
  cell_prop_activation_function: "sigmoid_all_norm"
  cell_prop_loss_type: "l1_kl"
  cell_prop_loss_kl_weight: 0.5
  cell_prop_loss_weighting: "none"
```

That isolates the effect of the new activation without mixing in weighting changes yet.

Then the next run:

```yaml
model:
  cell_prop_activation_function: "sigmoid_all_norm"
  cell_prop_loss_type: "l1_kl"
  cell_prop_loss_kl_weight: 0.5
  cell_prop_loss_weighting: "low_prop_inverse"
  cell_prop_loss_low_prop_epsilon: 0.01
  cell_prop_loss_weight_clamp: [1.0, 5.0]
```

This separation is important so we can tell whether the activation itself or the weighting provides the gain.

## Expected Outcomes

### What may improve

1. better recovery of low-abundance cell types
2. fewer near-zero collapses for minor cell types
3. better presence/absence discrimination without turning existence shift back on

### What may not improve automatically

1. overall CCC may not increase dramatically if it is already high
2. major cell types may stay similar
3. some runs may become slightly flatter unless a temperature is introduced later

### Possible failure mode

Because sigmoid compresses logits to $(0,1)$ before normalization, `sigmoid_all_norm` can produce vectors that are too smooth.

If that happens later, a second-stage refinement can add:

- temperature scaling
- learnable output bias initialized from empirical prevalence

Those are intentionally out of scope for the first implementation.

## Testing Plan

Add or update tests to cover:

1. `sigmoid_all_norm` head output dimension equals `n_cell_types`
2. `build_cell_prop_from_head_output(..., activation_function="sigmoid_all_norm")`
   - returns a tensor with shape `(B, C)`
   - all entries are non-negative
   - each row sums to 1 within tolerance
3. existing `sigmoid` mode remains unchanged
4. supervised loss for `sigmoid_all_norm`
   - uses the full target vector including cancer
5. low-proportion-aware weighting
   - returns normalized weights
   - respects clamp bounds
   - does not change behavior when mode is `"none"`
6. config YAML parsing accepts the new fields

## Scope Boundaries

This change does **not** include:

1. temperature as a user-exposed config parameter
2. learnable prevalence priors
3. threshold-aware evaluation changes
4. new plotting logic
5. changes to the current `sigmoid` meaning

Keeping those out helps make the first implementation interpretable.

## Recommendation

Implement this in two layers but in one code change:

1. add `sigmoid_all_norm`
2. add optional low-proportion-aware weighting

Then test them with two follow-up configs:

1. `sigmoid_all_norm` alone
2. `sigmoid_all_norm + low_prop_inverse`

That is the cleanest next step for improving low-proportion cell-type prediction while preserving backward compatibility.
