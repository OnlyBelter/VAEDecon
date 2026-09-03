# Design: hierarchy-aware repulsion margin

Last updated: 2026-09-03

## Status

This change is implemented in the current working tree.

## Goal

Replace the uniform repulsion hinge margin with a pair-specific margin derived
from the existing cell-type hierarchy codes. The goal is to keep repulsion as
a soft separation signal while letting biologically related subtypes remain
more similar than unrelated cell types.

## Motivation

The current repulsion loss uses a single margin for every cell-type pair:

```text
max(0, cos(mu_i, mu_j) - margin)
```

With the default `margin = 0`, every pair is pushed toward orthogonality in
the same way. This is useful for preventing latent collapse, but it is too
coarse for subtype structure. For example, two B-cell subtypes and a
`B cell` versus `CAF` pair receive the same geometric constraint.

The desired behavior is softer and more biology-aware:

- unrelated cell types remain strongly separated,
- related lineages are allowed higher cosine similarity, and
- sibling subtypes are not forced apart as aggressively as unrelated pairs.

## Current behavior

The existing implementation computes a cosine-similarity matrix for
`mu_types`, applies a scalar hinge margin, and averages the upper triangle.
`gamma` remains the global strength of this auxiliary term.

This means:

- `gamma = 0` disables repulsion,
- a larger `gamma` strengthens separation pressure, and
- the loss has no notion of lineage or subtype relatedness.

## Proposed change

The new design keeps the current hinge-loss form, but replaces the scalar
margin with a pair-specific margin matrix `m_ij`.

First, derive a hierarchy similarity score from the existing binary hierarchy
codes:

```text
s_ij = cosine(h_i, h_j)
```

Then build the allowed cosine margin:

```text
m_ij = base_margin + alpha * s_ij
```

The repulsion loss becomes:

```text
L_rep = mean_{i < j} max(0, cos(mu_i, mu_j) - m_ij)
```

This preserves the current optimization shape while relaxing the constraint
for related pairs.

## Initial parameterization

The first implementation keeps the parameterization fixed in code:

- `base_margin = 0.0`
- `alpha = 0.5`

This choice gives simple, interpretable behavior:

- unrelated pairs get `m_ij = 0.0`,
- partially related pairs get an intermediate margin, and
- sibling subtypes with identical hierarchy codes get `m_ij = 0.5`.

Examples under the current hierarchy table:

- `Non-plasma B cells` vs `Plasma B cells` -> `m_ij = 0.5`
- `Non-plasma B cells` vs `CD4 T` -> `m_ij = 0.25`
- `Non-plasma B cells` vs `CAFs` -> `m_ij = 0.0`

## Scope

This design is intentionally narrow. It changes only the margin used by the
repulsion loss.

The implementation:

- precomputes a hierarchy-derived margin matrix from
  `hierarchical_code_targets`,
- stores it as a model buffer,
- uses it inside `_repulsion_loss()` when the shape matches the current cell
  types, and
- falls back to the old scalar-margin behavior if the buffer is unavailable.

## Non-goals

This first iteration does not:

- add new config knobs for `base_margin` or `alpha`,
- change `gamma` semantics beyond updating its comment to reflect the new
  hierarchy-aware behavior,
- introduce pair-specific loss weights, or
- modify the hierarchy encoding table itself.

## Validation

The implementation is validated with focused unit tests that check:

- the hierarchy-derived margin matrix assigns larger margins to sibling pairs
  than to unrelated pairs, and
- `_repulsion_loss()` uses the hierarchy-specific margin when available and
  falls back to the scalar margin otherwise.

In the current environment, the modified Python files compile and the example
YAML parses successfully. Full `pytest` execution is not available here
because `pytest` is missing from the environment, and importing the full
package also hits missing optional dependencies such as `umap` and `seaborn`.

## Next steps

If the new geometry behaves well in training, the next logical extension is to
expose `base_margin` and `alpha` in config so the hierarchy-aware repulsion
strength can be tuned without code changes.
