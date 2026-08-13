# Design: masked matched-sctGEP supervision during training

Last updated: 2026-08-12

## Status
Proposed.

## Goal

Add a new **stage-1 training loss** that directly supervises the inferred
cell-type-specific outputs against the **matched ground-truth sctGEPs** used to
simulate each mixed bulk training sample.

This loss is intended to improve the **sample-conditioned specificity** of the
reconstructed cell-type-specific GEPs, not just their population-level
statistics.

## User requirements

The user clarified four design constraints:

1. Reuse the **same workflow as inference** for retrieving matched ground-truth
   sctGEPs whenever possible, and refactor shared logic when helpful so the
   training and inference paths stay more generalized, consistent, and
   efficient.
2. Each training bulk set has a **matched SCT dataset** and a
   **matched `sample2cell_id` mapping file**.
3. For this workflow, each mixed bulk sample uses **exactly one selected sctGEP per cell type**, so the target should be the matched selected sctGEP directly; **do not average** across multiple SCT cells.
4. The new supervision loss must already support **masking by true cell
   proportion threshold in stage 1**, analogous in spirit to
   `evaluation.cell_prop_threshold`, but applied during training.

Terminology in this design uses **`sctGEP`** consistently.

## Motivation

Current training already supervises:

- reconstructed bulk GEP quality,
- cell-fraction recovery,
- latent-space structure,
- population-level gene mean/std or cross-sample variance constraints.

However, these losses do **not** directly force the model to reconstruct the
exact sample-specific sctGEP that generated a given bulk sample.

Prediction-time evaluation already compares reconstructed cell-type-specific
outputs against matched ground-truth sctGEPs by:

1. reading the bulk sample IDs,
2. using `sample2cell_id_file_path`,
3. querying the matched SCT `.h5ad`,
4. aligning genes,
5. comparing the reconstructed output to the matched truth.

That workflow exists today in:

- `vaedecon/workflow/inference.py`
- `vaedecon/data/datasets.py::find_sct_gep_of_bulk_sample`
- `vaedecon/plot/evaluate_result.py`

This design brings the same idea into training.

## Non-goals

- Change the inference visualization workflow.
- Change simulation semantics.
- Add a stage-2 correlation/CCC-based loss yet.
- Replace existing reconstruction / cell-fraction / latent losses.

Stage 1 is intentionally a **simple masked matched-sctGEP loss**.

## Current behavior

### Prediction-time matched sctGEP retrieval

Inference already recovers matched ground-truth sctGEPs by:

1. filtering the sample-to-cell mapping to selected bulk sample IDs,
2. reading `selected_cell_id` values,
3. loading those cells from the SCT `.h5ad`,
4. aligning them to the bulk gene list,
5. grouping by cell type,
6. writing `sct_gep_<cell_type>_from_<n>_bulksamples.csv`.

Relevant implementation:

- `vaedecon/workflow/inference.py`
- `vaedecon/data/datasets.py::find_sct_gep_of_bulk_sample`
- `vaedecon/plot/evaluate_result.py::plot_single_cell_gep`

### Training-time limitation

Training batches currently expose only:

- `data`
- `labels`

via `DatasetOutput`.

So the model has no in-batch access to:

- sample IDs,
- matched selected SCT cell IDs,
- or ground-truth matched sctGEP tensors.

Therefore, adding the new loss requires a small extension to dataset
preprocessing and cached training tensors.

## Proposed change

### 1. New configuration fields

Add a new optional training-target bundle in `DataConfig`.

#### New config model

```python
class TrainingSetSCTTargetConfig(BaseModel):
    training_set_file_path: str | Path = ""
    training_set_sample2cell_id_file_path: str | Path = ""
    training_sct_gep_file_path: str | Path = ""
```

Add to `DataConfig`:

```python
training_target_sets: Dict[str, TrainingSetSCTTargetConfig] = Field(default_factory=dict)
training_sct_gep_cell_prop_threshold: float = 0.005
```

Add to `LossCoefficient`:

```python
cell_type_sct_gep_weight: float = 0.0
```

#### Semantics

- `training_target_sets` explicitly pairs each simulated bulk training set with:
  - its own bulk file,
  - its own `sample2cell_id` mapping file,
  - its own matched SCT `.h5ad`.
- `training_sct_gep_cell_prop_threshold` is the training-side analog of
  `evaluation.cell_prop_threshold`, but used only for masking the new training
  supervision term.
- `cell_type_sct_gep_weight == 0` means the new feature is disabled and the
  current workflow stays unchanged.

#### Why use explicit target sets?

Existing training config has:

- `data.simu_bulk_file_path`
- `data.sct_file_path`

but not an explicit one-to-one pairing between:

- each bulk training file,
- its matched mapping CSV,
- and its matched SCT dataset.

The new config makes that pairing explicit and avoids fragile guessing.

## 2. Reuse the inference-style matching workflow

Introduce a shared helper that generalizes the existing inference-side matching
logic from `find_sct_gep_of_bulk_sample(...)`.

The preferred implementation is to refactor the shared retrieval/matching path
into one reusable utility instead of duplicating separate training-only and
inference-only code paths. The goal is to keep:

- sample-to-cell filtering logic,
- SCT `.h5ad` querying,
- gene alignment,
- and cell-type target assembly

consistent across both workflows while reducing repeated I/O and maintenance
burden.

### New helper behavior

Add a reusable helper in `vaedecon/data/datasets.py` (exact function name may
vary) that can serve both training and inference use cases:

1. accepts:
   - one bulk dataset,
   - one matched `sample2cell_id` mapping file,
   - one matched SCT `.h5ad`,
   - the bulk gene list,
   - the target cell type list,
   - and optionally the exact list of bulk sample IDs to keep;
2. filters the mapping to the target bulk sample IDs;
3. loads only the required `selected_cell_id` rows from the SCT `.h5ad`;
4. aligns gene order to the bulk gene list;
5. constructs a dense matched target tensor:

```text
true_sct_gep: (n_samples, n_genes, n_cell_types)
```

6. constructs a boolean presence mask:

```text
true_sct_gep_present_mask: (n_samples, n_cell_types)
```

### Suggested helper layering

To keep the training and inference workflows aligned without overloading one
function with too many mode switches, structure the implementation as:

1. one **shared low-level matcher/loader** that:
   - reads the mapping CSV,
   - filters to requested bulk sample IDs,
   - queries the SCT `.h5ad`,
   - aligns genes,
   - assembles matched per-sample per-cell-type targets;
2. one thin **inference wrapper** that:
   - selects the requested subset of sample IDs,
   - writes the current `sct_gep_<cell_type>_from_<n>_bulksamples.csv` outputs,
   - preserves existing plotting/evaluation behavior;
3. one thin **training wrapper** that:
   - iterates over all configured training bulk sets,
   - builds dense `(N, G, C)` tensors,
   - and writes them into the dataset cache.

This keeps the expensive SCT lookup and gene-alignment logic in one place while
letting training and inference keep their own output formats.

### Suggested shared helper contract

The shared helper should return an object or dict equivalent to:

```python
{
    "sample_ids": list[str],                  # ordered bulk sample IDs kept
    "gene_list": list[str],                   # aligned gene order
    "cell_types": list[str],                  # target cell type order
    "true_sct_gep": np.ndarray,               # shape (N, G, C)
    "true_sct_gep_present_mask": np.ndarray,  # shape (N, C), bool
    "selected_sample2cell_id": pd.DataFrame,  # filtered mapping rows
}
```

Order guarantees are important:

- `sample_ids` must match the bulk dataset order used by training/inference.
- `gene_list` must match the processed bulk gene order exactly.
- `cell_types` must match the model output channel order exactly.

The training wrapper can cache `true_sct_gep` and
`true_sct_gep_present_mask`, while the inference wrapper can reshape/export the
same underlying data into the existing CSV layout expected by
`plot_single_cell_gep`.

### Important rule: do not average

For this workflow, each mixed bulk sample uses **one selected sctGEP per cell
type**, so the helper must:

- read exactly the matched `selected_cell_id`,
- map it directly to the SCT `.h5ad`,
- and store that single sctGEP as the ground truth target.

It must **not** average across multiple selected SCT cells in stage 1.

### Validation rules

When `cell_type_sct_gep_weight > 0`, fail fast if any configured training target
set violates:
1. missing `training_set_file_path`,
2. missing `training_set_sample2cell_id_file_path`,
3. missing `training_sct_gep_file_path`,
4. training bulk file is not represented in `data.simu_bulk_file_path`,
5. duplicate mapping rows for the same `(sample_id, cell_type)` when this
   workflow expects exactly one selected sctGEP per cell type,
6. missing required target cell type entries for a sample.

Additionally, the shared helper should explicitly validate:

7. the matched `selected_cell_id` exists in the SCT `.h5ad`,
8. the aligned gene order exactly matches the processed bulk gene list after
   reindex/fill,
9. the final assembled target tensor covers the same ordered sample IDs as the
   corresponding processed bulk training rows.

## 3. Precompute and cache training matched sctGEP targets

Do not query SCT `.h5ad` files inside every training step.

Instead, during `GEPDataset` preprocessing:

1. preprocess the simulated bulk training data as today;
2. for each configured training target set:
   - load the matched `sample2cell_id` mapping,
   - query the matched SCT `.h5ad`,
   - build `true_sct_gep`,
   - build `true_sct_gep_present_mask`;
3. concatenate the results in the same sample order as the processed bulk
   training matrix;
4. cache those arrays beside the other processed dataset outputs.

### Multi-file alignment rule

When `data.simu_bulk_file_path` contains multiple training bulk files, the
assembled `true_sct_gep` tensor must follow the exact same sample ordering as
the final concatenated processed bulk matrix. The safest implementation is:

1. preprocess/load each bulk file,
2. keep its sample ID order,
3. build its matched `true_sct_gep` block from the paired target-set config,
4. concatenate bulk data, labels, and matched targets in the same file order.

This avoids subtle sample misalignment bugs between the input bulk matrix and
the new supervised target tensor.

### New cached arrays

Extend the dataset cache with optional files for:

- `true_sct_gep.npy`
- `true_sct_gep_mask.npy`

Shapes:

- `true_sct_gep`: `(N, G, C)`
- `true_sct_gep_mask`: `(N, C)`

where:

- `N` = number of bulk samples,
- `G` = number of genes in training gene order,
- `C` = number of cell types.

### `true_sct_gep` value space

The cached `true_sct_gep` tensor stores the matched SCT expression values after
gene alignment in the same log-expression convention used by the training
dataset. Concretely:

- the matched SCT rows are loaded as log-space expression from the SCT
  `.h5ad`;
- if dataset-level constant scaling is enabled, `true_sct_gep` is divided by
  the same scaling factor as the bulk inputs.

So the cached tensor is:

- `log2(expression + 1)` when `scaling_by_constant=False`;
- `log2(expression + 1) / scaling_factor` when
  `scaling_by_constant=True`.

This matters for the supervision loss: the matched target tensor is already in
the same scaled log space as the converted prediction, so the loss compares
like with like and does **not** re-transform `true_sct_gep` a second time.

### Why cache?

This follows existing project constraints:

- large `.h5ad` files should be read once,
- repeated reads during training are too expensive,
- training `__getitem__` should stay light.

## 4. Extend dataset outputs

Extend `DatasetOutput` / `GEPDataset.__getitem__()` to optionally return:

```python
DatasetOutput(
    data=x,
    labels=y,
    true_sct_gep=true_sct_gep_i,              # (G, C)
    true_sct_gep_present_mask=true_sct_gep_present_mask_i,    # (C,)
)
```

This keeps the feature additive:

- if the new arrays are absent, old workflows behave the same,
- if the arrays exist and weight > 0, the model can use them.

## 5. Masking rule for stage 1

The new supervision must ignore low-abundance cell types using the **true cell
proportions**, not predicted proportions.

### Threshold parameter

Use:

```python
data.training_sct_gep_cell_prop_threshold
```

Default:

```python
0.005
```

This mirrors the current inference/evaluation threshold semantics, but is
training-specific and should not reuse `evaluation.cell_prop_threshold`.

### Effective mask

For each sample `i` and cell type `c`, define:

```text
mask_present[i, c] = 1 if matched sctGEP target exists else 0
mask_prop[i, c]    = 1 if true_cell_prop[i, c] >= threshold else 0
effective_mask[i, c] = mask_present[i, c] * mask_prop[i, c]
```

This ensures stage 1 only supervises cell types that are both:

- available in the matched target set,
- and abundant enough by ground-truth fraction.

### Implemented masking rule

The implemented helper follows exactly this logic:

```python
active_cell_type_mask = true_sct_gep_present_mask & (
    true_cell_prop >= cell_prop_threshold
)
```

and then broadcasts that `(B, C)` mask across genes:

```python
active_gene_mask = active_cell_type_mask.unsqueeze(1)
```

So the threshold-based masking is a **hard sample-by-cell-type mask**, not a
soft weighting term. Once a cell type is below threshold for a sample, all gene
dimensions for that sample-cell-type pair are excluded from this supervision
term.

## 6. New stage-1 masked sctGEP supervision loss

Add a new loss term to `vaedecon/models/vae/vae_model.py`.

### Inputs

Use:

- predicted `recon_x_all_types_cpm`: `(B, G, C)`
- batch `true_sct_gep`: `(B, G, C)`
- batch `true_sct_gep_present_mask`: `(B, C)`
- batch true cell fractions `y`: `(B, C)`

### Loss definition

Stage 1 uses a simple masked log-space MSE. Let:

- $\hat{X}_{b,g,c}^{\mathrm{cpm}}$ be the reconstructed cell-type-specific GEP
  in non-log CPM space;
- $\hat{X}_{b,g,c}^{\mathrm{log}}$ be the same prediction converted back to the
  scaled log space used for training;
- $X_{b,g,c}^{\mathrm{true}}$ be the cached matched `true_sct_gep` target in
  that same scaled log space;
- $P_{b,c}^{\mathrm{true}}$ be the true cell proportion;
- $\tau$ be `training_sct_gep_cell_prop_threshold`;
- $M_{b,c}^{\mathrm{present}}$ be `true_sct_gep_present_mask`.

The hard cell-type mask is:

$$
M_{b,c}
=
M_{b,c}^{\mathrm{present}}
\cdot
\mathbf{1}\left\{P_{b,c}^{\mathrm{true}} \ge \tau\right\}.
$$

This mask is then broadcast across genes:

$$
\widetilde{M}_{b,g,c} = M_{b,c}.
$$

The prediction is converted to the same scaled log space as the target:

$$
\hat{X}_{b,g,c}^{\mathrm{log}}
=
\frac{\log_2\left(\hat{X}_{b,g,c}^{\mathrm{cpm}} + 1\right)}{s},
$$

where $s$ is the scaling factor used by the dataset pipeline, or $1$ if
constant scaling is disabled.

The per-sample stage-1 masked supervision loss is:

$$
\mathcal{L}_{b}^{\mathrm{sctGEP}}
=
\frac{
\sum_{g,c}
\widetilde{M}_{b,g,c}
\left(
\hat{X}_{b,g,c}^{\mathrm{log}} - X_{b,g,c}^{\mathrm{true}}
\right)^2
}{
\max\left(1,\sum_{g,c}\widetilde{M}_{b,g,c}\right)
}.
$$

In code, the implemented computation is:

```python
pred_log = to_log_space(recon_x_all_types_cpm, scaling_factor)
mask_ct = (true_sct_gep_present_mask & (true_cell_prop >= threshold))
mask = mask_ct.unsqueeze(1).to(pred_log.dtype)           # (B, 1, C)

sq_err = (pred_log - true_sct_gep).pow(2) * mask
denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
cell_type_sct_gep_loss = sq_err.sum(dim=(1, 2)) / denom  # (B,)
```

Two implementation notes are important:

1. `true_sct_gep` is already in the scaled log space, so there is no
   `true_log = to_log_space(true_sct_gep, scaling_factor)` step.
2. The mask shape is `(B, 1, C)`, which broadcasts over genes automatically.
   This is equivalent to expanding it to `(B, G, C)`.

### Loss registration

Add to total loss:

```python
+ lo.cell_type_sct_gep_weight * cell_type_sct_gep_loss
```

Log it in `LossTerms` as a separate scalar mean.

### Backward compatibility behavior

If any of the following are true:

- `cell_type_sct_gep_weight == 0`
- `true_sct_gep` is absent
- `true_sct_gep_present_mask` is absent
- labels are unavailable

then return a zero vector `(B,)` for this term and keep current behavior.

More precisely, in the implemented code the matched `sctGEP` loss runs only
when all of the following are true:

- `cell_type_sct_gep_weight > 0`;
- `true_sct_gep` is present;
- `true_sct_gep_present_mask` is present;
- ground-truth cell proportions are available.

During inference or prediction, if the training-only tensors are absent, this
term is skipped and a zero vector is returned so prediction behavior remains
unchanged.

## 7. File-level design changes

### `vaedecon/configs/default_config.py`

- add `TrainingSetSCTTargetConfig`
- add `DataConfig.training_target_sets`
- add `DataConfig.training_sct_gep_cell_prop_threshold`
- add `LossCoefficient.cell_type_sct_gep_weight`
- add validation for the new fields when the loss weight is enabled

### `vaedecon/configs/example_config.yaml`

Add an example block showing:

```yaml
data:
  training_sct_gep_cell_prop_threshold: 0.005
  training_target_sets:
    train_set_1:
      training_set_file_path: "./datasets/train_set_1.h5ad"
      training_set_sample2cell_id_file_path: "./datasets/train_set_1_sampled_sc_cell_id.csv"
      training_sct_gep_file_path: "./datasets/train_set_1_sct.h5ad"
```

and under `loss_coefficient`:

```yaml
  cell_type_sct_gep_weight: 0.0
```

### `vaedecon/data/datasets.py`

- add shared matched-sctGEP loading helper
- split the helper into shared core + thin training/inference wrappers if that
  makes the implementation clearer
- add cache file support for `true_sct_gep` and `true_sct_gep_mask`
- extend preprocessing to build those targets
- extend `__getitem__()` to emit them

### `vaedecon/workflow/train.py`

- pass new target-set config into dataset construction
- ensure all training bulk files and matched target sets are aligned

### `vaedecon/models/vae/vae_model.py`

- add `cell_type_sct_gep` field to `LossTerms`
- add `_cell_type_sct_gep_loss(...)`
- integrate the new term into total loss

## 8. Testing plan

Add tests for:

1. config parsing and validation:
   - new training target set fields parse,
   - threshold parses,
   - loss weight default is backward compatible,
   - invalid target-set coverage fails clearly when weight > 0.

2. dataset target construction:
   - matched `sample2cell_id` rows are filtered to training sample IDs,
   - one selected `selected_cell_id` per `(sample_id, cell_type)` is used,
   - genes align to the bulk gene list,
   - cached `true_sct_gep` has expected shape `(N, G, C)`.

3. masking behavior:
   - cell types below `training_sct_gep_cell_prop_threshold` are masked out,
   - cell types above threshold remain active,
   - missing matched targets are masked regardless of proportions.

4. VAE loss behavior:
   - loss returns zeros when weight is 0,
   - loss returns zeros when masks are empty,
   - masked log-MSE matches expected tensor math on a tiny example.

## 9. Backward compatibility

- Existing configs remain unchanged when `cell_type_sct_gep_weight == 0`.
- Existing datasets without training target sets keep working.
- Existing training runs should not pay any additional preprocessing or memory
  cost unless the feature is enabled.

## 10. Open questions intentionally deferred to later stages

These are explicitly out of scope for stage 1:

1. Using CCC/correlation instead of MSE.
2. Weighting by cell type or inverse prevalence.
3. Multi-SCT-per-cell-type averaging workflows.
4. Using predicted cell proportions for masking.

Stage 1 should stay simple:

- exact matched sctGEP targets,
- true-proportion masking,
- masked log-MSE supervision.
