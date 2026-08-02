# Selected-sample scGEP comparison figure and merged cell-proportion table

## Goal

Improve the selected-sample scGEP comparison outputs produced by VAEDecon
inference so that they are easier to inspect and reuse downstream.

The changes requested are:

1. Merge the selected-sample true and predicted cell-proportion tables into one
   long-format table.
2. Keep the current all-cell-types scGEP comparison figure.
3. Add a second all-cell-types scGEP comparison figure that only shows selected
   samples whose ground-truth cell proportion for the current cell type is
   greater than or equal to `0.005`.
4. Show the ground-truth cell proportion in the legend for the selected samples
   plotted in the all-cell-types figures and in each per-cell-type selected-
   sample scGEP figure.

## Scope

This design only affects the inference-time result export and figure generation
for the selected-sample scGEP outputs under:

- `test_results/<result_set>/gep/sc_gep/`

It does not change:

- model training
- deconvolution prediction logic
- the existing aligned `selected_samples_true_cell_prop.csv` and
  `selected_samples_predicted_cell_prop.csv` outputs

## Current behavior

During inference, VAEDecon already saves:

- `selected_samples_true_cell_prop.csv`
- `selected_samples_predicted_cell_prop.csv`
- `y_true_vs_y_pred_gep_all_cell_types.png`
- `y_true_vs_y_pred_DeSide_<cell_type>.png`

The selected-sample cell-proportion tables are aligned but split across two
files. The current all-cell-types figure plots the configured visible selected
samples without showing their cell proportions in the legend and without
filtering by cell-type-specific proportion. The current per-cell-type selected-
sample scGEP figures also do not show the selected-sample ground-truth cell
proportions in the legend, and filtered panels can reshuffle sample colors when
some samples drop out.

## Target behavior

### 1. Merged long-format cell-proportion table

In addition to the two existing wide tables, inference should save a merged
long-format table in `sc_gep/`:

- `selected_samples_cell_prop_long.csv`

Each row represents one `(sample_id, cell_type)` pair with both true and
predicted cell proportion values.

Required columns:

- `sample_id`
- `cell_type`
- `true_cell_prop`
- `pred_cell_prop`

This format mirrors the downstream-friendly long-table style used by existing
comparison outputs such as `ccc_all_50samples_16cell_types.csv`.

### 2. Preserve the current all-cell-types figure

The current figure should still be generated:

- `y_true_vs_y_pred_gep_all_cell_types.<format>`

Its current plotting scope stays the same: all selected samples are shown for
each of the 16 cell types.

The only requested visual enhancement for this figure is the legend text:

- legend entries should show the selected sample ID and the ground-truth cell
  proportion for the current cell type

Example legend entry:

- `s_segment_0_2120 (true=0.843)`

### 3. Add a filtered companion figure

Inference should generate a second all-cell-types figure:

- `y_true_vs_y_pred_gep_all_cell_types_true_prop_ge_0p005.<format>`

For each cell-type panel, only selected samples with:

- `ground-truth cell proportion >= 0.005`

should be plotted. The metrics displayed inside that filtered panel should be
recomputed from the filtered subset rather than reused from the unfiltered
points.

This filtering is applied independently per cell type. A selected sample may
appear in some panels and be omitted from others.

The filtered panel must preserve the original visible sample-to-color pairing
from the unfiltered panel for that same cell type. Filtering only removes
samples from the panel; it must not reassign the surviving samples to new
colors. This rule must work for any configured number of visible samples rather
than assuming exactly three.

### 4. Empty-panel behavior

If none of the selected samples satisfy the threshold for a specific cell type,
the panel should still be kept in the grid so the 16-cell-type layout remains
stable. That panel should contain:

- the diagonal reference line
- axis labels/ticks consistent with the existing layout
- a short note such as `No sample with true prop >= 0.005`

This keeps the figure layout comparable across runs.

## Data flow

### Cell-proportion export path

1. Load aligned true and predicted selected-sample cell-proportion tables.
2. Keep saving the two existing wide CSV files unchanged.
3. Convert the aligned tables to long format by:
   - preserving shared sample order
   - iterating over cell types
   - writing one row per `(sample_id, cell_type)` pair
4. Save the merged long table into `sc_gep/selected_samples_cell_prop_long.csv`.

### Figure generation path

1. Reuse the selected-sample cell-proportion table as the metadata source for
   legend text and threshold filtering.
2. For each cell-type panel in the all-cell-types scGEP figure:
   - map each plotted selected sample to its ground-truth cell proportion for
     that cell type
   - render legend labels using that ground-truth proportion
3. For each per-cell-type selected-sample scGEP figure:
   - map each plotted selected sample to its ground-truth cell proportion for
     that cell type
   - render legend labels using that ground-truth proportion
4. For the filtered companion figure:
   - apply the `>= 0.005` rule before plotting points for each selected sample
     within each cell-type panel
   - recompute panel-level metrics from the filtered subset
   - reuse the unfiltered panel's sample-to-color map for any surviving samples
   - keep the same panel order and general style as the current figure

## Implementation structure

### `vaedecon/workflow/inference.py`

Add a small helper to convert aligned selected-sample true/predicted
cell-proportion tables into one long-format merged table and save it under
`sc_gep/`.

This logic should stay close to the existing selected-sample export code so all
selected-sample outputs are produced together.

### `vaedecon/plot/evaluate_result.py`

Extend the all-cell-types scGEP plotting helper so it can:

- accept per-sample legend annotations based on ground-truth cell proportion
- accept a stable per-sample color map built from the original visible sample
  order for each panel
- optionally filter plotted selected samples by a minimum ground-truth
  cell-proportion threshold
- retain stable grid layout when a panel has no eligible selected sample

The selected-sample plotting path should support both modes:

1. unfiltered mode for the current figure
2. filtered mode for the new companion figure

The per-cell-type selected-sample scGEP figures should reuse the same legend
annotation source so their legend labels remain consistent with the all-cell-
types figure.

## Output contract

After the change, `sc_gep/` should contain:

- existing files:
  - `selected_samples_true_cell_prop.csv`
  - `selected_samples_predicted_cell_prop.csv`
  - `y_true_vs_y_pred_gep_all_cell_types.<format>`
  - `y_true_vs_y_pred_DeSide_<cell_type>.<format>`
- new file:
  - `selected_samples_cell_prop_long.csv`
- new companion figure:
  - `y_true_vs_y_pred_gep_all_cell_types_true_prop_ge_0p005.<format>`

Behavioral change for existing figures:

- `y_true_vs_y_pred_gep_all_cell_types.<format>` keeps its current point set but
  shows ground-truth cell proportions in the legend
- `y_true_vs_y_pred_DeSide_<cell_type>.<format>` keeps its current scope but
  also shows ground-truth cell proportions in the legend

## Backward compatibility

- Existing file names and current all-cell-types figure behavior remain valid.
- Downstream code that already consumes the existing selected-sample CSV files
  does not need to change.
- The new merged long table is additive.
- The new filtered figure is additive.

## Testing

Add focused tests for:

1. selected-sample cell-proportion merge to long format:
   - verify expected columns
   - verify one row per `(sample_id, cell_type)`
   - verify true/predicted values stay aligned

2. filtered all-cell-types figure selection logic:
   - verify only samples with `true_cell_prop >= 0.005` are included for a
     given cell type
   - verify legend labels use ground-truth proportions
   - verify surviving filtered samples keep the same colors assigned in the
     unfiltered panel, for any visible sample count
   - verify filtered-panel metrics are recomputed from the filtered subset
   - verify panels with no eligible sample are handled without crashing

3. per-cell-type selected-sample figure legend logic:
   - verify existing `y_true_vs_y_pred_DeSide_<cell_type>` figures use
     ground-truth cell proportion labels in the legend

4. inference output naming:
   - verify the new long table and filtered figure are saved in `sc_gep/`

## Why this design

This design keeps the package behavior easy to understand:

- the current figure stays available for direct comparison with older runs
- the new filtered figure surfaces the most relevant selected-sample views for
  each cell type
- the existing per-cell-type selected-sample scGEP figures become more
  interpretable by showing the true cell proportion in the legend
- the merged long-format table makes downstream summaries and external plotting
  much easier than working from two wide CSV files
