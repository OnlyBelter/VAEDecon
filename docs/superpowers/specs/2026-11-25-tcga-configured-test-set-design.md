# Configured TCGA test sets

Add TCGA inference to the package's configured test-set workflow so that a
training run can predict TCGA expression and, when reference files are
available, compare predictions with DeSide and CPE. Simulated test-set
behavior remains unchanged.

## Configuration

A named test set opts into this behavior with `dataset_type: tcga`. Its
`test_set_file_path` points to a sample-by-gene CSV of linear TPM values.
The first column contains sample IDs, and the remaining columns contain gene
symbols. The default dataset type remains `test`, so existing configurations
and their log-expression CSV and H5AD inputs retain their current behavior.

For example:

```yaml
data:
  test_sets:
    TCGA:
      dataset_type: tcga
      test_set_file_path: './datasets/TCGA/tpm/merged_tpm.csv'
      tcga_comparisons:
        deside_proportion_file_path: './datasets/predicted_cell_proportion_by_DeSide/Table_S3.csv'
        cancer_purity_file_path: './datasets/cancer_purity.csv'
        cancer_type_mapping_file_path: './datasets/TCGA/tpm/tcga_sample_id2cancer_type.csv'
        deside_cpe_ccc_file_path: './applications/DeSide/cancer_cell_vs_cpe_ccc_diff_DNN_models.csv'
```

The comparison block and each of its paths are optional. A TCGA test set does
not require `sct_gep_file_path` or
`test_set_sample2cell_id_file_path`; it must not inherit either from another
test set or the legacy top-level settings. Reject those SCT fields on a TCGA
entry to avoid implying that SCT-based evaluation will run.

Resolve relative paths consistently with existing test-set paths. Keep the
package independent of the example project's directory structure and data.

## Prediction

Dispatch the TCGA entry from `predict_configured_test_sets` after the
appropriate model stage has finished. The same dispatch works for configured
prediction-only runs. Preserve the existing checkpoint selection and staged
test-result reuse rules.

Read the CSV in bounded chunks. Validate sample and gene identifiers, reject
duplicate IDs and genes, and reject negative or nonfinite TPM values. Use
the saved model gene order and ignore extra genes. Reject inputs without
shared model genes.

For a `staged_tpm_v1` checkpoint, select present model genes, normalize
their linear TPM values to one million per sample, apply `log2(TPM + 1)`,
insert zeros for missing model genes, and apply the model's saved constant
scaling when enabled. Exclude samples with zero selected-gene expression
and record their IDs and reasons. Reject a checkpoint explicitly marked
incompatible with this normalization instead of silently producing
predictions.

For older checkpoints without the staged input contract, retain the
standalone TCGA script's legacy preprocessing and missing-gene behavior;
do not impose the staged contract on them. Predictor-only and full-model
paths must both be covered. Tests must establish parity with the standalone
script on representative inputs rather than inferring the input space from
values.

Predictor-only checkpoints export cell proportions. Full VAE checkpoints
also export sample-specific purified cell-type GEPs in the same format as
the standalone TCGA workflow, regardless of simulated-test visualization
settings. Do not perform simulated ground-truth cell-proportion or SCT GEP
comparisons for TCGA samples. The full-model GEP output can be large; note
this in the user documentation.

## Outputs and comparisons

Keep existing test-set result naming by input file stem. Place TCGA outputs
in a dedicated folder below that result, for example:

```text
test_results/merged_tpm/tcga/
  predicted_cell_prop.csv
  excluded_samples.csv
  run_manifest.json
  gep/
  compare_with_DeSide/
  compare_with_CPE/
```

The GEP directory is produced only for full models. The manifest records
the input, checkpoint, model kind, preprocessing contract, sample counts,
excluded samples, missing model genes, generated outputs, and the status
of each comparison. Respect the package's existing output safety and
explicit overwrite policy; never silently reuse or overwrite a nonempty
TCGA output directory.

Move the reusable comparison logic into the package with explicit input
paths; the package must not import the example script. Preserve the
standalone script's current comparison outputs and behavior. DeSide
comparison runs when its proportion file is available. CPE comparison
runs when the cancer purity, cancer-type mapping, and precomputed DeSide
CPE CCC files are all available. Each comparison is independent of the
other.

When a required comparison path is omitted or the file is unavailable,
skip that comparison, log the missing prerequisites, and record the
reason in the manifest. Predictions and any full-model GEPs remain
available. If files exist but contain malformed or inconsistent data,
report the comparison failure in the manifest and log it without
discarding completed predictions or preventing the other comparison
from running.

## Verification

Add regression tests for configuration parsing, mixed simulated and TCGA
test sets, automatic post-stage dispatch, and prediction-only dispatch.
Verify the following:

- Raw TPM preprocessing matches the standalone TCGA workflow for both
  predictor-only and full-model staged checkpoints.
- Missing genes, extra genes, zero-expression samples, invalid values,
  duplicate IDs and genes, and incompatible checkpoints are handled as
  specified.
- Full models export cell-type GEPs; predictor-only models do not.
- No TCGA test set inherits simulated SCT inputs or generates comparisons
  against unavailable simulated ground truth.
- DeSide and CPE comparisons run independently with complete inputs and
  skip cleanly with missing inputs; malformed references are reported.
- Result directories remain isolated, existing outputs are protected, and
  existing simulated test-set behavior remains unchanged.

Update package configuration and usage documentation with the input
format, optional reference paths, output layout, and automatic post-training
behavior. Include the corresponding TCGA entry in the user's example
configuration without changing its existing simulated entries.
