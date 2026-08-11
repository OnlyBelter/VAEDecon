from pathlib import Path

import pandas as pd
import pytest

from vaedecon.workflow.inference import _infer_result_set_name
from vaedecon.workflow.inference import _merge_aligned_cell_prop_long_table
from vaedecon.workflow.inference import _validate_input_file_path
from vaedecon.workflow.inference import _build_simutme_path_suggestions
from vaedecon.workflow.inference import _resolve_model_artifact_path
from vaedecon.workflow.inference import _validate_required_model_artifact_path


def test_infer_result_set_name_uses_full_h5ad_stem():
    path = Path(
        "/tmp/simu_bulk_exp_Mixed_N100K_segment_without_filtering_hnscc_log2cpm1p_selected_50.h5ad"
    )
    assert (
        _infer_result_set_name(path)
        == "simu_bulk_exp_Mixed_N100K_segment_without_filtering_hnscc_log2cpm1p_selected_50"
    )


def test_infer_result_set_name_falls_back_for_blank_stem():
    assert _infer_result_set_name("   ") == "test_set"


def test_merge_aligned_cell_prop_long_table_preserves_alignment():
    true_df = pd.DataFrame(
        [[0.1, 0.9], [0.2, 0.8]],
        index=["sample_a", "sample_b"],
        columns=["Cancer Cells", "CD8 T"],
    )
    pred_df = pd.DataFrame(
        [[0.15, 0.85], [0.25, 0.75]],
        index=["sample_a", "sample_b"],
        columns=["Cancer Cells", "CD8 T"],
    )

    merged_long = _merge_aligned_cell_prop_long_table(true_df, pred_df)

    assert merged_long.columns.tolist() == [
        "sample_id",
        "cell_type",
        "true_cell_prop",
        "pred_cell_prop",
    ]
    assert merged_long.shape[0] == 4
    sample_b_cd8 = merged_long.loc[
        (merged_long["sample_id"] == "sample_b") & (merged_long["cell_type"] == "CD8 T")
    ].iloc[0]
    assert sample_b_cd8["true_cell_prop"] == 0.8
    assert sample_b_cd8["pred_cell_prop"] == 0.75


def test_validate_input_file_path_returns_existing_file(tmp_path: Path):
    existing = tmp_path / "good.h5ad"
    existing.write_bytes(b"not-really-h5-but-exists")

    result = _validate_input_file_path(existing, context="test set data file")

    assert result == existing


def test_validate_input_file_path_raises_with_context_and_requested_path(tmp_path: Path):
    missing = tmp_path / "segment_11ds_n_base30_no_filtering_median_gep" / "missing.h5ad"

    with pytest.raises(FileNotFoundError) as exc_info:
        _validate_input_file_path(missing, context="configured test set 'set1'")

    message = str(exc_info.value)
    assert "configured test set 'set1'" in message
    assert str(missing) in message


def test_build_simutme_path_suggestions_lists_nearby_candidates(tmp_path: Path):
    parent = tmp_path / "segment_11ds_n_base30_no_filtering_median_gep"
    parent.mkdir()
    (parent / "simu_gep_Mixed_Test_set1_10Aug_segment_gepsamp-n_neighbors_log2cpm1p.h5ad").write_bytes(b"a")
    (parent / "simu_gep_Mixed_Test_set2_gepsamp-n_neighbors_log2cpm1p.h5ad").write_bytes(b"b")
    (parent / "not_relevant.txt").write_text("x")

    wrong = parent / "simu_gep_Mixed_Test_set1_OLD_12ds_gepsamp-n_neighbors_log2cpm1p.h5ad"
    suggestions = _build_simutme_path_suggestions(wrong, limit=5)

    assert len(suggestions) >= 2
    assert any(str(parent / "simu_gep_Mixed_Test_set1_10Aug_segment_gepsamp-n_neighbors_log2cpm1p.h5ad") in s for s in suggestions)
    assert any(str(parent / "simu_gep_Mixed_Test_set2_gepsamp-n_neighbors_log2cpm1p.h5ad") in s for s in suggestions)


def test_resolve_model_artifact_path_falls_back_to_default_file_in_model_dir(tmp_path: Path):
    model_dir = tmp_path / "final_model"
    model_dir.mkdir()
    expected = model_dir / "input_gene_list.txt"
    expected.write_text("GAPDH\nACTB\n")

    resolved = _resolve_model_artifact_path(
        model_dir=model_dir,
        configured_path=None,
        default_file_name="input_gene_list.txt",
    )

    assert resolved == expected


def test_validate_required_model_artifact_path_raises_clear_error_when_missing(tmp_path: Path):
    model_dir = tmp_path / "final_model"
    model_dir.mkdir()

    with pytest.raises(FileNotFoundError) as exc_info:
        _validate_required_model_artifact_path(
            model_dir=model_dir,
            configured_path=None,
            default_file_name="input_gene_list.txt",
            label="input gene list",
        )

    message = str(exc_info.value)
    assert "input gene list" in message
    assert str(model_dir / "input_gene_list.txt") in message
