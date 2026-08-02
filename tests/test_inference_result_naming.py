from pathlib import Path

import pandas as pd

from vaedecon.workflow.inference import _infer_result_set_name
from vaedecon.workflow.inference import _merge_aligned_cell_prop_long_table


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
