from pathlib import Path

from vaedecon.workflow.inference import _infer_result_set_name


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
