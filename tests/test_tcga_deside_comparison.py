import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "VAEDecon_example/applications/tcga_deconvolution/predict_tcga.py"
)
if not SCRIPT.is_file():
    pytest.skip("TCGA comparison script is outside this checkout", allow_module_level=True)

spec = importlib.util.spec_from_file_location("tcga_deside_prediction", SCRIPT)
tcga_prediction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tcga_prediction)


def _inputs(tmp_path, monkeypatch):
    result = tmp_path / "result"
    (result / "merged_tpm").mkdir(parents=True)
    pd.DataFrame(
        {
            "Cancer Cells": [0.2, 0.8, 0.3, 0.7],
            "CAFs": [0.1, 0.2, np.nan, 0.4],
        },
        index=["s1", "s2", "s3", "s4"],
    ).to_csv(result / "merged_tpm/predicted_cell_prop.csv")
    deside = tmp_path / "deside.csv"
    pd.DataFrame(
        {
            "sample_id": ["s4", "s2", "s1", "s3", "other"],
            "cancer_type": ["B", "A", "A", "B", "A"],
            "Cancer Cells": [0.7, 0.8, 0.2, 0.3, 0.1],
            "CAFs": [0.5, 0.3, 0.1, 0.25, 0.0],
        }
    ).to_csv(deside, index=False)
    monkeypatch.setattr(tcga_prediction, "DESIDE_PROPORTION_FILE", deside)
    return result, deside


def test_deside_comparison_exports_all_cell_types(tmp_path, monkeypatch):
    result, _ = _inputs(tmp_path, monkeypatch)
    summary = tcga_prediction.compare_cancer_cell_ccc(result)
    metrics = pd.read_csv(summary["cell_type_metrics_file"], index_col="cell_type")

    assert summary["shared_sample_count"] == 4
    assert summary["cell_type_count"] == 2
    assert set(metrics.index) == {"Cancer Cells", "CAFs"}
    assert metrics.loc["Cancer Cells", "ccc"] == pytest.approx(1.0)
    assert metrics.loc["CAFs", "n_samples"] == 3
    assert metrics.loc["CAFs", "rmse"] == pytest.approx(np.sqrt(0.02 / 3))
    output = result / "compare_with_DeSide"
    for stem in (
        "cell_proportion_Cancer_Cells_by_cancer_type",
        "cell_proportion_CAFs_by_cancer_type",
        "cell_type_ccc_summary",
    ):
        for suffix in ("png", "svg"):
            assert (output / f"{stem}.{suffix}").stat().st_size > 0
    assert (output / "cancer_cell_comparison_CCC.png").is_file()
    cancer = pd.read_csv(output / "cancer_cell_comparison_CCC_by_cancer_type.csv")
    assert cancer["ccc"].tolist() == pytest.approx([1.0, 1.0])


def test_deside_comparison_rejects_mismatched_cell_types(tmp_path, monkeypatch):
    result, deside = _inputs(tmp_path, monkeypatch)
    data = pd.read_csv(deside).drop(columns="CAFs")
    data.to_csv(deside, index=False)

    with pytest.raises(ValueError, match="Cell-type columns do not match"):
        tcga_prediction.compare_cancer_cell_ccc(result)
    assert not (result / "compare_with_DeSide").exists()


def test_deside_comparison_rejects_duplicate_sample_ids(tmp_path, monkeypatch):
    result, deside = _inputs(tmp_path, monkeypatch)
    data = pd.read_csv(deside)
    pd.concat([data, data.iloc[[0]]]).to_csv(deside, index=False)

    with pytest.raises(ValueError, match="DeSide sample IDs must be present and unique"):
        tcga_prediction.compare_cancer_cell_ccc(result)
