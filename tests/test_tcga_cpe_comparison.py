import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "VAEDecon_example/applications/tcga_deconvolution/predict_tcga.py"
)
spec = importlib.util.spec_from_file_location("tcga_prediction", SCRIPT)
tcga_prediction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tcga_prediction)


def _inputs(tmp_path, monkeypatch):
    result = tmp_path / "result"
    (result / "merged_tpm").mkdir(parents=True)
    pd.DataFrame(
        {"Cancer Cells": [0.2, 0.8, 0.5, 0.4]},
        index=["s1", "s2", "s3", "s4"],
    ).to_csv(result / "merged_tpm/predicted_cell_prop.csv")
    mapping = tmp_path / "mapping.csv"
    pd.DataFrame(
        {"cancer_type": ["ACC", "ACC", "ACC", "PAAD"]},
        index=["s1", "s2", "s3", "s4"],
    ).to_csv(mapping)
    cpe = tmp_path / "cpe.csv"
    pd.DataFrame(
        {
            "Sample ID": ["s1", "s2", "s3"],
            "Cancer type": ["ACC", "ACC", "ACC"],
            "CPE": [0.2, 0.8, float("nan")],
        }
    ).to_csv(cpe, index=False)
    deside = tmp_path / "deside.csv"
    pd.DataFrame(
        [[0.1, 0.2, 0.3, 0.4, 0.5]],
        index=["ACC"],
        columns=tcga_prediction.DESIDE_CPE_COLUMNS,
    ).to_csv(deside)
    monkeypatch.setattr(tcga_prediction, "CPE_FILE", cpe)
    monkeypatch.setattr(tcga_prediction, "TCGA_CANCER_TYPES_FILE", mapping)
    monkeypatch.setattr(tcga_prediction, "DESIDE_CPE_CCC_FILE", deside)
    return result, cpe


def test_cpe_comparison_matches_samples_and_preserves_missing_types(tmp_path, monkeypatch):
    result, _ = _inputs(tmp_path, monkeypatch)
    summary = tcga_prediction.compare_cancer_cell_cpe(result)
    metrics = pd.read_csv(summary["metrics_file"], index_col="cancer_type")

    assert summary["matched_sample_count"] == 3
    assert summary["valid_pair_count"] == 2
    assert summary["cancer_types_without_cpe_pairs"] == ["PAAD"]
    assert metrics.loc["ACC", "n_excluded_nonfinite"] == 1
    assert metrics.loc["ACC", "vaedecon_ccc"] == pytest.approx(1.0)
    assert metrics.loc["ACC", "deside_ccc_mean"] == pytest.approx(0.3)
    assert metrics.loc["PAAD", "n_matched"] == 0
    assert pd.isna(metrics.loc["PAAD", "vaedecon_ccc"])
    svg = result / "compare_with_CPE/cancer_cell_cpe_ccc_comparison.svg"
    assert svg.is_file()
    assert "PAAD" not in svg.read_text(encoding="utf-8")


def test_cpe_comparison_rejects_duplicate_sample_ids(tmp_path, monkeypatch):
    result, cpe = _inputs(tmp_path, monkeypatch)
    purity = pd.read_csv(cpe)
    pd.concat([purity, purity.iloc[[0]]]).to_csv(cpe, index=False)

    with pytest.raises(ValueError, match="CPE sample IDs must be present and unique"):
        tcga_prediction.compare_cancer_cell_cpe(result)


def test_cpe_figure_orders_cancer_types_by_descending_vaedecon_ccc(tmp_path, monkeypatch):
    result, cpe = _inputs(tmp_path, monkeypatch)
    prediction_path = result / "merged_tpm/predicted_cell_prop.csv"
    predictions = pd.read_csv(prediction_path, index_col=0)
    predictions.loc["s5", "Cancer Cells"] = 0.1
    predictions.loc["s6", "Cancer Cells"] = 0.9
    predictions.to_csv(prediction_path)
    mapping = pd.read_csv(tcga_prediction.TCGA_CANCER_TYPES_FILE, index_col=0)
    mapping.loc["s5", "cancer_type"] = "AAA"
    mapping.loc["s6", "cancer_type"] = "AAA"
    mapping.to_csv(tcga_prediction.TCGA_CANCER_TYPES_FILE)
    purity = pd.read_csv(cpe)
    purity = pd.concat(
        [purity, pd.DataFrame({"Sample ID": ["s5", "s6"], "Cancer type": ["AAA", "AAA"], "CPE": [0.5, 0.5]})],
        ignore_index=True,
    )
    purity.to_csv(cpe, index=False)

    tcga_prediction.compare_cancer_cell_cpe(result)
    svg = (result / "compare_with_CPE/cancer_cell_cpe_ccc_comparison.svg").read_text(encoding="utf-8")
    assert svg.index("<!-- ACC -->") < svg.index("<!-- AAA -->")
