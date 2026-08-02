import pandas as pd
import pytest

pytest.importorskip("umap")
pytest.importorskip("statsmodels")

from vaedecon.plot.evaluate_result import (
    _build_selected_sample_color_map,
    _build_selected_sample_legend_label_map,
    compare_y_y_pred_subplot,
    _draw_empty_selected_sample_panel,
    _filter_selected_samples_by_true_prop,
    _format_threshold_for_filename,
)


def test_build_selected_sample_legend_label_map_uses_ground_truth_props():
    selected_true_cell_prop = pd.DataFrame(
        [[0.843, 0.001], [0.010, 0.250]],
        index=["s1", "s2"],
        columns=["Cancer Cells", "CD8 T"],
    )

    label_map = _build_selected_sample_legend_label_map(
        selected_true_cell_prop=selected_true_cell_prop,
        cell_type="Cancer Cells",
        sample_ids=["s1", "s2"],
    )

    assert label_map["s1"] == "s1 (true=0.843)"
    assert label_map["s2"] == "s2 (true=0.01)"


def test_filter_selected_samples_by_true_prop_keeps_only_eligible_samples():
    selected_true_cell_prop = pd.DataFrame(
        [[0.004, 0.006], [0.200, 0.001]],
        index=["s1", "s2"],
        columns=["Cancer Cells", "CD8 T"],
    )

    kept = _filter_selected_samples_by_true_prop(
        selected_true_cell_prop=selected_true_cell_prop,
        cell_type="Cancer Cells",
        sample_ids=["s1", "s2"],
        min_true_cell_prop=0.005,
    )

    assert kept == ["s2"]


def test_draw_empty_selected_sample_panel_adds_threshold_note():
    matplotlib = pytest.importorskip("matplotlib")
    pyplot = pytest.importorskip("matplotlib.pyplot")

    fig, ax = pyplot.subplots(figsize=(2, 2))
    _draw_empty_selected_sample_panel(ax=ax, threshold=0.005)

    text_values = [text.get_text() for text in ax.texts]
    assert "No sample with true prop >= 0.005" in text_values
    matplotlib.pyplot.close(fig)


def test_format_threshold_for_filename_matches_expected_style():
    assert _format_threshold_for_filename(0.005) == "0p005"
    assert _format_threshold_for_filename(0.01) == "0p01"


def test_filtered_selected_samples_keep_original_color_mapping():
    matplotlib = pytest.importorskip("matplotlib")
    pyplot = pytest.importorskip("matplotlib.pyplot")

    y_true = pd.DataFrame(
        {
            "s1": [0.1, 0.2],
            "s2": [0.3, 0.4],
            "s3": [0.5, 0.6],
            "s4": [0.7, 0.8],
        }
    )
    y_pred = y_true.copy()
    color_map = _build_selected_sample_color_map(["s1", "s2", "s3", "s4"])

    fig, ax = pyplot.subplots(figsize=(2, 2))
    compare_y_y_pred_subplot(
        y_true=y_true,
        y_pred=y_pred,
        show_columns=["s2", "s4"],
        ax=ax,
        show_legend=True,
        legend_label_map={"s2": "s2", "s4": "s4"},
        series_color_map=color_map,
    )

    facecolors = [collection.get_facecolors()[0] for collection in ax.collections]
    expected_colors = [
        matplotlib.colors.to_rgba(color_map["s2"]),
        matplotlib.colors.to_rgba(color_map["s4"]),
    ]

    for actual, expected in zip(facecolors, expected_colors):
        assert tuple(actual) == pytest.approx(expected)

    matplotlib.pyplot.close(fig)
