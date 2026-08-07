import pandas as pd
import pytest

pytest.importorskip("umap")
pytest.importorskip("statsmodels")

from vaedecon.plot.evaluate_result import (
    _build_selected_sample_color_map,
    _build_selected_sample_legend_label_map,
    _compute_pairwise_ccc_matrix,
    _plot_pairwise_ccc_clustermap,
    _plot_pairwise_ccc_heatmap,
    _save_selected_sample_similarity_outputs,
    _write_similarity_result_gallery,
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

    facecolors = [collection.get_facecolors()[0][:3] for collection in ax.collections]
    expected_colors = [
        matplotlib.colors.to_rgb(color_map["s2"]),
        matplotlib.colors.to_rgb(color_map["s4"]),
    ]

    for actual, expected in zip(facecolors, expected_colors):
        assert tuple(actual) == pytest.approx(expected)

    matplotlib.pyplot.close(fig)


def test_compute_pairwise_ccc_matrix_returns_expected_shape_and_labels():
    y_true = pd.DataFrame(
        {
            "s1": [1.0, 2.0, 3.0],
            "s2": [1.1, 2.1, 3.1],
        },
        index=["g1", "g2", "g3"],
    )
    y_pred = pd.DataFrame(
        {
            "s1": [1.0, 2.0, 3.0],
            "s2": [1.2, 2.2, 3.2],
        },
        index=["g1", "g2", "g3"],
    )

    matrix = _compute_pairwise_ccc_matrix(
        left_df=y_pred,
        right_df=y_true,
        row_sample_ids=["s1", "s2"],
        col_sample_ids=["s1", "s2"],
    )

    assert list(matrix.index) == ["s1", "s2"]
    assert list(matrix.columns) == ["s1", "s2"]
    assert matrix.shape == (2, 2)
    assert matrix.loc["s1", "s1"] == pytest.approx(1.0)


def test_save_selected_sample_similarity_outputs_writes_expected_matrix_types(tmp_path):
    y_true = pd.DataFrame(
        {
            "s1": [1.0, 2.0, 3.0],
            "s2": [1.1, 2.1, 3.1],
        },
        index=["g1", "g2", "g3"],
    )
    y_pred = pd.DataFrame(
        {
            "s1": [1.0, 2.0, 3.0],
            "s2": [1.2, 2.2, 3.2],
        },
        index=["g1", "g2", "g3"],
    )

    _save_selected_sample_similarity_outputs(
        cell_type="Cancer Cells",
        y_true=y_true,
        y_pred=y_pred,
        sample_ids=["s1", "s2"],
        similarity_result_dir=tmp_path,
        threshold=0.005,
        figure_format="png",
    )

    expected_stem = "Cancer Cells_ccc_true_prop_ge_0p005"
    for matrix_name in ["true_vs_true", "recon_vs_recon", "true_vs_recon"]:
        assert (tmp_path / f"{expected_stem}_{matrix_name}.csv").exists()
        assert (tmp_path / f"{expected_stem}_{matrix_name}.png").exists()

    true_vs_recon = pd.read_csv(
        tmp_path / f"{expected_stem}_true_vs_recon.csv",
        index_col=0,
    )
    assert true_vs_recon.loc["s1", "s1"] == pytest.approx(1.0)


def test_plot_pairwise_ccc_heatmap_uses_nonnegative_data_driven_bounds(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("matplotlib.pyplot")

    matrix_df = pd.DataFrame(
        [[0.21, 0.35], [0.41, 0.93]],
        index=["s1", "s2"],
        columns=["s1", "s2"],
    )
    captured = {}

    def _fake_heatmap(*_args, **kwargs):
        captured["vmin"] = kwargs.get("vmin")
        captured["vmax"] = kwargs.get("vmax")
        return kwargs["ax"]

    monkeypatch.setattr("vaedecon.plot.evaluate_result.sns.heatmap", _fake_heatmap)

    _plot_pairwise_ccc_heatmap(
        matrix_df=matrix_df,
        output_fp=tmp_path / "ccc.png",
        title="CCC",
    )

    assert captured["vmin"] == pytest.approx(0.21)
    assert captured["vmax"] == pytest.approx(0.93)


def test_plot_pairwise_ccc_clustermap_uses_nonnegative_data_driven_bounds(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("matplotlib.pyplot")

    matrix_df = pd.DataFrame(
        [[0.21, 0.35], [0.41, 0.93]],
        index=["s1", "s2"],
        columns=["s1", "s2"],
    )
    captured = {}

    class _FakeAxis:
        def set_title(self, *_args, **_kwargs):
            return None

        def set_xlabel(self, *_args, **_kwargs):
            return None

        def set_ylabel(self, *_args, **_kwargs):
            return None

        def tick_params(self, *_args, **_kwargs):
            return None

    class _FakeFigure:
        pass

    class _FakeClusterGrid:
        def __init__(self):
            self.ax_heatmap = _FakeAxis()
            self.figure = _FakeFigure()

        def savefig(self, output_fp, dpi=300):
            captured["output_fp"] = output_fp
            captured["dpi"] = dpi

    def _fake_clustermap(*_args, **kwargs):
        captured["vmin"] = kwargs.get("vmin")
        captured["vmax"] = kwargs.get("vmax")
        return _FakeClusterGrid()

    monkeypatch.setattr("vaedecon.plot.evaluate_result.sns.clustermap", _fake_clustermap)
    monkeypatch.setattr("vaedecon.plot.evaluate_result.plt.close", lambda *_args, **_kwargs: None)

    _plot_pairwise_ccc_clustermap(
        matrix_df=matrix_df,
        output_fp=tmp_path / "ccc_clustermap.png",
        title="CCC",
    )

    assert captured["vmin"] == pytest.approx(0.21)
    assert captured["vmax"] == pytest.approx(0.93)
    assert str(captured["output_fp"]).endswith("ccc_clustermap.png")


def test_write_similarity_result_gallery_includes_expected_links(tmp_path):
    prefix_cancer = "Cancer Cells_ccc_true_prop_ge_0p005"
    prefix_cafs = "CAFs_ccc_true_prop_ge_0p005"
    (tmp_path / f"{prefix_cancer}_recon_vs_recon.csv").write_text("a,b\ns1,1.0\n")
    (tmp_path / f"{prefix_cancer}_recon_vs_recon.png").write_text("png")
    (tmp_path / f"{prefix_cancer}_recon_vs_recon_clustermap.png").write_text("png")
    (tmp_path / f"{prefix_cancer}_true_vs_true.png").write_text("png")
    (tmp_path / f"{prefix_cafs}_true_vs_true.png").write_text("png")

    _write_similarity_result_gallery(
        similarity_result_dir=tmp_path,
        figure_format="png",
    )

    html_text = (tmp_path / "index.html").read_text()
    assert "Cancer Cells" in html_text
    assert "Cell Types" in html_text
    assert "Figure Types: CAFs" in html_text
    assert "href=\"#cell-type-cancer-cells\"" in html_text
    assert "data-cell-type-link=\"cell-type-cancer-cells\"" in html_text
    assert "href=\"#variant-cafs-ccc-true-prop-ge-0p005-true-vs-true\"" in html_text
    assert f"{prefix_cancer}_recon_vs_recon.csv" in html_text
    assert f"{prefix_cancer}_recon_vs_recon.png" in html_text
    assert f"{prefix_cancer}_recon_vs_recon_clustermap.png" in html_text
    assert f"{prefix_cancer}_true_vs_true.png" in html_text


def test_save_selected_sample_similarity_outputs_writes_clustermap_and_gallery(tmp_path, monkeypatch):
    y_true = pd.DataFrame(
        {
            "s1": [1.0, 2.0, 3.0],
            "s2": [1.1, 2.1, 3.1],
        },
        index=["g1", "g2", "g3"],
    )
    y_pred = pd.DataFrame(
        {
            "s1": [1.0, 2.0, 3.0],
            "s2": [1.2, 2.2, 3.2],
        },
        index=["g1", "g2", "g3"],
    )
    called_outputs = {"heatmap": [], "clustermap": []}

    def _fake_heatmap(matrix_df, output_fp, title):
        called_outputs["heatmap"].append((matrix_df.copy(), output_fp, title))
        output_fp.write_text("heatmap")

    def _fake_clustermap(matrix_df, output_fp, title):
        called_outputs["clustermap"].append((matrix_df.copy(), output_fp, title))
        output_fp.write_text("clustermap")

    monkeypatch.setattr("vaedecon.plot.evaluate_result._plot_pairwise_ccc_heatmap", _fake_heatmap)
    monkeypatch.setattr("vaedecon.plot.evaluate_result._plot_pairwise_ccc_clustermap", _fake_clustermap)

    _save_selected_sample_similarity_outputs(
        cell_type="Cancer Cells",
        y_true=y_true,
        y_pred=y_pred,
        sample_ids=["s1", "s2"],
        similarity_result_dir=tmp_path,
        threshold=0.005,
        figure_format="png",
    )

    expected_stem = "Cancer Cells_ccc_true_prop_ge_0p005"
    assert len(called_outputs["heatmap"]) == 3
    assert len(called_outputs["clustermap"]) == 3
    assert (tmp_path / f"{expected_stem}_recon_vs_recon_clustermap.png").exists()
    assert (tmp_path / "index.html").exists()


def test_save_selected_sample_similarity_outputs_empty_case_skips_clustermap(tmp_path, monkeypatch):
    called_outputs = {"clustermap": 0}

    def _fake_clustermap(*_args, **_kwargs):
        called_outputs["clustermap"] += 1

    monkeypatch.setattr("vaedecon.plot.evaluate_result._plot_pairwise_ccc_clustermap", _fake_clustermap)

    _save_selected_sample_similarity_outputs(
        cell_type="Cancer Cells",
        y_true=pd.DataFrame(),
        y_pred=pd.DataFrame(),
        sample_ids=[],
        similarity_result_dir=tmp_path,
        threshold=0.005,
        figure_format="png",
    )

    assert called_outputs["clustermap"] == 0
    assert (tmp_path / "index.html").exists()
