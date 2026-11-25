"""TCGA comparisons against DeSide and consensus purity estimates."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

DESIDE_CPE_COLUMNS = ["DeSide-D1"] + [f"DeSide-D1_{i}" for i in range(1, 5)]
CPE_SCATTER_CANCER_TYPES = ("LIHC", "BLCA", "CESC", "HNSC")


def _read_sample_indexed_csv(path: Path) -> pd.DataFrame:
    """Keep sample IDs as strings even when every ID looks numeric."""
    frame = pd.read_csv(path, converters={0: str})
    return frame.set_index(frame.columns[0])


def _calculate_ccc(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, int]:
    """Calculate Lin's CCC for finite paired observations."""
    valid = np.isfinite(values_a) & np.isfinite(values_b)
    values_a = np.asarray(values_a[valid], dtype=np.float64)
    values_b = np.asarray(values_b[valid], dtype=np.float64)
    n_samples = int(values_a.size)
    if n_samples < 2:
        return float("nan"), n_samples

    centered_a = values_a - values_a.mean()
    centered_b = values_b - values_b.mean()
    variance_a = np.mean(centered_a**2)
    variance_b = np.mean(centered_b**2)
    covariance = np.mean(centered_a * centered_b)
    denominator = variance_a + variance_b + (values_a.mean() - values_b.mean()) ** 2
    ccc = 2 * covariance / denominator if denominator > 0 else float("nan")
    return float(ccc), n_samples


def _load_cpe_matched_predictions(
    result_dir: Path, cancer_purity_file: Path, cancer_type_mapping_file: Path
) -> pd.DataFrame:
    """Match predictions to CPE using exact sample IDs and validated cancer types."""
    prediction_path = Path(result_dir) / "predicted_cell_prop.csv"
    predictions = _read_sample_indexed_csv(prediction_path)
    if predictions.index.hasnans or predictions.index.duplicated().any():
        raise ValueError("VAEDecon sample IDs must be present and unique.")
    if (predictions.index.astype(str).str.strip() == "").any():
        raise ValueError("VAEDecon sample IDs must be present and nonempty.")
    if "Cancer Cells" not in predictions:
        raise ValueError("VAEDecon predictions are missing the 'Cancer Cells' column.")

    cancer_types = _read_sample_indexed_csv(cancer_type_mapping_file)
    if cancer_types.index.hasnans or cancer_types.index.duplicated().any():
        raise ValueError("TCGA cancer-type mapping sample IDs must be present and unique.")
    if "cancer_type" not in cancer_types:
        raise ValueError("TCGA cancer-type mapping is missing 'cancer_type'.")
    mapped_types = cancer_types["cancer_type"].reindex(predictions.index)
    if mapped_types.isna().any() or mapped_types.astype(str).str.strip().eq("").any():
        raise ValueError("Every VAEDecon sample must have a TCGA cancer-type label.")

    purity = pd.read_csv(
        cancer_purity_file, usecols=["Sample ID", "Cancer type", "CPE"], dtype={"Sample ID": str}
    )
    if purity["Sample ID"].isna().any() or purity["Sample ID"].duplicated().any():
        raise ValueError("CPE sample IDs must be present and unique.")
    if purity["Sample ID"].astype(str).str.strip().eq("").any():
        raise ValueError("CPE sample IDs must be present and nonempty.")
    purity = purity.set_index("Sample ID")
    matched = predictions[["Cancer Cells"]].join(purity, how="left")
    matched["cancer_type"] = mapped_types
    has_cpe_sample = matched.index.isin(purity.index)
    if matched.loc[has_cpe_sample, "Cancer type"].isna().any():
        raise ValueError("Matched CPE samples must have cancer-type labels.")
    if not (
        matched.loc[has_cpe_sample, "Cancer type"]
        == matched.loc[has_cpe_sample, "cancer_type"]
    ).all():
        raise ValueError("Matched samples have conflicting cancer-type labels.")
    if not has_cpe_sample.any():
        raise ValueError("No exact sample IDs match between predictions and CPE.")
    matched["has_cpe_sample"] = has_cpe_sample
    matched["Cancer Cells"] = pd.to_numeric(matched["Cancer Cells"], errors="coerce")
    matched["CPE"] = pd.to_numeric(matched["CPE"], errors="coerce")
    return matched


def _plot_cpe_scatter(matched: pd.DataFrame, comparison_dir: Path) -> Path:
    """Plot finite paired values for four cancer types on common axes."""
    import matplotlib.pyplot as plt

    pairs = {}
    for cancer_type in CPE_SCATTER_CANCER_TYPES:
        group = matched.loc[matched["cancer_type"] == cancer_type]
        x = group["CPE"].to_numpy(dtype=float)
        y = group["Cancer Cells"].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        pairs[cancer_type] = (x[valid], y[valid])

    values = [value for x, y in pairs.values() for value in (x, y) if value.size]
    lower = min(0.0, min(float(value.min()) for value in values)) if values else 0.0
    upper = max(1.0, max(float(value.max()) for value in values)) if values else 1.0
    margin = 0.025 * (upper - lower)
    limits = (lower - margin, upper + margin)

    def format_metric(value: float) -> str:
        return f"{value:.3f}" if np.isfinite(value) else "NA"

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.5), sharex=True, sharey=True,
                             constrained_layout=True)
    for axis, cancer_type in zip(axes.flat, CPE_SCATTER_CANCER_TYPES):
        x, y = pairs[cancer_type]
        ccc, n_samples = _calculate_ccc(x, y)
        rmse = float(np.sqrt(np.mean((x - y) ** 2))) if n_samples else float("nan")
        pearson = (
            float(np.corrcoef(x, y)[0, 1])
            if n_samples >= 2 and np.var(x) > 0 and np.var(y) > 0
            else float("nan")
        )
        axis.scatter(x, y, s=12, color="#2878B5", alpha=0.55, linewidths=0,
                     rasterized=True)
        axis.plot(limits, limits, color="#555555", linestyle="--", linewidth=0.8)
        axis.set(title=cancer_type, xlim=limits, ylim=limits, aspect="equal")
        axis.text(
            0.04, 0.96,
            f"n = {n_samples}\nCCC = {format_metric(ccc)}\n"
            f"Pearson r = {format_metric(pearson)}\nRMSE = {format_metric(rmse)}",
            transform=axis.transAxes, ha="left", va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
        axis.grid(alpha=0.18, linewidth=0.5)
    fig.supxlabel("Consensus purity estimate (CPE)")
    fig.supylabel("VAEDecon cancer cell proportion")
    comparison_dir = Path(comparison_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(comparison_dir / f"cancer_cell_cpe_scatter.{extension}", dpi=300)
    plt.close(fig)
    return comparison_dir / "cancer_cell_cpe_scatter.png"


def compare_cancer_cell_cpe(
    result_dir: Path,
    cancer_purity_file: Path,
    cancer_type_mapping_file: Path,
    deside_cpe_ccc_file: Path,
) -> dict:
    """Compare existing Cancer Cells predictions with CPE and DeSide CCCs."""
    matched = _load_cpe_matched_predictions(
        result_dir, cancer_purity_file, cancer_type_mapping_file
    )

    rows = []
    for cancer_type, group in matched.groupby("cancer_type", sort=True):
        ccc, n_valid = _calculate_ccc(
            group["Cancer Cells"].to_numpy(dtype=float),
            group["CPE"].to_numpy(dtype=float),
        )
        n_matched = int(group["has_cpe_sample"].sum())
        rows.append(
            {
                "cancer_type": cancer_type,
                "n_predictions": len(group),
                "n_matched": n_matched,
                "n_excluded_nonfinite": n_matched - n_valid,
                "n_valid": n_valid,
                "vaedecon_ccc": ccc,
            }
        )
    metrics = pd.DataFrame(rows).set_index("cancer_type")

    deside = pd.read_csv(deside_cpe_ccc_file, index_col=0)
    if deside.index.hasnans or deside.index.duplicated().any():
        raise ValueError("DeSide cancer-type labels must be present and unique.")
    missing_columns = [column for column in DESIDE_CPE_COLUMNS if column not in deside]
    if missing_columns:
        raise ValueError(f"DeSide CCC table is missing columns: {missing_columns}")
    replicates = deside[DESIDE_CPE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(replicates.to_numpy(dtype=float)).all():
        raise ValueError("DeSide CCC replicates must all be finite.")
    replicates["deside_ccc_mean"] = replicates.mean(axis=1)
    metrics = metrics.join(replicates, how="outer").sort_index()
    metrics.index.name = "cancer_type"

    comparison_dir = Path(result_dir) / "compare_with_CPE"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = comparison_dir / "cancer_cell_cpe_ccc_by_cancer_type.csv"
    metrics.to_csv(metrics_path, float_format="%.8g")

    import matplotlib.pyplot as plt

    plot_metrics = metrics.drop(index="PAAD", errors="ignore").sort_values(
        "vaedecon_ccc", ascending=False, kind="stable", na_position="last"
    )
    positions = np.arange(len(plot_metrics))
    width = 0.38
    fig, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    for offset, column, label, color in (
        (-width / 2, "vaedecon_ccc", "VAEDecon", "#2878B5"),
        (width / 2, "deside_ccc_mean", "DeSide (mean of 5)", "#D18028"),
    ):
        values = plot_metrics[column].to_numpy(dtype=float)
        valid = np.isfinite(values)
        bars = axis.bar(positions[valid] + offset, values[valid], width, label=label, color=color)
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=6, rotation=90)
        for position in positions[~valid]:
            axis.text(position + offset, 0.02, "NA", ha="center", va="bottom", fontsize=6, rotation=90)
    finite_values = plot_metrics[["vaedecon_ccc", "deside_ccc_mean"]].to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    lower = min(0.0, float(finite_values.min()) - 0.08) if finite_values.size else 0.0
    axis.set(
        xticks=positions,
        xticklabels=plot_metrics.index,
        ylim=(lower, 1.12),
        ylabel="Concordance correlation coefficient (CCC)",
    )
    axis.set_title("Cancer cell proportions vs CPE by cancer type", pad=25)
    axis.text(
        0.5, 1.01, "DeSide CCCs are precomputed; matched sample sets may differ.",
        transform=axis.transAxes, ha="center", va="bottom", fontsize=8,
    )
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    for extension in ("png", "svg"):
        fig.savefig(comparison_dir / f"cancer_cell_cpe_ccc_comparison.{extension}", dpi=300)
    plt.close(fig)
    scatter_path = _plot_cpe_scatter(matched, comparison_dir)

    return {
        "prediction_count": len(matched),
        "matched_sample_count": int(matched["has_cpe_sample"].sum()),
        "valid_pair_count": int(metrics["n_valid"].sum()),
        "cancer_types_without_cpe_pairs": metrics.index[metrics["n_valid"].fillna(0) == 0].tolist(),
        "metrics_file": str(metrics_path),
        "figure_file": str(comparison_dir / "cancer_cell_cpe_ccc_comparison.png"),
        "scatter_figure_file": str(scatter_path),
    }


def _compare_all_cell_types_deside(
    vaedecon: pd.DataFrame,
    deside: pd.DataFrame,
    cancer_types: pd.Series,
    output_dir: Path,
) -> dict:
    """Compare paired cell proportions across all shared TCGA samples."""
    import matplotlib.pyplot as plt

    cell_types = vaedecon.columns.tolist()
    labels = sorted(cancer_types.unique())
    colors = plt.get_cmap("tab20").colors
    color_map = {label: colors[index % len(colors)] for index, label in enumerate(labels)}
    rows = []
    for cell_type in cell_types:
        x = pd.to_numeric(deside[cell_type], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(vaedecon[cell_type], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x_valid, y_valid = x[valid], y[valid]
        ccc, n_samples = _calculate_ccc(x, y)
        rmse = float(np.sqrt(np.mean((x_valid - y_valid) ** 2))) if n_samples else float("nan")
        pearson = (
            float(np.corrcoef(x_valid, y_valid)[0, 1])
            if n_samples >= 2 and np.var(x_valid) > 0 and np.var(y_valid) > 0
            else float("nan")
        )
        rows.append({
            "cell_type": cell_type,
            "n_samples": n_samples,
            "rmse": rmse,
            "pearson_r": pearson,
            "ccc": ccc,
        })

        fig, axis = plt.subplots(figsize=(7.6, 5.7), constrained_layout=True)
        for label in labels:
            group = valid & (cancer_types.to_numpy() == label)
            axis.scatter(
                x[group], y[group], s=9, alpha=0.5, color=color_map[label],
                linewidths=0, label=label, rasterized=True,
            )
        finite_values = np.concatenate([x_valid, y_valid])
        lower = min(0.0, float(finite_values.min())) if finite_values.size else 0.0
        upper = max(0.0, float(finite_values.max())) if finite_values.size else 0.0
        margin = max((upper - lower) * 0.05, 1e-6)
        lower, upper = (lower - margin if lower < 0 else 0.0), upper + margin
        axis.plot([lower, upper], [lower, upper], color="#4d4d4d", linestyle="--", linewidth=0.9)
        axis.set(
            title=f"{cell_type}: VAEDecon vs DeSide",
            xlabel="DeSide cell proportion", ylabel="VAEDecon cell proportion",
            xlim=(lower, upper), ylim=(lower, upper),
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.18, linewidth=0.45)
        axis.text(
            0.03, 0.97,
            f"RMSE = {rmse:.3f}\nPearson r = {pearson:.3f}\nCCC = {ccc:.3f}\nn = {n_samples:,}",
            transform=axis.transAxes, ha="left", va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        axis.legend(
            title="Cancer type", loc="upper left", bbox_to_anchor=(1.02, 1),
            borderaxespad=0, frameon=False, fontsize=7.5, title_fontsize=8,
        )
        filename = re.sub(r"[^A-Za-z0-9_-]+", "_", cell_type).strip("_")
        for extension in ("png", "svg"):
            fig.savefig(
                output_dir / f"cell_proportion_{filename}_by_cancer_type.{extension}",
                dpi=300, bbox_inches="tight",
            )
        plt.close(fig)

    metrics = pd.DataFrame(rows).set_index("cell_type")
    metrics_path = output_dir / "cell_type_metrics.csv"
    metrics.to_csv(metrics_path, float_format="%.8g")
    ordered = metrics.sort_values("ccc", ascending=True, na_position="last")
    fig, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    bars = axis.bar(ordered.index, ordered["ccc"], color="#2878B5", edgecolor="white", linewidth=0.5)
    finite_ccc = ordered["ccc"].dropna()
    lower = min(0.0, float(finite_ccc.min()) - 0.05) if not finite_ccc.empty else 0.0
    axis.set(
        ylabel="Concordance correlation coefficient (CCC)", ylim=(lower, 1.0),
        title="VAEDecon vs DeSide CCC by Cell Type",
    )
    axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", labelrotation=55, labelsize=8)
    for tick in axis.get_xticklabels():
        tick.set_ha("right")
    axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"cell_type_ccc_summary.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "cell_type_count": len(metrics),
        "cell_type_metrics_file": str(metrics_path),
        "cell_type_figure_file": str(output_dir / "cell_type_ccc_summary.png"),
    }


def compare_cancer_cell_ccc(result_dir: Path, deside_proportion_file: Path) -> dict:
    """Compare Cancer Cells proportions by cancer type without running inference."""
    prediction_path = Path(result_dir) / "predicted_cell_prop.csv"
    comparison_dir = Path(result_dir) / "compare_with_DeSide"
    required_paths = (prediction_path, deside_proportion_file)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required comparison input does not exist: {path}")

    vaedecon = _read_sample_indexed_csv(prediction_path)
    if (
        vaedecon.index.isna().any()
        or (vaedecon.index.astype(str).str.strip() == "").any()
    ):
        raise ValueError("VAEDecon sample IDs must be present and nonempty.")
    vaedecon.index = vaedecon.index.astype(str)
    if vaedecon.index.duplicated().any():
        raise ValueError("VAEDecon sample IDs must be unique.")
    cell_type = "Cancer Cells"
    if cell_type not in vaedecon.columns:
        raise ValueError(f"VAEDecon predictions are missing the {cell_type!r} column.")

    deside = pd.read_csv(
        deside_proportion_file,
        dtype={"sample_id": str, "cancer_type": str},
    )
    deside["cancer_type"] = deside["cancer_type"].str.strip()
    if (
        deside["sample_id"].isna().any()
        or deside["sample_id"].str.strip().eq("").any()
        or deside["sample_id"].duplicated().any()
    ):
        raise ValueError("DeSide sample IDs must be present and unique.")
    if deside["cancer_type"].isna().any() or deside["cancer_type"].str.strip().eq("").any():
        raise ValueError("DeSide cancer-type labels must be present and nonempty.")
    deside = deside.set_index("sample_id")
    deside_columns = [
        column for column in deside.columns
        if column != "cancer_type" and not column.startswith("Unnamed:")
    ]
    if set(vaedecon.columns) != set(deside_columns):
        raise ValueError(
            "Cell-type columns do not match between VAEDecon and DeSide: "
            f"VAEDecon-only {sorted(set(vaedecon.columns) - set(deside_columns))}; "
            f"DeSide-only {sorted(set(deside_columns) - set(vaedecon.columns))}."
        )

    shared_ids = vaedecon.index[vaedecon.index.isin(deside.index)]
    if shared_ids.empty:
        raise ValueError("No shared sample IDs were found between VAEDecon and DeSide.")
    vaedecon_values = pd.to_numeric(
        vaedecon.loc[shared_ids, cell_type], errors="coerce"
    )
    deside_values = pd.to_numeric(
        deside.loc[shared_ids, cell_type], errors="coerce"
    )
    cancer_types = deside.loc[shared_ids, "cancer_type"].astype(str)

    rows = []
    for cancer in sorted(cancer_types.unique()):
        group_mask = cancer_types.to_numpy() == cancer
        ccc, n_samples = _calculate_ccc(
            vaedecon_values.to_numpy(dtype=np.float64)[group_mask],
            deside_values.to_numpy(dtype=np.float64)[group_mask],
        )
        rows.append(
            {
                "cancer_type": cancer,
                "n_samples": n_samples,
                "ccc": ccc,
            }
        )
    metrics = pd.DataFrame(rows)
    if metrics.empty or not np.isfinite(metrics["ccc"]).all():
        raise ValueError("Could not calculate finite Cancer Cells CCC for every cancer type.")

    comparison_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = comparison_dir / "cancer_cell_comparison_CCC_by_cancer_type.csv"
    figure_path = comparison_dir / "cancer_cell_comparison_CCC.png"
    metrics.to_csv(metrics_path, index=False, float_format="%.8g")

    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab20").colors
    color_map = {
        cancer: colors[index % len(colors)]
        for index, cancer in enumerate(metrics["cancer_type"])
    }
    fig, axis = plt.subplots(figsize=(9.5, 6), constrained_layout=True)
    ccc_values = metrics["ccc"].to_numpy(dtype=np.float64)
    axis.boxplot(
        ccc_values,
        vert=False,
        positions=[1],
        widths=0.45,
        patch_artist=True,
        boxprops={"facecolor": "#d9e8f5", "edgecolor": "#3a536b"},
        medianprops={"color": "#b23a48", "linewidth": 1.5},
        whiskerprops={"color": "#3a536b"},
        capprops={"color": "#3a536b"},
        flierprops={"marker": ""},
    )
    offsets = np.linspace(-0.16, 0.16, len(metrics))
    for offset, row in zip(offsets, metrics.itertuples(index=False)):
        axis.scatter(
            row.ccc,
            1 + offset,
            color=color_map[row.cancer_type],
            edgecolor="white",
            linewidth=0.5,
            s=45,
            label=f"{row.cancer_type} (n={row.n_samples})",
            zorder=3,
        )

    axis.set(
        xlim=(min(0.0, float(ccc_values.min()) - 0.05), 1),
        ylim=(0.65, 1.35),
        xlabel="Concordance correlation coefficient (CCC)",
        yticks=[1],
        yticklabels=["All cancer types"],
        title=f"Cancer Cells: VAEDecon vs DeSide ({len(shared_ids):,} matched samples)",
    )
    axis.grid(axis="x", alpha=0.2, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.legend(
        title="Cancer type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=7.5,
        title_fontsize=8,
        ncol=1,
    )
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    all_cell_types = _compare_all_cell_types_deside(
        vaedecon.loc[shared_ids], deside.loc[shared_ids], cancer_types, comparison_dir
    )
    return {
        "shared_sample_count": int(len(shared_ids)),
        "cancer_type_count": int(len(metrics)),
        "ccc_min": float(metrics["ccc"].min()),
        "ccc_median": float(metrics["ccc"].median()),
        "ccc_max": float(metrics["ccc"].max()),
        "metrics_file": str(metrics_path),
        "figure_file": str(figure_path),
        **all_cell_types,
    }
