"""Inference for predictor-only cell-proportion checkpoints."""

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch

from ..data.bulk_sources import group_cell_fractions


def prepare_staged_expression(values, source_genes, genes, scaling_factor=None):
    positions = {gene: index for index, gene in enumerate(source_genes)}
    if len(positions) != len(source_genes):
        raise ValueError("Prediction input contains duplicate genes")
    present = [(i, positions[gene]) for i, gene in enumerate(genes) if gene in positions]
    if not present:
        raise ValueError("Prediction input has no genes in common with the model")
    selected = np.asarray(values[:, [source for _, source in present]], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("Prediction input contains non-finite expression")
    linear = np.maximum(np.exp2(selected) - 1.0, 0.0)
    if not np.isfinite(linear).all():
        raise ValueError("Prediction input contains invalid linear expression")
    totals = linear.sum(axis=1, keepdims=True)
    if not np.isfinite(totals).all():
        raise ValueError("Prediction input has non-finite expression totals")
    valid = totals[:, 0] > 0
    selected = np.log2(linear * (1e6 / np.maximum(totals, 1e-12)) + 1.0)
    aligned = np.zeros((len(selected), len(genes)), dtype=np.float32)
    aligned[:, [target for target, _ in present]] = selected
    if scaling_factor is not None:
        aligned /= float(scaling_factor)
    return aligned, valid


def _predict_expression_batch(model, values, source_genes, genes, device):
    scale = model.data_config.scaling_factor if model.data_config.scaling_by_constant else None
    aligned, _ = prepare_staged_expression(values, source_genes, genes, scale)
    return _predict_prepared_batch(model, aligned, device)


def _predict_prepared_batch(model, aligned, device):
    output = model({"data": torch.as_tensor(aligned, device=device)})
    return postprocess_predictions(output["raw_non_cancer_cell_prop"].cpu().numpy(), model)


def postprocess_predictions(raw, model):
    """Apply DeSide's complement, threshold, normalization, and residual."""
    config = model.model_config
    values = raw.copy()
    if config.deside_one_minus_alpha:
        values = 1.0 - values
    values[values < config.deside_prediction_threshold] = 0.0
    if config.deside_renormalize_non_cancer:
        totals = values.sum(axis=1, keepdims=True)
        values = values / np.maximum(totals, 1.0)
    cancer_index = model.cell_prop_predictor.cancer_cell_type_index
    if not config.deside_fill_cancer_residual:
        return values
    cancer = np.maximum(1.0 - values.sum(axis=1, keepdims=True), 0.0)
    return np.concatenate((values[:, :cancer_index], cancer, values[:, cancer_index:]), axis=1)


def _save_proportion_metrics(prediction, truth, output_dir):
    """Save aligned labels and finite-pair metrics without rounding values."""
    truth = truth.loc[prediction.index, prediction.columns]
    truth.to_csv(output_dir / "true_cell_prop.csv")
    metrics = []
    for name in prediction.columns:
        x = pd.to_numeric(truth[name], errors="coerce").to_numpy(dtype=float)
        y = prediction[name].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        rmse = float(np.sqrt(np.mean((x - y) ** 2))) if x.size else np.nan
        if x.size >= 2 and np.std(x) > 0 and np.std(y) > 0:
            pearson = float(np.corrcoef(x, y)[0, 1])
        else:
            pearson = np.nan
        denominator = np.var(x) + np.var(y) + (np.mean(x) - np.mean(y)) ** 2 if x.size else 0
        ccc = float(2 * np.mean((x - x.mean()) * (y - y.mean())) / denominator) if denominator else np.nan
        metrics.append({"cell_type": name, "n": len(x), "rmse": rmse, "pearson": pearson, "ccc": ccc})
    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)


def save_validation_predictions(model, subset, output_dir, batch_size):
    """Save held-out predictions and per-cell-type regression metrics."""
    dataset = subset.dataset
    indices = list(subset.indices)
    predictions = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            inputs = torch.as_tensor(np.asarray(dataset.data[batch_indices]), device=device)
            raw = model({"data": inputs})["raw_non_cancer_cell_prop"].cpu().numpy()
            predictions.append(postprocess_predictions(raw, model))
    columns = dataset.cell_types
    if not model.model_config.deside_fill_cancer_residual:
        columns = [name for i, name in enumerate(columns)
                   if i != model.cell_prop_predictor.cancer_cell_type_index]
    names = [dataset.sample_ids[index] for index in indices]
    prediction = pd.DataFrame(np.concatenate(predictions), index=names, columns=columns)
    truth = dataset.get_cell_prop().loc[names, columns]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(output_dir / "predicted_cell_prop.csv")
    _save_proportion_metrics(prediction, truth, output_dir)


def predict_bulk_h5ad(model, path, output_dir, device, batch_size, staged_contract=False):
    """Predict from log2(CPM+1) H5AD data in bounded batches."""
    genes = Path(model.model_config.input_gene_list_fp).read_text(encoding="utf-8").splitlines()
    cell_types = Path(model.model_config.cell_type_fp).read_text(encoding="utf-8").splitlines()
    dataset = anndata.read_h5ad(path, backed="r")
    try:
        if dataset.n_obs == 0:
            raise ValueError("Prediction H5AD contains no samples")
        if not dataset.obs_names.is_unique:
            raise ValueError("Prediction H5AD contains duplicate sample IDs")
        source_genes = [str(gene) for gene in dataset.var_names]
        if len(source_genes) != len(set(source_genes)):
            raise ValueError("Prediction H5AD contains duplicate genes")
        predictions, accepted_ids, exclusions = [], [], []
        model.eval()
        with torch.no_grad():
            for start in range(0, dataset.n_obs, batch_size):
                block = dataset.X[start:start + batch_size]
                if hasattr(block, "toarray"):
                    block = block.toarray()
                ids = dataset.obs_names[start:start + len(block)].astype(str).tolist()
                if staged_contract:
                    scale = model.data_config.scaling_factor if model.data_config.scaling_by_constant else None
                    aligned, valid = prepare_staged_expression(block, source_genes, genes, scale)
                    exclusions.extend({"source": Path(path).stem, "sample_id": name,
                                       "reason": "zero_selected_expression"}
                                      for name, keep in zip(ids, valid) if not keep)
                    aligned = aligned[valid]
                    ids = [name for name, keep in zip(ids, valid) if keep]
                accepted_ids.extend(ids)
                if not ids:
                    continue
                predictions.append(
                    _predict_prepared_batch(model, aligned, device) if staged_contract
                    else _predict_expression_batch(model, block, source_genes, genes, device)
                )
        output_types = cell_types
        if not model.model_config.deside_fill_cancer_residual:
            output_types = [name for i, name in enumerate(cell_types)
                            if i != model.cell_prop_predictor.cancer_cell_type_index]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if staged_contract:
            pd.DataFrame(exclusions, columns=["source", "sample_id", "reason"]).to_csv(
                output_dir / "excluded_samples.csv", index=False
            )
        if not predictions:
            raise ValueError("No prediction samples have nonzero expression in the model genes")
        predicted = np.concatenate(predictions, axis=0)
        prediction_path = output_dir / "predicted_cell_prop.csv"
        prediction = pd.DataFrame(predicted, index=accepted_ids, columns=output_types)
        prediction.to_csv(prediction_path)
        true = dataset.obs.loc[accepted_ids].copy()
        if not true.empty and model.model_config.deside_group_cell_types:
            groups = model.model_config.deside_group_cell_types
            relevant = set(output_types) | {name for names in groups.values() for name in names}
            true = true.loc[:, [name for name in true.columns if name in relevant]]
            complete_groups = all(
                group in true.columns or set(names).issubset(true.columns)
                for group, names in groups.items()
            )
            if not true.empty and complete_groups:
                true = group_cell_fractions(true, groups)
            elif not complete_groups:
                true = pd.DataFrame()
        if not set(output_types).issubset(true.columns):
            true = pd.DataFrame()
        else:
            true = true.loc[:, output_types]
            _save_proportion_metrics(prediction, true, output_dir)
        cell_prop_dir = output_dir / "cell_prop"
        cell_prop_dir.mkdir(exist_ok=True)
        return {
            "true_cell_prop": true,
            "pred_cell_prop_file_path": str(prediction_path),
            "cell_types": output_types,
            "pred_a": {},
            "cell_prop_result_dir": str(cell_prop_dir),
            "gep_result_dir": None,
            "test_set_result_dir": str(output_dir),
            "test_set": None,
        }
    finally:
        dataset.file.close()


def predict_bulk_csv(model, path, output_dir, device, batch_size, staged_contract=False):
    """Predict cell proportions from a sample-by-gene log-expression CSV."""
    genes = Path(model.model_config.input_gene_list_fp).read_text(encoding="utf-8").splitlines()
    cell_types = Path(model.model_config.cell_type_fp).read_text(encoding="utf-8").splitlines()
    predictions, sample_ids, all_ids, exclusions = [], [], [], []
    model.eval()
    with torch.no_grad():
        for frame in pd.read_csv(path, index_col=0, chunksize=batch_size):
            ids = frame.index.astype(str).tolist()
            all_ids.extend(ids)
            if staged_contract:
                scale = model.data_config.scaling_factor if model.data_config.scaling_by_constant else None
                aligned, valid = prepare_staged_expression(
                    frame.to_numpy(), frame.columns.astype(str).tolist(), genes, scale
                )
                exclusions.extend({"source": Path(path).stem, "sample_id": name,
                                   "reason": "zero_selected_expression"}
                                  for name, keep in zip(ids, valid) if not keep)
                ids = [name for name, keep in zip(ids, valid) if keep]
            sample_ids.extend(ids)
            if not ids:
                continue
            predictions.append(
                _predict_prepared_batch(model, aligned[valid], device) if staged_contract
                else _predict_expression_batch(
                    model, frame.to_numpy(), frame.columns.astype(str).tolist(), genes, device
                )
            )
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Prediction CSV contains duplicate sample IDs")
    if not model.model_config.deside_fill_cancer_residual:
        cell_types = [name for i, name in enumerate(cell_types)
                      if i != model.cell_prop_predictor.cancer_cell_type_index]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if staged_contract:
        pd.DataFrame(exclusions, columns=["source", "sample_id", "reason"]).to_csv(
            output_dir / "excluded_samples.csv", index=False
        )
    if not predictions:
        raise ValueError("Prediction CSV contains no samples with nonzero model-gene expression")
    path = output_dir / "predicted_cell_prop.csv"
    pd.DataFrame(np.concatenate(predictions), index=sample_ids, columns=cell_types).to_csv(path)
    cell_prop_dir = output_dir / "cell_prop"
    cell_prop_dir.mkdir(exist_ok=True)
    return {
        "true_cell_prop": pd.DataFrame(), "pred_cell_prop_file_path": str(path),
        "cell_types": cell_types, "pred_a": {}, "cell_prop_result_dir": str(cell_prop_dir),
        "gep_result_dir": None, "test_set_result_dir": str(output_dir), "test_set": None,
    }
