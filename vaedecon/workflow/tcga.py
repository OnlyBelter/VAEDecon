"""Configured TCGA inference from sample-by-gene linear TPM."""

import csv
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..models.cell_prop_only import CellPropOnlyModel
from .bulk_prediction import _predict_prepared_batch
from .tcga_comparison import compare_cancer_cell_ccc, compare_cancer_cell_cpe
from .workflow import _resolve_trained_checkpoint_path, evaluate_model

logger = logging.getLogger(__name__)


def _comparison_status(output_dir, references):
    """Run independent comparisons without losing completed predictions."""
    checks = {
        "deside": (
            ("deside_proportion_file_path",),
            compare_cancer_cell_ccc,
        ),
        "cpe": (
            ("cancer_purity_file_path", "cancer_type_mapping_file_path",
             "deside_cpe_ccc_file_path"),
            compare_cancer_cell_cpe,
        ),
    }
    statuses = {}
    for name, (fields, compare) in checks.items():
        paths = [getattr(references, field) for field in fields]
        missing = [field for field, path in zip(fields, paths)
                   if not path or not Path(path).is_file()]
        if missing:
            logger.info("Skipping TCGA %s comparison; unavailable: %s", name, ", ".join(missing))
            statuses[name] = {"status": "skipped", "missing_inputs": missing}
            continue
        try:
            summary = compare(output_dir, *(Path(path) for path in paths))
            statuses[name] = {"status": "complete", **summary}
        except Exception as exc:
            logger.exception("TCGA %s comparison failed", name)
            statuses[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    return statuses


def predict_tcga_test_set(predictor, test_config, output_dir):
    """Predict one configured TCGA test set and record optional comparisons."""
    from .inference import _StagedInferenceDataset, _infer_result_set_name, _validate_input_file_path

    source = _validate_input_file_path(test_config.test_set_file_path, context="TCGA test set")
    result_root = Path(output_dir) / _infer_result_set_name(source)
    result_dir = result_root / "tcga"
    protected = [source]
    protected.extend(
        Path(value) for value in test_config.tcga_comparisons.model_dump().values()
        if value and str(value).strip()
    )
    resolved = result_dir.resolve()
    for path in protected:
        input_path = Path(path).expanduser().resolve()
        if input_path == resolved or resolved in input_path.parents:
            raise ValueError(f"TCGA input {input_path} overlaps output directory {resolved}")
    if result_dir.is_symlink() or (
        result_dir.exists() and (not result_dir.is_dir() or any(result_dir.iterdir()))
    ):
        raise FileExistsError(f"Refusing to reuse nonempty TCGA output directory: {result_dir}")

    predictor._ensure_model_loaded()
    model_dir = Path(predictor.model_dir)
    gene_path = model_dir / "input_gene_list.txt"
    cell_type_path = model_dir / "cell_type_list.txt"
    for path in (gene_path, cell_type_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required TCGA model artifact is missing: {path}")
    predictor.config.model.input_gene_list_fp = gene_path
    predictor.config.model.cell_type_fp = cell_type_path
    marker_path = model_dir / "model_kind.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.is_file() else {}
    if marker.get("normalize_training_intersection") is False:
        raise ValueError("Checkpoint is incompatible with the TCGA TPM normalization contract")
    contract_path = model_dir / "input_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8")).get("input_contract") \
        if contract_path.is_file() else None
    genes = gene_path.read_text(encoding="utf-8").splitlines()
    with Path(source).open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    missing_genes = [gene for gene in genes if gene not in set(header[1:])]

    config = predictor._build_gepdataset_config(
        data_file_path=str(source), dataset_type="tcga",
        require_model_artifacts=True, staged_contract=True,
    ).model_copy(update={
        "scaling_by_constant": predictor.model.data_config.scaling_by_constant,
        "scaling_factor": predictor.model.data_config.scaling_factor,
        "persist_processed_data": False,
        "processed_data_dir": None,
        "chunk_size": min(predictor.config.data.chunk_size, predictor.config.evaluation.val_batch_size),
    })
    is_predictor_only = isinstance(predictor.model, CellPropOnlyModel)
    dataset = _StagedInferenceDataset(
        config, result_dir, linear_tpm=True,
        legacy_round=not is_predictor_only and contract != "staged_tpm_v1",
    )
    if is_predictor_only:
        cell_types = Path(predictor.config.model.cell_type_fp).read_text(
            encoding="utf-8"
        ).splitlines()
        if not predictor.model.model_config.deside_fill_cancer_residual:
            cancer_index = predictor.model.cell_prop_predictor.cancer_cell_type_index
            cell_types = [name for i, name in enumerate(cell_types) if i != cancer_index]
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(dataset), predictor.config.evaluation.val_batch_size):
                values = dataset.data[start:start + predictor.config.evaluation.val_batch_size]
                predictions.append(_predict_prepared_batch(predictor.model, values, predictor.device))
        values = np.concatenate(predictions)
        if not np.isfinite(values).all():
            raise RuntimeError("TCGA predictions contain non-finite values")
        prediction_path = result_dir / "predicted_cell_prop.csv"
        pd.DataFrame(values, index=dataset.sample_ids, columns=cell_types).to_csv(
            prediction_path, float_format="%g"
        )
        result = {
            "pred_cell_prop_file_path": str(prediction_path),
            "test_set_result_dir": str(result_dir),
            "cell_types": cell_types,
        }
    else:
        devices = [torch.device(predictor.device).index or 0] if str(predictor.device).startswith("cuda") else []
        mps_state = torch.mps.get_rng_state() if str(predictor.device) == "mps" else None
        try:
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(42)
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(42)
                if mps_state is not None:
                    torch.mps.manual_seed(42)
                result = evaluate_model(
                    trained_model=predictor.model, test_set=dataset, result_dir=str(result_root),
                    model_config=predictor.config.model, output_dir=str(model_dir),
                    device=predictor.device, val_batch_size=predictor.config.evaluation.val_batch_size,
                    save_reconstructed_geps=True, dataset_type="tcga", result_set_name="tcga",
                )
        finally:
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)
        prediction_path = Path(result["pred_cell_prop_file_path"])
        values = pd.read_csv(prediction_path, index_col=0).to_numpy(dtype=float)
        if values.shape != (len(dataset), len(result["cell_types"])) or not np.isfinite(values).all():
            raise RuntimeError("TCGA predictions have an unexpected shape or non-finite values")

    gep_paths = [
        result_dir / "gep" / f"reconstructed_gep_{cell_type}_log2p1.csv"
        for cell_type in result["cell_types"]
    ] if not is_predictor_only else []
    for path in gep_paths:
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"Missing TCGA cell-type GEP output: {path}")
        if pd.read_csv(path, nrows=0, index_col=0).columns.tolist() != genes:
            raise RuntimeError(f"TCGA GEP gene order differs from the model: {path}")

    exclusions_path = result_dir / "excluded_samples.csv"
    exclusions = pd.read_csv(exclusions_path)
    checkpoint = _resolve_trained_checkpoint_path(model_dir, predictor.config.training)
    manifest = {
        "status": "comparisons_pending",
        "model_kind": "cell_prop_only" if is_predictor_only else "full_vae",
        "model_dir": str(model_dir.resolve()),
        "checkpoint": str(checkpoint),
        "input_file": str(Path(source).resolve()),
        "input_contract": contract or "legacy",
        "sample_count": len(dataset),
        "excluded_sample_count": len(exclusions),
        "gene_count": len(genes),
        "cell_types": result["cell_types"],
        "missing_model_genes_zero_filled": missing_genes,
        "cell_proportion_file": str(prediction_path),
        "excluded_samples_file": str(exclusions_path),
        "cell_type_gep_files": [str(path) for path in gep_paths],
    }
    manifest_path = result_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["comparisons"] = _comparison_status(result_dir, test_config.tcga_comparisons)
    manifest["status"] = "comparison_failed" if any(
        item["status"] == "failed" for item in manifest["comparisons"].values()
    ) else "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result["run_manifest"] = str(manifest_path)
    result["comparisons"] = manifest["comparisons"]
    return result
