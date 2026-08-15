"""
Inference pipeline for VAEDecon
"""
import os
import gc
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import torch

from ..data import GEPDataset, find_sct_gep_of_bulk_sample
from ..utility import check_dir
from ..workflow import (
    load_trained_model,
    evaluate_model,
    _apply_training_config_override,
    _resolve_trained_checkpoint_path,
)
from ..configs.default_config import VAEDeconConfig, GEPDatasetConfig, TestSetConfig, TrainingConfig

logger = logging.getLogger(__name__)


def _selected_sc_gep_cache_complete(
    *,
    sc_gep_result_dir: str | Path,
    cell_types: list[str],
    n_samples: int,
    selected_sample2cell_id_file_path: str | Path,
) -> bool:
    """Return True only when the selected-sample manifest and all expected
    ground-truth SCT GEP CSVs exist for the current plotting request.
    """
    selected_fp = Path(str(selected_sample2cell_id_file_path))
    if not selected_fp.exists():
        return False

    result_dir = Path(str(sc_gep_result_dir))
    for cell_type in cell_types:
        expected_fp = result_dir / f"sct_gep_{cell_type}_from_{n_samples}_bulksamples.csv"
        if not expected_fp.exists():
            return False
    return True

def _cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.tensor([0.0], device="cuda")
        (x + 1).sum().item()
        return True
    except Exception:
        return False


def _infer_result_set_name(data_file_path: str | Path) -> str:
    """Use the input file stem as the inference result subfolder name."""
    stem = Path(str(data_file_path)).stem.strip()
    return stem or "test_set"


def _build_simutme_path_suggestions(requested_path: str | Path, limit: int = 5) -> list[str]:
    """Suggest nearby SimuTME-style files/dirs when the configured path is missing.

    Helps catch post-regression mismatches such as ``11ds`` vs ``12ds`` or
    ``n_base30`` vs ``n_base100`` in generated dataset folder names.
    """
    requested = Path(str(requested_path))
    suggestions: list[str] = []

    parent = requested.parent
    if parent.exists() and parent.is_dir():
        try:
            for child in sorted(parent.iterdir()):
                if child.is_dir():
                    continue
                if child.suffix.lower() in {".h5ad", ".csv"}:
                    suggestions.append(str(child))
                    if len(suggestions) >= limit:
                        break
        except OSError:
            pass
    if suggestions:
        return suggestions[:limit]

    ancestor = parent
    visited: set[Path] = set()
    while ancestor != ancestor.parent and len(suggestions) < limit:
        if ancestor in visited:
            break
        visited.add(ancestor)
        if ancestor.exists() and ancestor.is_dir():
            try:
                for child in sorted(ancestor.iterdir()):
                    if child.is_dir() or child.suffix.lower() in {".h5ad", ".csv"}:
                        suggestions.append(str(child))
                        if len(suggestions) >= limit:
                            break
            except OSError:
                pass
        if len(suggestions) >= limit:
            break
        ancestor = ancestor.parent

    return suggestions[:limit]


def _validate_input_file_path(
    requested_path: str | Path,
    context: str = "test set",
) -> Path:
    """Validate a VAEDecon input file path exists and return it as a Path.

    Raises ``FileNotFoundError`` with a diagnostic message including nearby
    candidate files/dirs if the configured path is missing.
    """
    path = Path(str(requested_path))
    if path.exists() and path.is_file():
        return path

    suggestions = _build_simutme_path_suggestions(path)
    msg_lines = [
        f"Configured {context} file does not exist:",
        f"  requested: {path}",
    ]
    if not path.is_absolute():
        try:
            resolved = path.resolve()
            msg_lines.append(f"  resolved (cwd): {resolved}")
        except Exception:
            pass
    if suggestions:
        msg_lines.append("Nearby candidate files/dirs you may have intended:")
        for s in suggestions:
            msg_lines.append(f"    - {s}")
    else:
        msg_lines.append(
            "No nearby .h5ad/.csv candidates were found. Double-check the configured "
            "path and confirm SimuTME dataset generation finished successfully before "
            "running VAEDecon inference."
        )
    raise FileNotFoundError("\n".join(msg_lines))


def _resolve_model_artifact_path(
    model_dir: str | Path,
    configured_path: Optional[str | Path],
    default_file_name: str,
) -> Optional[Path]:
    """Resolve a trained-model artifact path from config or model_dir."""
    base_dir = Path(str(model_dir))
    if configured_path is not None and str(configured_path).strip() != "":
        configured = Path(str(configured_path))
        if configured.exists():
            return configured
        candidate = base_dir / configured.name
        if candidate.exists():
            return candidate

    default_candidate = base_dir / default_file_name
    if default_candidate.exists():
        return default_candidate
    return None


def _validate_required_model_artifact_path(
    model_dir: str | Path,
    configured_path: Optional[str | Path],
    default_file_name: str,
    label: str,
) -> Path:
    """Resolve a required trained-model artifact path or raise a clear error."""
    resolved = _resolve_model_artifact_path(
        model_dir=model_dir,
        configured_path=configured_path,
        default_file_name=default_file_name,
    )
    if resolved is not None:
        return resolved

    base_dir = Path(str(model_dir))
    msg_lines = [
        f"Required trained-model artifact is missing for inference: {label}",
        f"  model_dir: {base_dir}",
        f"  configured path: {configured_path}",
        f"  expected default file: {base_dir / default_file_name}",
        "Inference needs the exact training-time model artifacts saved under final_model/.",
        "Re-run training metadata export or point the config at an existing trained-model artifact.",
    ]
    raise FileNotFoundError("\n".join(msg_lines))


def _configured_test_sets(config: VAEDeconConfig) -> Dict[str, TestSetConfig]:
    """Return configured test sets from the loaded config."""
    return dict(getattr(config.data, "test_sets", {}) or {})


def _align_cell_prop_tables(
        true_cell_prop: pd.DataFrame,
        pred_cell_prop: pd.DataFrame,
        cell_types: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align true and predicted cell proportions to shared sample IDs and cell types."""
    shared_columns = [cell_type for cell_type in cell_types
                      if cell_type in true_cell_prop.columns and cell_type in pred_cell_prop.columns]
    shared_sample_ids = [sample_id for sample_id in pred_cell_prop.index if sample_id in true_cell_prop.index]
    if not shared_columns:
        raise ValueError("No shared cell-type columns found between true and predicted cell proportions.")
    if not shared_sample_ids:
        raise ValueError("No shared sample IDs found between true and predicted cell proportions.")
    aligned_true = true_cell_prop.loc[shared_sample_ids, shared_columns].copy()
    aligned_pred = pred_cell_prop.loc[shared_sample_ids, shared_columns].copy()
    return aligned_true, aligned_pred


def _merge_cell_prop_tables(
        true_cell_prop: pd.DataFrame,
        pred_cell_prop: pd.DataFrame,
) -> pd.DataFrame:
    """Merge aligned true and predicted cell proportions into one comparison table."""
    return pd.concat(
        [
            true_cell_prop.add_prefix("true_"),
            pred_cell_prop.add_prefix("pred_"),
        ],
        axis=1,
    )


def _merge_aligned_cell_prop_long_table(
        true_cell_prop: pd.DataFrame,
        pred_cell_prop: pd.DataFrame,
) -> pd.DataFrame:
    """Convert aligned true/predicted cell proportions into one long table."""
    rows = []
    for sample_id in true_cell_prop.index.tolist():
        for cell_type in true_cell_prop.columns.tolist():
            rows.append(
                {
                    "sample_id": sample_id,
                    "cell_type": cell_type,
                    "true_cell_prop": float(true_cell_prop.at[sample_id, cell_type]),
                    "pred_cell_prop": float(pred_cell_prop.at[sample_id, cell_type]),
                }
            )
    return pd.DataFrame(rows, columns=["sample_id", "cell_type", "true_cell_prop", "pred_cell_prop"])


def _save_cell_prop_comparison_table(
        true_cell_prop: pd.DataFrame,
        pred_cell_prop_file_path: str,
        cell_prop_result_dir: str,
        cell_types: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save one merged comparison table for the aligned true and predicted cell proportions."""
    pred_cell_prop = pd.read_csv(pred_cell_prop_file_path, index_col=0)
    aligned_true, aligned_pred = _align_cell_prop_tables(
        true_cell_prop=true_cell_prop,
        pred_cell_prop=pred_cell_prop,
        cell_types=cell_types,
    )
    merged_cell_prop = _merge_cell_prop_tables(aligned_true, aligned_pred)
    merged_cell_prop.to_csv(os.path.join(cell_prop_result_dir, "cell_prop_comparison.csv"))
    return aligned_true, aligned_pred


def _save_selected_sample_cell_props(
        true_cell_prop: pd.DataFrame,
        pred_cell_prop_file_path: str,
        selected_sample2cell_id_file_path: str,
        sc_gep_result_dir: str,
        cell_types: list[str],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Save aligned true and predicted cell proportions for the selected scGEP samples."""
    if (not pred_cell_prop_file_path or not selected_sample2cell_id_file_path
            or not os.path.exists(pred_cell_prop_file_path)
            or not os.path.exists(selected_sample2cell_id_file_path)):
        return None, None

    if not isinstance(true_cell_prop, pd.DataFrame) or true_cell_prop.empty:
        return None, None

    pred_cell_prop = pd.read_csv(pred_cell_prop_file_path, index_col=0)
    selected_sample2cell_id = pd.read_csv(selected_sample2cell_id_file_path, index_col=0)
    selected_sample_ids = [sample_id for sample_id in selected_sample2cell_id.index.drop_duplicates().tolist()
                           if sample_id in pred_cell_prop.index and sample_id in true_cell_prop.index]
    if not selected_sample_ids:
        return None, None

    selected_true = true_cell_prop.loc[selected_sample_ids, :].copy()
    selected_pred = pred_cell_prop.loc[selected_sample_ids, :].copy()
    aligned_true, aligned_pred = _align_cell_prop_tables(
        true_cell_prop=selected_true,
        pred_cell_prop=selected_pred,
        cell_types=cell_types,
    )
    aligned_true.to_csv(os.path.join(sc_gep_result_dir, "selected_samples_true_cell_prop.csv"))
    aligned_pred.to_csv(os.path.join(sc_gep_result_dir, "selected_samples_predicted_cell_prop.csv"))
    _merge_aligned_cell_prop_long_table(
        true_cell_prop=aligned_true,
        pred_cell_prop=aligned_pred,
    ).to_csv(
        os.path.join(sc_gep_result_dir, "selected_samples_cell_prop_long.csv"),
        index=False,
        float_format="%.6f",
    )
    return aligned_true, aligned_pred


class VAEDeconPredictor:
    """Predictor for VAEDecon model"""

    def __init__(
            self,
            model_dir: str,
            config: Optional[VAEDeconConfig] = None,
            device: str = 'auto'
    ):
        """
        Initializes the predictor

        Parameters:
            model_dir: The directory of the trained model
            config: VAEDecon Config object
            device: ('auto', 'cuda', 'cpu')
        """
        self.model_dir = model_dir
        self.config = config or VAEDeconConfig()
        self.save_reconstructed_gep = self.config.evaluation.save_reconstructed_gep
        self.figure_format = self.config.evaluation.figure_format

        # Device setup
        if device == 'auto':
            if _cuda_usable():
                self.device = 'cuda'
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            if self.device == "cuda" and not _cuda_usable():
                raise RuntimeError(
                    "Requested device 'cuda' but CUDA kernels cannot run on this machine. "
                    "This commonly indicates a GPU compute capability mismatch with the installed PyTorch CUDA build. "
                    "Use device='cpu' or install a compatible PyTorch CUDA build for your GPU."
                )

        logger.info(f"Using device: {self.device}")

        self._hydrate_model_config_paths()
        self.model = None

        # Allow config-only inspection helpers to work even when a checkpoint
        # has not been materialized in model_dir yet.
        if any(Path(str(self.model_dir)).glob("*.ckpt")):
            self._load_model()
        else:
            logger.warning(
                "No checkpoint file was found under %s during predictor initialization. "
                "Model loading will be deferred until prediction is requested.",
                self.model_dir,
            )

    def _hydrate_model_config_paths(self):
        """Backfill trained-model artifact paths from model_dir for inference."""
        model_dir = Path(str(self.model_dir))
        if getattr(self.config.model, "model_dir", None) in (None, ""):
            self.config.model.model_dir = model_dir

        input_gene_list_fp = _resolve_model_artifact_path(
            model_dir=model_dir,
            configured_path=getattr(self.config.model, "input_gene_list_fp", None),
            default_file_name="input_gene_list.txt",
        )
        if input_gene_list_fp is not None:
            self.config.model.input_gene_list_fp = input_gene_list_fp

        cell_type_fp = _resolve_model_artifact_path(
            model_dir=model_dir,
            configured_path=getattr(self.config.model, "cell_type_fp", None),
            default_file_name="cell_type_list.txt",
        )
        if cell_type_fp is not None:
            self.config.model.cell_type_fp = cell_type_fp

        gene_mean_std_fp = getattr(self.config.model, "gene_mean_std_fp", None)
        if gene_mean_std_fp is not None and str(gene_mean_std_fp).strip() != "":
            resolved_gene_mean_std_fp = _resolve_model_artifact_path(
                model_dir=model_dir,
                configured_path=gene_mean_std_fp,
                default_file_name=Path(str(gene_mean_std_fp)).name,
            )
            if resolved_gene_mean_std_fp is not None:
                self.config.model.gene_mean_std_fp = resolved_gene_mean_std_fp

    def _load_model(self):
        """Load the trained model"""
        checkpoint_files = list(Path(str(self.model_dir)).glob("*.ckpt"))
        if not checkpoint_files:
            raise FileNotFoundError(
                f"No .ckpt checkpoint was found under model_dir: {self.model_dir}"
            )
        logger.info(f"Loading model from: {self.model_dir}")
        training_config_override = getattr(self.config, "training", None)
        model_file_path = _resolve_trained_checkpoint_path(
            model_dir=self.model_dir,
            training_config=_apply_training_config_override(
                saved_training_config=TrainingConfig.from_json_file(
                    Path(str(self.model_dir)) / "training_config.json"
                ),
                training_config_override=training_config_override,
            ),
        )
        logger.info("Loading checkpoint: %s", model_file_path)
        self.model = load_trained_model(
            model_dir=self.model_dir,
            training_config_override=training_config_override,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully!")

    def _ensure_model_loaded(self):
        """Load the trained model on demand before running prediction."""
        if self.model is None:
            self._load_model()

    def predict(
            self,
            data_file_path: str,
            output_dir: Optional[str] = None,
            pred_cell_prop_file_path: Optional[str] = None,
            dataset_type: str = 'test'
    ) -> Dict[str, Any]:
        """
        Predict on new data

        Args:
            data_file_path: Input data file path
            output_dir: Output directory
            pred_cell_prop_file_path: Predicted cell proportion file path (for comparison)
            dataset_type: Dataset type ('test', 'tcga', etc.)

        Returns:
            Dictionary containing prediction results
        """
        # Set output directory
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(self.model_dir),
                f'{dataset_type}_results'
            )
        check_dir(Path(output_dir))

        logger.info(f"Processing data from: {data_file_path}")
        data_file_path = str(_validate_input_file_path(data_file_path, context="test set data file"))
        result_set_name = _infer_result_set_name(data_file_path)
        logger.info(f"Inference result subfolder: {result_set_name}")

        gep_dataset_config = self._build_gepdataset_config(
            data_file_path=data_file_path,
            dataset_type=dataset_type,
            require_model_artifacts=True,
        )
        dataset = GEPDataset(config=gep_dataset_config)

        logger.info(f"Dataset shape: {dataset.data.shape}")

        logger.info("Running inference...")
        self._ensure_model_loaded()
        val_batch_size = self.config.evaluation.val_batch_size
        results = evaluate_model(
            trained_model=self.model,
            test_set=dataset,
            result_dir=output_dir,
            output_dir=self.model_dir,
            device=self.device,
            pred_cell_prop_file_path=pred_cell_prop_file_path,
            model_config=self.config.model,
            val_batch_size=val_batch_size,
            save_reconstructed_geps=self.save_reconstructed_gep,
            dataset_type=dataset_type,
            result_set_name=result_set_name,
        )

        logger.info("Inference completed!")

        return results

    def _build_gepdataset_config(
        self,
        data_file_path: str | Path,
        dataset_type: str = 'test',
        require_model_artifacts: bool = False,
    ) -> GEPDatasetConfig:
        """
        Build a config dict for GEPDataset from self.config.data
        """
        # Prepare dataset
        processed_data_dir = os.path.join(
            os.path.dirname(data_file_path),
            f'processed_{dataset_type}'
        )
        if require_model_artifacts:
            input_gene_list_fp = _validate_required_model_artifact_path(
                model_dir=self.model_dir,
                configured_path=getattr(self.config.model, "input_gene_list_fp", None),
                default_file_name="input_gene_list.txt",
                label="input gene list",
            )
        else:
            input_gene_list_fp = _resolve_model_artifact_path(
                model_dir=self.model_dir,
                configured_path=getattr(self.config.model, "input_gene_list_fp", None),
                default_file_name="input_gene_list.txt",
            )

        def _optional_path(path_value: Optional[str | Path]) -> Optional[Path]:
            if path_value is None or str(path_value).strip() == "":
                return None
            return Path(str(path_value))

        return GEPDatasetConfig(
            file_paths=[data_file_path],
            scaling_by_constant=self.config.data.scaling_by_constant,
            remove_low_var_genes=self.config.evaluation.remove_low_var_genes,
            force_reprocess=self.config.data.force_reprocess,
            use_memmap=self.config.data.use_memmap,
            chunk_size=self.config.data.chunk_size,
            min_var=self.config.data.min_var,
            scaling_factor=self.config.data.scaling_factor,
            gene_mean_std_source=self.config.data.gene_mean_std_source,
            gene_mean_std_sct_gep_file_path=_optional_path(self.config.data.gene_mean_std_sct_gep_file_path),
            sct_gep_file_path=_optional_path(self.config.data.sct_gep_file_path),
            pooled_sc_h5ad_path=_optional_path(self.config.data.pooled_sc_h5ad_path),
            pooled_sc_cell_type_col=self.config.data.pooled_sc_cell_type_col,
            pooled_sc_cell_subtype_col=self.config.data.pooled_sc_cell_subtype_col,
            pooled_sc_sample_size=self.config.data.pooled_sc_sample_size,
            pooled_sc_seed=self.config.data.pooled_sc_seed,
            processed_data_dir=processed_data_dir,
            gene_list_file=input_gene_list_fp,
        )

    def predict_and_visualize(
            self,
            data_file_path: str,
            output_dir: Optional[str] = None,
            pred_cell_prop_file_path: Optional[str] = None,
            sample2cell_id_file_path: Optional[str] = None,
            sct_gep_file_path: Optional[str] = None,
            # batch_size: int = 1024,
    ) -> Dict[str, Any]:
        """
        Predict and visualize results

        Parameters:
            data_file_path: File path of input data to be predicted
            output_dir: Output directory
            pred_cell_prop_file_path: Predicted cell proportion file path
            sample2cell_id_file_path: Sample to cell ID mapping file during simulation
            sct_gep_file_path: Single-cell gene expression profile file
            batch_size: Batch size for prediction

        Returns:
            Prediction results in a dictionary
        """

        results = self.predict(
            data_file_path=data_file_path,
            output_dir=output_dir,
            pred_cell_prop_file_path=pred_cell_prop_file_path,
            dataset_type='test'
        )

        # Visualizations
        if sample2cell_id_file_path is None or sample2cell_id_file_path == '':
            sample2cell_id_file_path = self.config.data.test_set_sample2cell_id_file_path
        if sct_gep_file_path is None or sct_gep_file_path == '':
            sct_gep_file_path = self.config.data.sct_gep_file_path
        validated_sample2cell: Optional[str] = None
        if sample2cell_id_file_path and str(sample2cell_id_file_path).strip() != '':
            try:
                validated_sample2cell = str(_validate_input_file_path(
                    sample2cell_id_file_path,
                    context="sample-to-cell mapping file",
                ))
            except FileNotFoundError as exc:
                logger.warning(
                    "Optional sample-to-cell mapping file missing; skipping any "
                    "visualizations that depend on it. Details: %s", exc
                )
        validated_sct_gep: Optional[str] = None
        if sct_gep_file_path and str(sct_gep_file_path).strip() != '':
            try:
                validated_sct_gep = str(_validate_input_file_path(
                    sct_gep_file_path,
                    context="SCT GEP reference file",
                ))
            except FileNotFoundError as exc:
                logger.warning(
                    "Optional SCT GEP reference file missing; skipping any "
                    "visualizations that depend on it. Details: %s", exc
                )
        self._generate_visualizations(
            results=results,
            sample2cell_id_file_path=validated_sample2cell,
            sct_gep_file_path=validated_sct_gep,
        )

        return results

    def predict_configured_test_sets(
            self,
            output_dir: Optional[str] = None,
            dataset_type: str = 'test',
            visualize: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Run inference for all configured test sets in the config."""
        configured_sets = _configured_test_sets(self.config)
        if not configured_sets:
            raise ValueError(
                "No configured test sets were found in config.data.test_sets, "
                "and no legacy test_set_file_path was available."
            )

        all_results: Dict[str, Dict[str, Any]] = {}
        for test_name, test_cfg in configured_sets.items():
            logger.info("Running inference for configured test set: %s", test_name)
            if visualize:
                result = self.predict_and_visualize(
                    data_file_path=str(test_cfg.test_set_file_path),
                    output_dir=output_dir,
                    sample2cell_id_file_path=(
                        str(test_cfg.test_set_sample2cell_id_file_path)
                        if test_cfg.test_set_sample2cell_id_file_path else None
                    ),
                    sct_gep_file_path=(
                        str(test_cfg.sct_gep_file_path)
                        if test_cfg.sct_gep_file_path else None
                    ),
                )
            else:
                result = self.predict(
                    data_file_path=str(test_cfg.test_set_file_path),
                    output_dir=output_dir,
                    dataset_type=dataset_type,
                )
            all_results[test_name] = result
        return all_results

    def _generate_visualizations(
            self,
            results: Dict[str, Any],
            sample2cell_id_file_path: Optional[str] = None,
            sct_gep_file_path: Optional[str] = None,
    ):
        """Generate visualizations based on prediction results"""
        logger.info("Generating visualizations...")

        try:
            from ..plot import (
                compare_y_y_pred_subplot,
                plot_single_cell_gep,
                plot_bulk_gep,
                plot_latent_space,
            )
            from ..utility.evaluation import calculate_single_cell_gep_metrics_per_sample
        except Exception as e:
            logger.warning(f"Could not import plotting utilities (skipping visualizations): {e}")
            return

        true_cell_prop = results.get('true_cell_prop')
        pred_cell_prop_fp = results.get('pred_cell_prop_file_path')
        cell_types = results.get('cell_types')
        pred_a = results.get('pred_a')
        cell_prop_result_dir = results.get('cell_prop_result_dir')
        gep_result_dir = results.get('gep_result_dir')
        test_set_result_dir = results.get('test_set_result_dir')
        test_set = results.get('test_set')

        # 1. Plot cell proportions comparison
        if (
            self.config.evaluation.plot_cell_proportions
            and pred_cell_prop_fp
            and isinstance(true_cell_prop, pd.DataFrame)
            and not true_cell_prop.empty
        ):
            logger.info("Plotting cell proportions...")
            aligned_true_cell_prop, aligned_pred_cell_prop = _save_cell_prop_comparison_table(
                true_cell_prop=true_cell_prop,
                pred_cell_prop_file_path=pred_cell_prop_fp,
                cell_prop_result_dir=cell_prop_result_dir,
                cell_types=cell_types,
            )
            _, _, metrics = compare_y_y_pred_subplot(
                y_true=aligned_true_cell_prop,
                y_pred=aligned_pred_cell_prop,
                show_columns=aligned_true_cell_prop.columns.tolist(),
                result_file_dir=cell_prop_result_dir,
                dataset_name='VAEDecon',
                show_metrics=self.config.evaluation.show_metrics,
                x_label='Predicted cell proportion',
                y_label='True cell proportion',
                figsize=self.config.evaluation.figsize,
                figure_format=self.figure_format,
                return_metrics=True,
                show_legend=True,
                collapse_columns=False,
            )
            pd.DataFrame([metrics]).to_csv(
                os.path.join(cell_prop_result_dir, "prediction_metrics.csv"),
                index=False,
            )
        elif self.config.evaluation.plot_cell_proportions and pred_cell_prop_fp:
            logger.info("Skipping cell proportion comparison plot because no true cell fractions are available.")

        # 2. Plot single-cell gene expression profiles, comparing purified cell-type-specific GEPs with original sctGEPs
        if (self.config.evaluation.plot_single_cell_gep and
                sample2cell_id_file_path and sct_gep_file_path):
            logger.info("Plotting single cell GEP...")
            sc_gep_result_dir = os.path.join(gep_result_dir, "sc_gep")
            check_dir(Path(sc_gep_result_dir))

            selected_sample2cell_id_fp = os.path.join(
                sc_gep_result_dir,
                f"selected_{self.config.evaluation.n_samples}_samples2sct_ids.csv"
            )

            if not _selected_sc_gep_cache_complete(
                sc_gep_result_dir=sc_gep_result_dir,
                cell_types=cell_types,
                n_samples=self.config.evaluation.n_samples,
                selected_sample2cell_id_file_path=selected_sample2cell_id_fp,
            ):
                if os.path.exists(selected_sample2cell_id_fp):
                    logger.warning(
                        "Found selected sample mapping file but one or more "
                        "ground-truth SCT GEP CSVs are missing in %s. "
                        "Regenerating the selected-sample SCT cache.",
                        sc_gep_result_dir,
                    )
                find_sct_gep_of_bulk_sample(
                    sct_gep_dataset_file_path=sct_gep_file_path,
                    result_dir=sc_gep_result_dir,
                    sample2cell_id_file_path=sample2cell_id_file_path,
                    bulk_dataset=test_set,
                    cell_types=cell_types,
                    n_samples=self.config.evaluation.n_samples,
                    random_seed=42,
                    selected_sample2cell_id_file_path=selected_sample2cell_id_fp
                )
            selected_true_cell_prop, _ = _save_selected_sample_cell_props(
                true_cell_prop=true_cell_prop,
                pred_cell_prop_file_path=pred_cell_prop_fp,
                selected_sample2cell_id_file_path=selected_sample2cell_id_fp,
                sc_gep_result_dir=sc_gep_result_dir,
                cell_types=cell_types,
            )
            plot_single_cell_gep(
                test_set=test_set,
                cell_types=cell_types,
                pred_a=pred_a,
                figure_format=self.figure_format,
                sc_gep_result_dir=sc_gep_result_dir,
                n_samples=self.config.evaluation.n_samples,
                max_visualize_samples=self.config.evaluation.visualize_n_sample,
                selected_sample2cell_id_file_path=selected_sample2cell_id_fp,
                sct_gep_file_path=sct_gep_file_path,
                return_metrics=False,
                selected_true_cell_prop=selected_true_cell_prop,
                filtered_min_true_cell_prop=self.config.evaluation.cell_prop_threshold,
            )
            if getattr(self.config.evaluation, 'save_cell_type_specific_gep_metrics', False):
                metrics_df = calculate_single_cell_gep_metrics_per_sample(
                    sc_gep_result_dir=sc_gep_result_dir,
                    cell_types=cell_types,
                    n_samples=self.config.evaluation.n_samples,
                    selected_sample2cell_id_file_path=selected_sample2cell_id_fp,
                )
                metrics_df.to_csv(
                    os.path.join(sc_gep_result_dir, "cell_type_specific_gep_metrics.csv"),
                    index=False,
                )

        # 3. Plot latent space
        if self.config.evaluation.plot_latent_space:
            logger.info("Plotting latent space...")
            plot_latent_space(
                test_set=test_set,
                pred_a=pred_a,
                cell_types=cell_types,
                test_set_result_dir=test_set_result_dir,
                n_neighbors=self.config.evaluation.n_neighbors,
                min_dist=self.config.evaluation.min_dist,
                figure_format=self.figure_format,
            )

        # 4. Plot bulk gene expression profiles
        if self.config.evaluation.plot_bulk_gep and sample2cell_id_file_path:
            logger.info("Plotting bulk GEP...")
            sc_gep_result_dir = os.path.join(gep_result_dir, "sc_gep")
            selected_sample2cell_id_fp = os.path.join(
                sc_gep_result_dir,
                f"selected_{self.config.evaluation.n_samples}_samples2sct_ids.csv"
            )

            plot_bulk_gep(
                test_set=test_set,
                pred_a=pred_a,
                figure_format=self.figure_format,
                gep_result_dir=gep_result_dir,
                n_samples=self.config.evaluation.n_samples,
                selected_sample2cell_id_file_path=selected_sample2cell_id_fp,
                save_bulk_gep_input=self.config.evaluation.save_bulk_gep_input,
                save_recon_bulk_gep_conv=self.config.evaluation.save_recon_bulk_gep_conv,
            )

        logger.info("Visualizations completed!")


def predict_vaedecon(
        model_dir: str | Path,
        data_file_path: Optional[str | Path] = None,
        output_dir: Optional[str] = None,
        config: Optional[VAEDeconConfig] = None,
        device: str = 'auto',
        visualize: bool = True,
        **kwargs
) -> Dict[str, Any] | Dict[str, Dict[str, Any]]:
    """
    Run inference (prediction) using a trained VAEDecon model.

    This function loads a trained model and predicts cell type proportions for new bulk data.
    It can optionally generate visualizations of the results.

    Parameters:
        model_dir (str | Path):
            Directory containing the trained model (checkpoint and config).
        data_file_path (Optional[str | Path]):
            Path to one input bulk expression file (.h5ad or .csv). If omitted,
            VAEDecon runs inference for all configured entries under
            `config.data.test_sets`.
        output_dir (Optional[str]):
            Directory to save prediction results. Defaults to a 'test_results' folder
            relative to the model directory.
        config (Optional[VAEDeconConfig]):
            Configuration object override. Usually loaded automatically from `model_dir`.
        device (str):
            Device to run inference on ('auto', 'cuda', 'mps', 'cpu'). Defaults to 'auto'.
        visualize (bool):
            If True, generates plots for cell proportions, latent space, etc.
            Defaults to True.
        **kwargs:
            Additional arguments passed to the underlying predictor methods.
            Common kwargs include:
            - pred_cell_prop_file_path (str): Path to ground truth cell proportions (for evaluation).
            - sample2cell_id_file_path (str): Path to sample-to-cell mapping (for simulated data).
            - sct_gep_file_path (str): Path to single-cell GEP reference (for visualization).
            - save_reconstructed_geps (bool): Whether to save reconstructed expression profiles.
            - dataset_type (str): Label for the dataset (e.g., 'test', 'tcga').

    Returns:
        Dict[str, Any]:
            A dictionary containing prediction results and paths to saved files.
            Key fields include:
            - 'pred_cell_prop': DataFrame of predicted cell proportions.
            - 'pred_cell_prop_file_path': Path to the saved prediction CSV.
            - 'gep_result_dir': Directory where GEP results are saved.
            - 'cell_prop_result_dir': Directory where proportion results are saved.

    Examples:
        # 1. Simple prediction
        results = predict_vaedecon(
            model_dir='./output/model',
            data_file_path='./data/test.h5ad'
        )
        print(results['pred_cell_prop'].head())

        # 2. Prediction with visualization and ground truth comparison
        predict_vaedecon(
            model_dir='./output/model',
            data_file_path='./data/test.h5ad',
            visualize=True,
            pred_cell_prop_file_path='./data/true_proportions.csv'
        )
    """
    gc.collect()
    torch.cuda.empty_cache()
    predictor = VAEDeconPredictor(
        model_dir=model_dir,
        config=config,
        device=device
    )

    if data_file_path is None or str(data_file_path).strip() == '':
        return predictor.predict_configured_test_sets(
            output_dir=output_dir,
            visualize=visualize,
        )

    data_file_path = str(data_file_path)
    if visualize:
        return predictor.predict_and_visualize(
            data_file_path=data_file_path,
            output_dir=output_dir,
            **kwargs
        )
    else:
        return predictor.predict(
            data_file_path=data_file_path,
            output_dir=output_dir,
            **kwargs
        )


def predict_configured_test_sets(
        model_dir: str | Path,
        config: VAEDeconConfig,
        output_dir: Optional[str] = None,
        device: str = 'auto',
        visualize: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Run inference for all configured test sets in `config.data.test_sets`."""
    gc.collect()
    torch.cuda.empty_cache()
    predictor = VAEDeconPredictor(
        model_dir=model_dir,
        config=config,
        device=device,
    )
    return predictor.predict_configured_test_sets(
        output_dir=output_dir,
        visualize=visualize,
    )
