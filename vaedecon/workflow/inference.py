"""
Inference pipeline for VAEDecon
"""
import os
import gc
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch

from ..data import GEPDataset, find_sct_gep_of_bulk_sample
from ..utility import check_dir
from ..workflow import load_trained_model, evaluate_model
from ..configs.default_config import VAEDeconConfig, GEPDatasetConfig

logger = logging.getLogger(__name__)

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

        # Load the trained model
        self._load_model()

    def _load_model(self):
        """Load the trained model"""
        logger.info(f"Loading model from: {self.model_dir}")
        self.model = load_trained_model(model_dir=self.model_dir)
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully!")

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
        result_set_name = _infer_result_set_name(data_file_path)
        logger.info(f"Inference result subfolder: {result_set_name}")

        gep_dataset_config = self._build_gepdataset_config(
            data_file_path=data_file_path,
            dataset_type=dataset_type,
        )
        dataset = GEPDataset(config=gep_dataset_config)

        logger.info(f"Dataset shape: {dataset.data.shape}")

        logger.info("Running inference...")
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
    ) -> GEPDatasetConfig:
        """
        Build a config dict for GEPDataset from self.config.data
        """
        # Prepare dataset
        processed_data_dir = os.path.join(
            os.path.dirname(data_file_path),
            f'processed_{dataset_type}'
        )

        return GEPDatasetConfig(
            file_paths=[data_file_path],
            scaling_by_constant=self.config.data.scaling_by_constant,
            remove_low_var_genes=self.config.evaluation.remove_low_var_genes,
            force_reprocess=self.config.data.force_reprocess,
            use_memmap=self.config.data.use_memmap,
            chunk_size=self.config.data.chunk_size,
            scaling_factor=self.config.data.scaling_factor,
            processed_data_dir=processed_data_dir,
            gene_list_file=Path(self.config.model.input_gene_list_fp),
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
        self._generate_visualizations(
            results=results,
            sample2cell_id_file_path=sample2cell_id_file_path,
            sct_gep_file_path=sct_gep_file_path,
        )

        return results

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
                compare_y_y_pred_plot,
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
        if self.config.evaluation.plot_cell_proportions and pred_cell_prop_fp:
            logger.info("Plotting cell proportions...")
            compare_y_y_pred_plot(
                y_true=true_cell_prop,
                y_pred=pred_cell_prop_fp,
                show_columns=cell_types,
                result_file_dir=cell_prop_result_dir,
                model_name='VAEDecon',
                show_metrics=self.config.evaluation.show_metrics,
                y_label='y_pred',
                rasterized=self.config.evaluation.rasterized,
                figsize=self.config.evaluation.figsize,
                figure_format=self.figure_format,
            )

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

            if not os.path.exists(selected_sample2cell_id_fp):
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
            plot_single_cell_gep(
                test_set=test_set,
                cell_types=cell_types,
                pred_a=pred_a,
                figure_format=self.figure_format,
                sc_gep_result_dir=sc_gep_result_dir,
                n_samples=self.config.evaluation.n_samples,
                max_visualize_samples=3,
                selected_sample2cell_id_file_path=selected_sample2cell_id_fp,
                return_metrics=False,
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
        data_file_path: str | Path,
        output_dir: Optional[str] = None,
        config: Optional[VAEDeconConfig] = None,
        device: str = 'auto',
        visualize: bool = True,
        **kwargs
) -> Dict[str, Any]:
    """
    Run inference (prediction) using a trained VAEDecon model.

    This function loads a trained model and predicts cell type proportions for new bulk data.
    It can optionally generate visualizations of the results.

    Parameters:
        model_dir (str | Path):
            Directory containing the trained model (checkpoint and config).
        data_file_path (str | Path):
            Path to the input bulk expression file (.h5ad or .csv).
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
    data_file_path = str(data_file_path)
    predictor = VAEDeconPredictor(
        model_dir=model_dir,
        config=config,
        device=device
    )

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
