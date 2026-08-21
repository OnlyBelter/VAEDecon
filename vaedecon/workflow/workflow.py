import json
import os
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Dict, Any, Type, Union, List
import torch
from torch.utils.data import DataLoader

from ..utility import check_dir, non_log2log_cpm
from ..data import GEPDataset
from ..models import AutoModel, BaseAE
from ..models.base import BaseEncoder
from ..models.base import has_usable_labels
from ..models.gnn import EncoderSGNN
from ..models.nn import (EncoderMLP, DecoderMLP, DecoderConditionalMLP, EncoderHybrid, EncoderResMLP, DecoderResMLP,
                         PositionalEncoding, GeneTransformerEncoder, EncoderPathNet, DeSideCellPropPredictor)
from ..models.vae import VAE
from ..configs import ModelConfig, TrainingConfig, DataConfig
from ..trainers import BaseTrainerL, PLTrainer, TrainingPipeline

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def _cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.tensor([0.0], device="cuda")
        (x + 1).sum().item()
        return True
    except Exception:
        return False


def _cell_prop_batch_to_numpy(cell_prop: Union[torch.Tensor, np.ndarray, list]) -> np.ndarray:
    """Normalize one batch of cell proportions to a 2D numpy array."""
    if isinstance(cell_prop, torch.Tensor):
        array = cell_prop.detach().cpu().numpy()
    else:
        array = np.asarray(cell_prop)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def _detach_model_output_to_cpu(output: Dict[str, Any]) -> Dict[str, Any]:
    """Move tensor values in one model output dict to CPU before accumulation."""
    detached_output: Dict[str, Any] = {}
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            detached_output[key] = value.detach().cpu()
        else:
            detached_output[key] = value
    return detached_output


def create_model(
    model_config: ModelConfig,
    data_config: DataConfig,
    encoder_cls_name_list: List[str],
    decoder_cls: List[str],
    device: str = "auto",
) -> VAE:
    """Creates the VAE model."""
    position_encoding = PositionalEncoding(
        d_model=model_config.latent_dim,
        dropout=0,
        max_len=model_config.n_cell_types
    )
    encoders = []
    cell_prop_predictor = None
    kwargs: Dict[str, Any] = {
        "args": model_config,
        "data_config": data_config,
        "position_encoding": position_encoding,
    }

    if not isinstance(encoder_cls_name_list, list):
        raise TypeError("encoder_cls_name_list must be a list of encoder class names.")
    
    if not (1 <= len(encoder_cls_name_list) <= 4):
        raise ValueError("encoder_cls_name_list must contain between 1 and 4 encoder names.")
    
    if any(not isinstance(name, str) or not name.strip() for name in encoder_cls_name_list):
        raise ValueError("All encoder names in encoder_cls_name_list must be non-empty strings.")
    
    encoder_registry: Dict[str, Type[BaseEncoder]] = {
        "encodersgnn": EncoderSGNN,
        "encodermlp": EncoderMLP,
        "encoderresmlp": EncoderResMLP,
        "genetransformerencoder": GeneTransformerEncoder,
        "encoderpathnet": EncoderPathNet,
        "encoderhybrid": EncoderHybrid,
    }
    
    normalized_encoder_names = [name.strip().lower() for name in encoder_cls_name_list]
    unsupported = [name for name in normalized_encoder_names if name not in encoder_registry]
    if unsupported:
        raise NotImplementedError(f"Unsupported encoder class name(s): {unsupported}")
    
    encoder_aliases = list(getattr(model_config, "encoder_aliases", []) or [])
    if not encoder_aliases:
        encoder_aliases = [f"encoder_{idx}" for idx in range(len(normalized_encoder_names))]
    elif len(encoder_aliases) != len(normalized_encoder_names):
        raise ValueError(
            f"encoder_aliases (len={len(encoder_aliases)}) must match the number of encoders "
            f"({len(normalized_encoder_names)})."
        )

    for idx, encoder_name in enumerate(normalized_encoder_names):
        encoder_cls = encoder_registry[encoder_name]
        if encoder_name == "encoderhybrid":
            hybrid_kwargs = kwargs.copy()
            hybrid_kwargs["mlp_encoder"] = EncoderMLP(**kwargs)
            hybrid_kwargs["gnn_encoder"] = EncoderSGNN(**kwargs)
            encoder = EncoderHybrid(**hybrid_kwargs)
        else:
            encoder = encoder_cls(**kwargs)
        encoder.encoder_alias = encoder_aliases[idx]
        encoders.append(encoder)

    predictor_cls_name = str(getattr(model_config, "cell_prop_predictor_cls", "") or "").strip().lower()
    if predictor_cls_name:
        predictor_registry = {
            "desidecellproppredictor": DeSideCellPropPredictor,
        }
        if predictor_cls_name not in predictor_registry:
            raise NotImplementedError(f"Unsupported cell_prop_predictor_cls: {predictor_cls_name}")
        predictor_cls = predictor_registry[predictor_cls_name]
        cell_prop_predictor = predictor_cls(**kwargs)
        cell_prop_predictor.encoder_alias = getattr(model_config, "cell_prop_predictor_alias", "cell_prop_predictor")

    for decoder_cls_name in decoder_cls:
        decoder_cls_name = decoder_cls_name.lower()
        if decoder_cls_name == "DecoderMLP".lower():
            decoder_cls = DecoderMLP
        elif decoder_cls_name == "DecoderConditionalMLP".lower():
            decoder_cls = DecoderConditionalMLP
        elif decoder_cls_name == "DecoderResMLP".lower():
            decoder_cls = DecoderResMLP
        else:
            raise NotImplementedError(decoder_cls_name)
    decoder = decoder_cls(args=model_config)
    model = VAE(
        model_config=model_config,
        data_config=data_config,
        encoders=encoders,
        cell_prop_predictor=cell_prop_predictor,
        decoder=decoder,
    )

    if model_config.torch_compile:
        requested_device = (device or "auto").lower()
        effective_device = requested_device
        if effective_device == "auto":
            effective_device = "cuda" if _cuda_usable() else "cpu"

        if effective_device != "cuda":
            warnings.warn(
                f"torch.compile() requested but skipped on device '{effective_device}'. "
                "Compilation is only enabled for CUDA to avoid backend issues (e.g., MPS/Inductor)."
            )
        elif hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
            except RuntimeError as e:
                warnings.warn(
                    f"torch.compile() requested but failed: {e}. "
                    "Skipping compilation."
                )
        else:
            warnings.warn(
                "torch.compile() requested but torch version < 2.0. "
                "Skipping compilation. Upgrade PyTorch for faster training."
            )

    return model


def train_model(
        model: VAE,
        train_set: torch.utils.data.Dataset,
        val_set: torch.utils.data.Dataset,
        trainer_cls: Type[BaseTrainerL],
        training_config: TrainingConfig,
        data_config: DataConfig,
        device: str,
        result_dir: str,
) -> None:
    """Trains the model using the training pipeline."""
    if device == "cuda" and not _cuda_usable():
        raise RuntimeError(
            "Requested device 'cuda' but CUDA kernels cannot run on this machine. "
            "This commonly indicates an outdated NVIDIA driver or a GPU compute capability mismatch with the installed "
            "PyTorch CUDA build. Update your NVIDIA driver or install a compatible PyTorch build, or use device='cpu'."
        )
    training_pipeline = TrainingPipeline(
        training_config=training_config,
        model=model.to(device),
        data_config=data_config,
        trainer_cls=trainer_cls,
        result_dir=result_dir,
        debug_model=training_config.debug_model,
    )
    training_pipeline(train_data=train_set, eval_data=val_set)


def save_metadata(dataset: GEPDataset, model_config: ModelConfig) -> None:
    """Saves the gene list and cell type list."""
    dataset.save_gene_list(Path(model_config.input_gene_list_fp))
    dataset.save_cell_types(Path(model_config.cell_type_fp))


def _resolve_trained_checkpoint_path(
    model_dir: str | Path,
    training_config: TrainingConfig,
) -> Path:
    """Resolve which checkpoint file should be loaded for inference."""
    model_dir = Path(model_dir)
    checkpoint_files = sorted(model_dir.glob("*.ckpt"))
    selection = getattr(training_config, "saved_model_selection", "best")
    if not checkpoint_files:
        raise FileNotFoundError(f"No .ckpt checkpoint was found under model_dir: {model_dir}")
    if len(checkpoint_files) == 1:
        only_ckpt = checkpoint_files[0]
        if selection == "last" and only_ckpt.name != "last_model.ckpt":
            raise FileNotFoundError(
                "training.saved_model_selection='last' but only one checkpoint was found "
                f"and it is not last_model.ckpt: {only_ckpt}. This usually means the "
                "model directory was created before last-checkpoint saving was added, "
                "or training was skipped because an older checkpoint already existed."
            )
        return only_ckpt

    metadata_path = model_dir / "checkpoint_paths.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    if selection == "last":
        last_model_path = model_dir / "last_model.ckpt"
        metadata_last = metadata.get("last_model_path", "")
        if metadata_last:
            candidate = Path(metadata_last)
            if not candidate.is_absolute():
                candidate = model_dir / candidate
            if candidate.exists():
                return candidate
        if last_model_path.exists():
            return last_model_path
        raise FileNotFoundError(
            f"training.saved_model_selection='last' but last checkpoint was not found: {last_model_path}"
        )

    best_model_path = metadata.get("best_model_path", "")
    if best_model_path:
        best_candidate = Path(best_model_path)
        if not best_candidate.is_absolute():
            best_candidate = model_dir / best_candidate
        if best_candidate.exists():
            return best_candidate

    best_candidates = sorted(model_dir.glob("best_model_epoch=*.ckpt"))
    if len(best_candidates) == 1:
        return best_candidates[0]

    raise FileNotFoundError(
        "Could not resolve the requested best checkpoint under "
        f"{model_dir}. Found checkpoints: {[p.name for p in checkpoint_files]}"
    )


def _apply_training_config_override(
    saved_training_config: TrainingConfig,
    training_config_override: TrainingConfig | None,
) -> TrainingConfig:
    """Apply runtime overrides that should affect checkpoint selection."""
    if training_config_override is None:
        return saved_training_config
    override_selection = getattr(training_config_override, "saved_model_selection", None)
    if override_selection:
        saved_training_config.saved_model_selection = override_selection
    return saved_training_config


def load_trained_model(
    model_dir: str,
    training_config_override: TrainingConfig | None = None,
) -> Union[AutoModel, BaseAE]:
    """Loads the trained model from the specified directory."""
    checkpoint_files = list(Path(model_dir).glob("*.ckpt"))
    if not checkpoint_files:
        # if no .ckpt file found, load the model from the folder
        trained_model = AutoModel.load_from_folder(model_dir)
        return trained_model

    model_config_path = os.path.join(model_dir, "model_config.json")
    data_config_path = os.path.join(model_dir, "data_config.json")
    training_config_path = os.path.join(model_dir, "training_config.json")
    model_config = ModelConfig.from_json_file(model_config_path)
    training_config = TrainingConfig.from_json_file(training_config_path)
    training_config = _apply_training_config_override(
        saved_training_config=training_config,
        training_config_override=training_config_override,
    )
    data_config = DataConfig.from_json_file(data_config_path)
    model_file_path = _resolve_trained_checkpoint_path(
        model_dir=model_dir,
        training_config=training_config,
    )

    base_dir = Path(model_dir)
    if getattr(model_config, "input_gene_list_fp", None) is None or not Path(model_config.input_gene_list_fp).exists():
        candidate = base_dir / "input_gene_list.txt"
        if candidate.exists():
            model_config.input_gene_list_fp = candidate
    if getattr(model_config, "cell_type_fp", None) is None or not Path(model_config.cell_type_fp).exists():
        candidate = base_dir / "cell_type_list.txt"
        if candidate.exists():
            model_config.cell_type_fp = candidate
    if getattr(model_config, "gene_mean_std_fp", None) is not None and not Path(model_config.gene_mean_std_fp).exists():
        candidate = base_dir / Path(model_config.gene_mean_std_fp).name
        if candidate.exists():
            model_config.gene_mean_std_fp = candidate
    if getattr(data_config, "ppi_file_path", None) is not None and not Path(data_config.ppi_file_path).exists():
        candidate = base_dir / Path(data_config.ppi_file_path).name
        if candidate.exists():
            data_config.ppi_file_path = candidate
    if getattr(data_config, "pathway_file_path", None):
        fixed_paths = []
        for p in data_config.pathway_file_path:
            pth = Path(p)
            if pth.exists():
                fixed_paths.append(pth)
                continue
            candidate = base_dir / pth.name
            fixed_paths.append(candidate if candidate.exists() else pth)
        data_config.pathway_file_path = fixed_paths

    model = create_model(model_config=model_config,
                         data_config=data_config,
                         encoder_cls_name_list=model_config.encoders,
                         decoder_cls=model_config.decoders,
                         device="cpu")
    # Lightning checkpoints saved by this project may include objects such as
    # LazyLinear/UninitializedParameter metadata. PyTorch 2.6+ changed
    # torch.load(..., weights_only=True) to be the default, which rejects those
    # trusted local checkpoint objects during inference-time reload.
    checkpoint = torch.load(model_file_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    if any(k.startswith("model._orig_mod.") for k in state_dict.keys()):
        fixed = {}
        for k, v in state_dict.items():
            while k.startswith("model._orig_mod."):
                k = "model." + k[len("model._orig_mod."):]
            fixed[k] = v
        state_dict = fixed

    pl = PLTrainer(model=model, training_config=training_config)
    pl.load_state_dict(state_dict, strict=True)

    pl.eval()
    pl.freeze()
    return pl.model


def evaluate_model(
        trained_model: AutoModel,
        test_set: Union[GEPDataset | DataLoader],
        result_dir: str,
        model_config: ModelConfig,
        output_dir: str,
        device: str,
        pred_cell_prop_file_path: str = None,
        val_batch_size: int = None,
        save_reconstructed_geps: bool = False,
        dataset_type: str = 'training',  # or test, tcga
        result_set_name: str = "test_set",
) -> Dict[str, Any]:
    """Evaluates the trained model on the test set."""
    if device == "cuda" and not _cuda_usable():
        raise RuntimeError(
            "Requested device 'cuda' but CUDA kernels cannot run on this machine. "
            "Update your NVIDIA driver or install a compatible PyTorch build, or use device='cpu'."
        )
    test_set_result_dir = os.path.join(result_dir, result_set_name)
    check_dir(Path(test_set_result_dir))
    cell_prop_result_dir = os.path.join(test_set_result_dir, "cell_prop")
    gep_result_dir = os.path.join(test_set_result_dir, "gep")
    check_dir(Path(cell_prop_result_dir))
    check_dir(Path(gep_result_dir))

    if dataset_type != 'tcga':
        true_cell_prop = test_set.get_cell_prop()
    else:
        true_cell_prop = pd.DataFrame()
    cell_types = pd.read_csv(model_config.cell_type_fp, index_col=0, header=None).index.to_list()
    test_set_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False)
    if pred_cell_prop_file_path is not None and os.path.exists(pred_cell_prop_file_path):
        pred_cell_prop_all = pd.read_csv(pred_cell_prop_file_path, index_col=0)
        pred_cell_prop_all = pred_cell_prop_all.loc[:, cell_types].values
    else:
        pred_cell_prop_all = None
    pred_all_dict = {}
    pred_cell_prop_list = []
    with torch.no_grad():
        for batch in test_set_loader:
            # Move the batch to the correct device (GPU)
            if isinstance(batch, dict):
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            elif isinstance(batch, torch.Tensor):
                batch = batch.to(device)
            elif isinstance(batch, list):
                batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]

            pred_a = trained_model(batch)  # A ModelOutput including 11 elements
            pred_cell_prop = None
            if model_config.predict_cell_prop:
                pred_cell_prop = _cell_prop_batch_to_numpy(pred_a["pred_cell_prop"])
            else:
                labels = batch.get('labels') if isinstance(batch, dict) else None
                if has_usable_labels(labels):
                    pred_cell_prop = _cell_prop_batch_to_numpy(labels)

            pred_cell_prop_list.append(pred_cell_prop)
            pred_a = _detach_model_output_to_cpu(pred_a)
            for key, value in pred_a.items():
                if key not in pred_all_dict:
                    pred_all_dict[key] = []
                pred_all_dict[key].append(value)
    valid_pred_cell_props = [arr for arr in pred_cell_prop_list if arr is not None and arr.size > 0]
    if pred_cell_prop_all is None and valid_pred_cell_props:
        pred_cell_prop_all = np.concatenate(valid_pred_cell_props, axis=0)
    for key, value in pred_all_dict.items():
        if value[0] is not None and len(value[0].shape) > 0:
            pred_all_dict[key] = torch.cat(value, dim=0)

    sc_gep_result_dir = os.path.join(gep_result_dir, "sc_gep")
    check_dir(Path(sc_gep_result_dir))
    if "mu" in pred_all_dict and pred_all_dict["mu"] is not None:
        mu = pred_all_dict["mu"]
        if isinstance(mu, torch.Tensor):
            mu = mu.detach().cpu().numpy()
        if mu.ndim == 1:
            mu = mu.reshape(-1, 1)
        mu_columns = [f"mu_{i}" for i in range(mu.shape[1])]
        mu_df = pd.DataFrame(mu, index=test_set.get_sample_ids(), columns=mu_columns)
        mu_df.to_csv(os.path.join(sc_gep_result_dir, "mu_embeddings.csv"))
    if pred_cell_prop_all is not None:
        pred_cell_prop_df = pd.DataFrame(
            pred_cell_prop_all,
            index=test_set.get_sample_ids(),
            columns=cell_types,
        )
        pred_cell_prop_file_path = os.path.join(
            test_set_result_dir, "predicted_cell_prop.csv"
        )
        pred_cell_prop_df.to_csv(pred_cell_prop_file_path)
    if save_reconstructed_geps:
        recon_geps = pred_all_dict["recon_x_all_types"].detach().cpu().numpy()  # in TPM format
        for i, ct in enumerate(cell_types):
            current_ct_file_path = os.path.join(gep_result_dir, f'reconstructed_gep_{ct}_log2p1.csv')
            if not os.path.exists(current_ct_file_path):
                current_recon_ct = recon_geps[:, :, i]
                current_recon_df = pd.DataFrame(data=current_recon_ct, index=test_set.get_sample_ids(),
                                                columns=test_set.get_gene_list())
                current_recon_df = non_log2log_cpm(current_recon_df, transpose=False)
                current_recon_df.to_csv(current_ct_file_path, float_format='%.3f')
    return {
        "true_cell_prop": true_cell_prop,
        "pred_cell_prop_file_path": pred_cell_prop_file_path,
        "cell_types": cell_types,
        "pred_a": pred_all_dict,
        "cell_prop_result_dir": cell_prop_result_dir,
        "gep_result_dir": gep_result_dir,
        "test_set_result_dir": test_set_result_dir,
        "model_dir": output_dir,
        "test_set": test_set,
    }
