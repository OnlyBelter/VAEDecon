import os
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Dict, Any, Type, Union, TypeVar, Sequence
import torch
from torch.utils.data import DataLoader

from ..utility import check_dir, non_log2log_cpm
from ..data import GEPDataset
from ..models import AutoModel, BaseAE, AutoConfig
from ..models.base import BaseEncoder
from ..models.gnn import EncoderGNN, EncoderSGNN
from ..models.nn import EncoderMLP, DecoderMLP, PositionalEncoding, EncoderHybrid
from ..models.vae import VAE, VAEConfig
from ..trainers import BaseTrainerConfig, BaseTrainerL, PLTrainer
from ..pipelines import TrainingPipeline

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Define type variables for better type hinting
T_Encoder = TypeVar('T_Encoder', bound=BaseEncoder)
T_Decoder = TypeVar('T_Decoder', bound=DecoderMLP)


def create_model(model_config: VAEConfig, encoder_cls_name_list: list[str], decoder_cls: Type[T_Decoder]) -> VAE:
    """Creates the VAE model."""
    position_encoding = PositionalEncoding(
        d_model=model_config.latent_dim,
        dropout=0,
        max_len=model_config.n_cell_types
    )
    encoders = []
    kwargs: Dict[str, Any] = {
        "args": model_config,
        "position_encoding": position_encoding,
    }
    for encoder_cls_name in encoder_cls_name_list:
        encoder_cls_name = encoder_cls_name.lower()
        if encoder_cls_name == "EncoderSGNN".lower():
            encoder_cls = EncoderSGNN
        elif encoder_cls_name == "EncoderMLP".lower():
            encoder_cls = EncoderMLP
        elif encoder_cls_name == "EncoderHybrid".lower():
            encoder_cls = EncoderHybrid
            kwargs_for_hybrid = kwargs.copy()
            kwargs_for_hybrid['mlp_encoder'] = EncoderMLP(**kwargs)
            kwargs_for_hybrid['gnn_encoder'] = EncoderSGNN(**kwargs)
            kwargs = kwargs_for_hybrid.copy()
        else:
            raise NotImplementedError(encoder_cls_name)

        encoders.append(encoder_cls(**kwargs))

    decoder = decoder_cls(args=model_config)
    model = VAE(
        model_config=model_config,
        encoders=encoders,
        decoder=decoder,
    )

    return model


def train_model(
        model: VAE,
        train_set: torch.utils.data.Dataset,
        val_set: torch.utils.data.Dataset,
        trainer_cls: Type[BaseTrainerL],
        config: BaseTrainerConfig,
        device: str,
        result_dir: str,
) -> None:
    """Trains the model using the training pipeline."""
    training_pipeline = TrainingPipeline(
        training_config=config,
        model=model.to(device),
        trainer_cls=trainer_cls,
        result_dir=result_dir,
        debug_model=config.debug_model,
        # n_early_stopping_patience=n_early_stopping_patience,
    )
    training_pipeline(train_data=train_set, eval_data=val_set)
    # output_dir = training_pipeline.final_output_dir
    # return training_pipeline, output_dir


def save_metadata(dataset: GEPDataset, model_config: VAEConfig) -> None:
    """Saves the gene list and cell type list."""
    dataset.save_gene_list(Path(model_config.input_gene_list_fp))
    dataset.save_cell_types(Path(model_config.cell_type_fp))


def load_trained_model(model_dir: str) -> Union[AutoModel, BaseAE]:
    """Loads the trained model from the specified directory."""
    try:
        # find the model file in the directory by .ckpt ending
        model_file = [f for f in os.listdir(model_dir) if f.endswith('.ckpt')]
        model_file_path = os.path.join(model_dir, model_file[0])
        model_config_path = os.path.join(model_dir, "model_config.json")
        training_config_path = os.path.join(model_dir, "training_config.json")
        model_config = AutoConfig.from_json_file(model_config_path)
        training_config = BaseTrainerConfig.from_json_file(training_config_path)

        model = create_model(model_config=model_config, encoder_cls_name_list=model_config.encoders,
                             decoder_cls=DecoderMLP)
        trained_model = PLTrainer.load_from_checkpoint(
            checkpoint_path=model_file_path,
            model=model,
            training_config=training_config)
        print('Model loaded from checkpoint:', model_file_path)
        trained_model.eval()
        trained_model.freeze()
        return trained_model.model
    except FileNotFoundError:
        # if no .ckpt file found, load the model from the folder
        trained_model = AutoModel.load_from_folder(model_dir)
        return trained_model


def evaluate_model(
        trained_model: AutoModel,
        test_set: Union[GEPDataset | DataLoader],
        result_dir: str,
        model_config: VAEConfig,
        output_dir: str,
        device: str,
        pred_cell_prop_file_path: str = None,
        val_batch_size: int = None,
        save_reconstructed_geps: bool = False,
        dataset_type: str = 'training',  # or test, tcga
) -> Dict[str, Any]:
    """Evaluates the trained model on the test set."""
    test_set_result_dir = os.path.join(result_dir, "test_set")
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
    pred_results = []
    pred_cell_prop_list = []
    with torch.no_grad():
        for batch in test_set_loader:
            pred_a = trained_model(batch)  # A ModelOutput including 11 elements
            if model_config.predict_cell_prop:
                # pred_a = trained_model(batch)
                # pred_a = trained_model(test_set_loader)
                pred_cell_prop = pred_a["pred_cell_prop"]
                pred_cell_prop = pred_cell_prop.squeeze().detach().cpu().numpy()
            else:
                if 'labels' in batch.keys() and batch['labels']:
                    pred_cell_prop = batch["labels"].squeeze().detach().cpu().numpy()
                else:
                    pred_cell_prop = []
                    # raise FileExistsError('Cell property prediction file not found.')

                # TODO, check the order of labels, using the ground truth as the predicted cell prop
                # pred_a = trained_model({"data": test_set.data.float().to(device),
                #                         "labels": torch.from_numpy(pred_cell_prop).float().to(device)})

                # pred_a = trained_model(test_set_loader)
            pred_cell_prop_list.append(pred_cell_prop)
            pred_results.append(pred_a)
    if pred_cell_prop_all is not None and pred_cell_prop_list[0]:
        pred_cell_prop_all = np.concatenate(pred_cell_prop_list, axis=0)
    pred_all_dict = {}
    for a_result in pred_results:
        for key, value in a_result.items():
            if key not in pred_all_dict:
                pred_all_dict[key] = []
            pred_all_dict[key].append(value)
    for key, value in pred_all_dict.items():
        if value[0] is not None and len(value[0].shape) > 0:
            pred_all_dict[key] = torch.cat(value, dim=0)
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
