import os
import pandas as pd
import warnings
from pathlib import Path
from typing import Dict, Any, Type
import torch
from torch.utils.data import DataLoader

from ..utility import check_dir
from ..data import GEPDataset
from ..models import AutoModel
from ..models.nn import EncoderMLP, DecoderMLP, PositionalEncoding
from ..models.vae import VAE, VAEConfig
from ..trainers import BaseTrainerConfig, BaseTrainerL
from ..pipelines.training import TrainingPipeline

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def create_model(model_config: VAEConfig, encoder_cls: Type[EncoderMLP],
                 decoder_cls: Type[DecoderMLP]) -> VAE:
    """Creates the VAE model."""
    model = VAE(
        model_config=model_config,
        encoder=encoder_cls(
            model_config,
            position_encoding=PositionalEncoding(
                d_model=model_config.latent_dim, dropout=0, max_len=model_config.n_cell_types
            ),
        ),
        decoder=decoder_cls(model_config),
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
        # n_early_stopping_patience=n_early_stopping_patience,
    )
    training_pipeline(train_data=train_set, eval_data=val_set)
    # output_dir = training_pipeline.final_output_dir
    # return training_pipeline, output_dir


def save_metadata(dataset: GEPDataset, output_dir: str, model_config: VAEConfig) -> None:
    """Saves the gene list and cell type list."""
    dataset.save_gene_list(Path(os.path.join(output_dir, model_config.input_gene_list)))
    dataset.save_cell_types(Path(os.path.join(output_dir, model_config.cell_type_list)))


def load_trained_model(model_dir: str) -> AutoModel:
    """Loads the trained model from the specified directory."""
    trained_model = AutoModel.load_from_folder(model_dir)
    return trained_model


def evaluate_model(
        trained_model: AutoModel,
        test_set: GEPDataset,
        result_dir: str,
        model_config: VAEConfig,
        output_dir: str,
        device: str,
        pred_cell_prop_file_path: str = None
) -> Dict[str, Any]:
    """Evaluates the trained model on the test set."""
    test_set_result_dir = os.path.join(result_dir, "test_set")
    check_dir(Path(test_set_result_dir))
    cell_prop_result_dir = os.path.join(test_set_result_dir, "cell_prop")
    gep_result_dir = os.path.join(test_set_result_dir, "gep")
    check_dir(Path(cell_prop_result_dir))
    check_dir(Path(gep_result_dir))

    true_cell_prop = test_set.get_cell_prop()
    cell_types = pd.read_csv(
        os.path.join(output_dir, "cell_type_list.txt"), index_col=0, header=None
    ).index.to_list()
    if model_config.predict_cell_prop:
        pred_a = trained_model({"data": test_set.data.to(device), "labels": None})
        pred_cell_prop = pred_a["pred_cell_prop"]
        pred_cell_prop = pred_cell_prop.squeeze().detach().cpu().numpy()
    else:
        pred_cell_prop = true_cell_prop.copy()
        if pred_cell_prop_file_path is not None and os.path.exists(pred_cell_prop_file_path):
            pred_cell_prop = pd.read_csv(pred_cell_prop_file_path, index_col=0)
        else:
            raise FileExistsError('Cell property prediction file not found.')
        pred_cell_prop = pred_cell_prop.loc[:, cell_types].values
        # TODO, check the order of labels, using the ground truth as the predicted cell prop
        pred_a = trained_model({"data": test_set.data.float().to(device),
                                "labels": torch.from_numpy(pred_cell_prop).float().to(device)})

    pred_cell_prop_df = pd.DataFrame(
        pred_cell_prop,
        index=test_set.gep_data.index,
        columns=cell_types,
    )
    pred_cell_prop_file_path = os.path.join(
        test_set_result_dir, "predicted_cell_prop.csv"
    )
    pred_cell_prop_df.to_csv(pred_cell_prop_file_path)
    return {
        "true_cell_prop": true_cell_prop,
        "pred_cell_prop_file_path": pred_cell_prop_file_path,
        "cell_types": cell_types,
        "pred_a": pred_a,
        "cell_prop_result_dir": cell_prop_result_dir,
        "gep_result_dir": gep_result_dir,
        "test_set_result_dir": test_set_result_dir,
        "model_dir": output_dir,
        "test_set": test_set,
    }
