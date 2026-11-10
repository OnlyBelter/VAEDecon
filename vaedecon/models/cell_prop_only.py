"""Predictor-only model used by cell-proportion pretraining."""

from torch import nn

from .base import ModelOutput, has_usable_labels, remove_cancer_cell_type
from .base.base_model import BaseAE
from .nn import DeSideCellPropPredictor
from .vae.vae_model import VAE


class CellPropOnlyModel(nn.Module):
    """Train the dedicated predictor without constructing a VAE."""

    def __init__(self, model_config, data_config):
        super().__init__()
        self.model_name = "CellPropOnly"
        self.model_config = model_config
        self.data_config = data_config
        name = str(model_config.cell_prop_predictor_cls or "").strip().lower()
        if name != "desidecellproppredictor":
            raise ValueError(f"Unsupported cell-proportion predictor: {name!r}")
        self.cell_prop_predictor = DeSideCellPropPredictor(
            args=model_config, data_config=data_config
        )

    def forward(self, inputs, **kwargs):
        del kwargs
        labels = inputs.get("labels")
        prediction = self.cell_prop_predictor(inputs["data"])
        output = ModelOutput(pred_cell_prop=prediction["cell_prop"])
        if "raw_non_cancer_cell_prop" in prediction:
            output["raw_non_cancer_cell_prop"] = prediction["raw_non_cancer_cell_prop"]
        if not has_usable_labels(labels):
            return output

        if self.model_config.deside_one_minus_alpha:
            labels = 1 - labels
        if self.model_config.cell_prop_activation_function == "sigmoid":
            supervised_pred = prediction["raw_non_cancer_cell_prop"]
            target = remove_cancer_cell_type(
                labels, self.cell_prop_predictor.cancer_cell_type_index
            )
        else:
            supervised_pred = prediction["cell_prop"]
            target = labels
        cell_prop_loss = VAE._cell_prop_supervision_loss(
            self, supervised_pred, target
        ).mean()
        output["cell_prop_loss"] = cell_prop_loss
        output["loss"] = self.model_config.loss_coefficient.cell_prop * cell_prop_loss
        return output

    def save(self, model_dir, training_config=None, data_config=None):
        BaseAE.save(self, model_dir, training_config=training_config, data_config=data_config)
