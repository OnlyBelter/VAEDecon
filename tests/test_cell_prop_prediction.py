from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet, kl_divergence

from vaedecon.models.base import (
    build_cell_prop_from_head_output,
    dirichlet_mean,
    has_usable_labels,
    remove_cancer_cell_type,
)
from vaedecon.models.vae.vae_model import VAE
from vaedecon.trainers.base_trainer import (
    _apply_aux_loss_schedules,
    _resolve_linear_schedule_value,
)
from vaedecon.workflow.workflow import evaluate_model


def _build_dummy_vae(
    cell_prop_weight: float = 2.0,
    kld_p_weight: float = 0.0,
    cell_type_existence_weight: float = 0.0,
    training: bool = True,
    activation_function: str = "softplus",
    cancer_cell_type_index: int | None = None,
    existence_shift_scale: float = 0.0,
    cell_prop_loss_type: str = "mse",
    cell_prop_loss_kl_weight: float = 0.5,
):
    class DummyVAE:
        pass

    dummy = DummyVAE()
    dummy.model_config = SimpleNamespace(
        loss_coefficient=SimpleNamespace(
            beta=0.0,
            gamma=0.0,
            attractor_weight=0.0,
            z_score_reg_weight=0.0,
            z_score_kl_weight=0.0,
            low_mean_std_weight=0.0,
            hierarchical_code_weight=0.0,
            low_mean_threshold=0.0,
            low_std_threshold=0.0,
            kld_p=kld_p_weight,
            cell_prop=cell_prop_weight,
            cell_type_existence_weight=cell_type_existence_weight,
            gene_mean_weight=0.0,
            gene_std_weight=0.0,
        ),
        learn_gep_residual=False,
        predict_cell_prop=True,
        cell_prop_activation_function=activation_function,
        cell_type_existence_shift_scale=existence_shift_scale,
        cell_prop_loss_type=cell_prop_loss_type,
        cell_prop_loss_kl_weight=cell_prop_loss_kl_weight,
    )
    dummy.data_config = SimpleNamespace(training_sct_gep_cell_prop_threshold=0.1)
    dummy.cell_prop_activation_function = activation_function
    dummy.cancer_cell_type_index = cancer_cell_type_index
    dummy.training = training
    dummy.scaling_factor = 1.0
    dummy.g_mean_non_log = torch.ones((3, 2), dtype=torch.float32)
    dummy.g_std_non_log = torch.ones((3, 2), dtype=torch.float32)
    dummy.z_scores = torch.ones((2, 3, 2), dtype=torch.float32)

    dummy._reconstruction_loss = lambda x, recon_x_conv: torch.zeros(x.shape[0], device=x.device)
    dummy._gene_statistics_loss = lambda recon_gene_mean, recon_gene_std, device: (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
    )
    dummy._z_score_kl_loss = lambda recon_x_all_types_cpm: torch.zeros(
        recon_x_all_types_cpm.shape[0], device=recon_x_all_types_cpm.device
    )
    dummy._cell_prop_dirichlet_loss = lambda y, dd_alpha, pred_cell_prop, batch_size, device: VAE._cell_prop_dirichlet_loss(
        dummy,
        y=y,
        dd_alpha=dd_alpha,
        pred_cell_prop=pred_cell_prop,
        batch_size=batch_size,
        device=device,
    )
    dummy._latent_kld_loss = lambda **kwargs: torch.zeros(
        kwargs["mu_types"].shape[0], device=kwargs["device"]
    )
    dummy._repulsion_loss = lambda **kwargs: torch.zeros(
        kwargs["mu_types"].shape[0], device=kwargs["mu_types"].device
    )
    dummy._attractor_loss = lambda **kwargs: torch.zeros(
        kwargs["mu_types"].shape[0], device=kwargs["mu_types"].device
    )
    dummy._hierarchical_code_loss = lambda **kwargs: torch.zeros(
        kwargs["mu_types"].shape[0], device=kwargs["mu_types"].device
    )
    dummy._cell_type_existence_loss = lambda **kwargs: VAE._cell_type_existence_loss(
        dummy,
        **kwargs,
    )
    return dummy


def test_has_usable_labels_handles_empty_tensor():
    assert not has_usable_labels(None)
    assert not has_usable_labels(torch.empty(0))
    assert has_usable_labels(torch.tensor([[0.7, 0.3]], dtype=torch.float32))


def test_dirichlet_mean_returns_deterministic_normalized_alpha():
    dd_alpha = torch.tensor([[1.0, 3.0], [3.0, 1.0]], dtype=torch.float32)

    expected = torch.tensor([[0.25, 0.75], [0.75, 0.25]], dtype=torch.float32)

    assert torch.allclose(dirichlet_mean(dd_alpha), expected)


def test_loss_function_includes_weighted_kld_p_and_supervised_cell_prop_term():
    dummy = _build_dummy_vae(cell_prop_weight=2.0, kld_p_weight=0.5, training=True)
    x = torch.zeros((2, 3), dtype=torch.float32)
    y = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float32)
    dd_alpha = torch.tensor([[1.0, 3.0], [3.0, 1.0]], dtype=torch.float32)

    loss_terms = VAE.loss_function(
        dummy,
        x=x,
        y=y,
        recon_x_conv=torch.zeros_like(x),
        mu_types=torch.zeros((2, 1, 2), dtype=torch.float32),
        logvar_types=torch.zeros((2, 1, 2), dtype=torch.float32),
        pred_cell_prop=dirichlet_mean(dd_alpha),
        existence_logits=None,
        dd_alpha=dd_alpha,
        mu_prior=torch.zeros((2, 1), dtype=torch.float32),
        recon_gene_mean=torch.ones((3, 2), dtype=torch.float32),
        recon_gene_std=torch.ones((3, 2), dtype=torch.float32),
        logvar_mean=torch.zeros((2, 1), dtype=torch.float32),
        mu_mean=torch.zeros((2, 1), dtype=torch.float32),
        device=torch.device("cpu"),
        recon_x_all_types_cpm=torch.ones((2, 3, 2), dtype=torch.float32),
    )

    normalized_dd_alpha = dd_alpha / dd_alpha.sum(dim=-1, keepdim=True)
    expected_cell_prop_loss = F.mse_loss(
        normalized_dd_alpha,
        y,
        reduction="none",
    ).sum(dim=-1)
    expected_kld_p = kl_divergence(Dirichlet(dd_alpha), Dirichlet(torch.ones_like(dd_alpha)))
    expected_total = (0.5 * expected_kld_p + 2.0 * expected_cell_prop_loss).mean()

    assert torch.isclose(loss_terms.cell_prop, expected_cell_prop_loss.mean())
    assert torch.isclose(loss_terms.kld_p, expected_kld_p.mean())
    assert torch.isclose(loss_terms.total, expected_total)


def test_loss_function_requires_labels_for_supervised_training():
    dummy = _build_dummy_vae(cell_prop_weight=1.0, training=True)
    x = torch.zeros((2, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="requires cell-fraction labels"):
        VAE.loss_function(
            dummy,
            x=x,
            y=torch.empty(0, dtype=torch.float32),
            recon_x_conv=torch.zeros_like(x),
            mu_types=torch.zeros((2, 1, 2), dtype=torch.float32),
            logvar_types=torch.zeros((2, 1, 2), dtype=torch.float32),
            pred_cell_prop=torch.ones((2, 2), dtype=torch.float32) / 2,
            existence_logits=None,
            dd_alpha=torch.ones((2, 2), dtype=torch.float32),
            mu_prior=torch.zeros((2, 1), dtype=torch.float32),
            recon_gene_mean=torch.ones((3, 2), dtype=torch.float32),
            recon_gene_std=torch.ones((3, 2), dtype=torch.float32),
            logvar_mean=torch.zeros((2, 1), dtype=torch.float32),
            mu_mean=torch.zeros((2, 1), dtype=torch.float32),
            device=torch.device("cpu"),
            recon_x_all_types_cpm=torch.ones((2, 3, 2), dtype=torch.float32),
        )


def test_cell_prop_dirichlet_loss_keeps_kl_without_labels():
    dummy = _build_dummy_vae(cell_prop_weight=1.0, training=False)
    dd_alpha = torch.tensor([[1.0, 3.0], [3.0, 1.0]], dtype=torch.float32)

    kld_p, cell_prop_loss = VAE._cell_prop_dirichlet_loss(
        dummy,
        y=torch.empty(0, dtype=torch.float32),
        dd_alpha=dd_alpha,
        pred_cell_prop=dirichlet_mean(dd_alpha),
        batch_size=dd_alpha.shape[0],
        device=torch.device("cpu"),
    )

    expected_kld_p = kl_divergence(Dirichlet(dd_alpha), Dirichlet(torch.ones_like(dd_alpha)))

    assert torch.allclose(kld_p, expected_kld_p)
    assert torch.allclose(cell_prop_loss, torch.zeros_like(cell_prop_loss))


def test_sigmoid_cell_prop_builder_inserts_cancer_and_normalizes_rows():
    logits = torch.tensor([[2.0, 2.0]], dtype=torch.float32)

    cell_prop, dd_alpha = build_cell_prop_from_head_output(
        head_output=logits,
        activation_function="sigmoid",
        n_cell_types=3,
        cancer_cell_type_index=1,
    )

    expected_non_cancer = torch.sigmoid(logits)
    expected_non_cancer = expected_non_cancer / expected_non_cancer.sum(dim=-1, keepdim=True)
    expected = torch.tensor(
        [[expected_non_cancer[0, 0].item(), 0.0, expected_non_cancer[0, 1].item()]],
        dtype=torch.float32,
    )

    assert dd_alpha is None
    assert torch.allclose(cell_prop.sum(dim=-1), torch.ones(1, dtype=torch.float32))
    assert torch.allclose(cell_prop, expected, atol=1e-6)


def test_sigmoid_cell_prop_loss_supervises_only_non_cancer_columns():
    dummy = _build_dummy_vae(
        cell_prop_weight=1.0,
        training=True,
        activation_function="sigmoid",
        cancer_cell_type_index=1,
    )
    pred_cell_prop = torch.tensor(
        [[0.25, 0.35, 0.40], [0.10, 0.60, 0.30]],
        dtype=torch.float32,
    )
    y = torch.tensor(
        [[0.20, 0.50, 0.30], [0.30, 0.40, 0.30]],
        dtype=torch.float32,
    )

    kld_p, cell_prop_loss = VAE._cell_prop_dirichlet_loss(
        dummy,
        y=y,
        dd_alpha=None,
        pred_cell_prop=pred_cell_prop,
        batch_size=pred_cell_prop.shape[0],
        device=torch.device("cpu"),
    )

    expected_loss = F.mse_loss(
        remove_cancer_cell_type(pred_cell_prop, 1),
        remove_cancer_cell_type(y, 1),
        reduction="none",
    ).sum(dim=-1)

    assert torch.allclose(kld_p, torch.zeros_like(kld_p))
    assert torch.allclose(cell_prop_loss, expected_loss)


def test_softmax_cell_prop_builder_predicts_all_cell_types_and_normalizes_rows():
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

    cell_prop, dd_alpha = build_cell_prop_from_head_output(
        head_output=logits,
        activation_function="softmax",
        n_cell_types=3,
    )

    expected = torch.softmax(logits, dim=-1)

    assert dd_alpha is None
    assert torch.allclose(cell_prop, expected)
    assert torch.allclose(cell_prop.sum(dim=-1), torch.ones(1, dtype=torch.float32))


def test_softmax_cell_prop_loss_supervises_all_cell_type_columns():
    dummy = _build_dummy_vae(
        cell_prop_weight=1.0,
        training=True,
        activation_function="softmax",
    )
    pred_cell_prop = torch.tensor(
        [[0.25, 0.35, 0.40], [0.10, 0.60, 0.30]],
        dtype=torch.float32,
    )
    y = torch.tensor(
        [[0.20, 0.50, 0.30], [0.30, 0.40, 0.30]],
        dtype=torch.float32,
    )

    kld_p, cell_prop_loss = VAE._cell_prop_dirichlet_loss(
        dummy,
        y=y,
        dd_alpha=None,
        pred_cell_prop=pred_cell_prop,
        batch_size=pred_cell_prop.shape[0],
        device=torch.device("cpu"),
    )

    expected_loss = F.mse_loss(
        pred_cell_prop,
        y,
        reduction="none",
    ).sum(dim=-1)

    assert torch.allclose(kld_p, torch.zeros_like(kld_p))
    assert torch.allclose(cell_prop_loss, expected_loss)


def test_softmax_cell_prop_loss_supports_l1_kl():
    dummy = _build_dummy_vae(
        cell_prop_weight=1.0,
        training=True,
        activation_function="softmax",
        cell_prop_loss_type="l1_kl",
        cell_prop_loss_kl_weight=0.5,
    )
    pred_cell_prop = torch.tensor(
        [[0.25, 0.35, 0.40], [0.10, 0.60, 0.30]],
        dtype=torch.float32,
    )
    y = torch.tensor(
        [[0.20, 0.50, 0.30], [0.30, 0.40, 0.30]],
        dtype=torch.float32,
    )

    _, cell_prop_loss = VAE._cell_prop_dirichlet_loss(
        dummy,
        y=y,
        dd_alpha=None,
        pred_cell_prop=pred_cell_prop,
        batch_size=pred_cell_prop.shape[0],
        device=torch.device("cpu"),
    )

    pred_safe = pred_cell_prop.clamp_min(1e-8)
    target_safe = y.clamp_min(1e-8)
    pred_safe = pred_safe / pred_safe.sum(dim=-1, keepdim=True)
    target_safe = target_safe / target_safe.sum(dim=-1, keepdim=True)
    expected_loss = torch.abs(pred_safe - target_safe).sum(dim=-1)
    expected_loss = expected_loss + 0.5 * (
        target_safe * (torch.log(target_safe) - torch.log(pred_safe))
    ).sum(dim=-1)

    assert torch.allclose(cell_prop_loss, expected_loss)


def test_shared_feature_mean_fusion_averages_projected_features():
    dummy = SimpleNamespace(
        cell_prop_feature_projectors=torch.nn.ModuleList(
            [torch.nn.Identity(), torch.nn.Identity()]
        ),
        cell_prop_fusion_strategy="shared_feature_mean",
        cell_prop_fusion_gate=None,
    )
    features = [
        torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 7.0], [6.0, 8.0]], dtype=torch.float32),
    ]

    fused_feature, gate_weights = VAE._fuse_cell_prop_features(dummy, features)

    expected = torch.stack(features, dim=1).mean(dim=1)
    assert gate_weights is None
    assert torch.allclose(fused_feature, expected)


def test_shared_feature_gated_fusion_uses_gate_weights():
    class _ConstantGate(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.0, 1.0]], dtype=x.dtype, device=x.device).expand(x.shape[0], -1)

    dummy = SimpleNamespace(
        cell_prop_feature_projectors=torch.nn.ModuleList(
            [torch.nn.Identity(), torch.nn.Identity()]
        ),
        cell_prop_fusion_strategy="shared_feature_gated",
        cell_prop_fusion_gate=_ConstantGate(),
    )
    features = [
        torch.tensor([[1.0, 3.0]], dtype=torch.float32),
        torch.tensor([[5.0, 7.0]], dtype=torch.float32),
    ]

    fused_feature, gate_weights = VAE._fuse_cell_prop_features(dummy, features)

    expected_gate_weights = torch.softmax(torch.tensor([[0.0, 1.0]], dtype=torch.float32), dim=-1)
    expected = (
        torch.stack(features, dim=1) * expected_gate_weights.unsqueeze(-1)
    ).sum(dim=1)
    assert torch.allclose(gate_weights, expected_gate_weights)
    assert torch.allclose(fused_feature, expected)


def test_apply_cell_type_existence_shift_uses_centered_soft_threshold():
    dummy = _build_dummy_vae(
        cell_prop_weight=0.0,
        training=False,
        activation_function="softmax",
        existence_shift_scale=0.2,
    )
    mu_types = torch.zeros((1, 2, 3), dtype=torch.float32)
    pred_cell_prop = torch.tensor([[0.01, 0.10, 0.60]], dtype=torch.float32)

    existence_logits, existence_probs, mu_types_shifted = VAE._apply_cell_type_existence_shift(
        dummy,
        mu_types=mu_types,
        pred_cell_prop=pred_cell_prop,
        device=torch.device("cpu"),
    )

    assert existence_logits.shape == pred_cell_prop.shape
    assert existence_probs.shape == pred_cell_prop.shape
    assert torch.all(existence_probs > 0)
    assert torch.all(existence_probs < 1)
    assert torch.allclose(mu_types_shifted[0, :, 0], mu_types_shifted[0, :, 0][0].expand(2))
    assert mu_types_shifted[0, 0, 0].item() < 0.0
    assert mu_types_shifted[0, 0, 2].item() > 0.0


def test_loss_function_skips_cell_type_existence_supervision_during_inference():
    dummy = _build_dummy_vae(
        cell_prop_weight=0.0,
        cell_type_existence_weight=1.0,
        training=False,
        activation_function="softmax",
    )
    x = torch.zeros((2, 3), dtype=torch.float32)
    existence_logits = torch.tensor([[1.0, -1.0], [0.5, -0.5]], dtype=torch.float32)

    loss_terms = VAE.loss_function(
        dummy,
        x=x,
        y=None,
        recon_x_conv=torch.zeros_like(x),
        mu_types=torch.zeros((2, 1, 2), dtype=torch.float32),
        logvar_types=torch.zeros((2, 1, 2), dtype=torch.float32),
        pred_cell_prop=torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float32),
        existence_logits=existence_logits,
        dd_alpha=None,
        mu_prior=torch.zeros((2, 1), dtype=torch.float32),
        recon_gene_mean=torch.ones((3, 2), dtype=torch.float32),
        recon_gene_std=torch.ones((3, 2), dtype=torch.float32),
        logvar_mean=torch.zeros((2, 1), dtype=torch.float32),
        mu_mean=torch.zeros((2, 1), dtype=torch.float32),
        device=torch.device("cpu"),
        recon_x_all_types_cpm=torch.ones((2, 3, 2), dtype=torch.float32),
    )

    assert torch.allclose(
        loss_terms.cell_type_existence,
        torch.tensor(0.0, dtype=torch.float32),
    )


def test_resolve_linear_schedule_value_interpolates_between_epochs():
    schedule = SimpleNamespace(
        start_epoch=10,
        end_epoch=20,
        start_value=0.0,
        end_value=1.0,
    )

    assert _resolve_linear_schedule_value(schedule, epoch=5) == 0.0
    assert _resolve_linear_schedule_value(schedule, epoch=10) == 0.0
    assert _resolve_linear_schedule_value(schedule, epoch=15) == 0.5
    assert _resolve_linear_schedule_value(schedule, epoch=25) == 1.0


def test_apply_aux_loss_schedules_updates_live_model_config():
    model = SimpleNamespace(
        model_config=SimpleNamespace(
            loss_coefficient=SimpleNamespace(
                cell_type_sct_gep_weight=0.0,
                hierarchical_code_weight=0.0,
                cell_type_existence_weight=0.0,
            ),
            cell_type_existence_shift_scale=0.0,
        )
    )
    training_config = SimpleNamespace(
        aux_loss_schedules={
            "cell_type_sct_gep_weight": SimpleNamespace(
                start_epoch=0, end_epoch=10, start_value=0.0, end_value=1.0
            ),
            "cell_type_existence_shift_scale": SimpleNamespace(
                start_epoch=0, end_epoch=10, start_value=0.0, end_value=0.5
            ),
        }
    )

    _apply_aux_loss_schedules(model, training_config, epoch=5)

    assert model.model_config.loss_coefficient.cell_type_sct_gep_weight == 0.5
    assert model.model_config.cell_type_existence_shift_scale == 0.25


class _DummyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self):
        self._sample_ids = ["sample_1"]
        self._gene_list = ["gene_1", "gene_2"]
        self._cell_prop = pd.DataFrame(
            [[0.1, 0.9]],
            index=self._sample_ids,
            columns=["Cancer Cells", "CD8 T"],
        )

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "data": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "labels": torch.tensor([0.1, 0.9], dtype=torch.float32),
        }

    def get_cell_prop(self):
        return self._cell_prop

    def get_sample_ids(self):
        return self._sample_ids

    def get_gene_list(self):
        return self._gene_list


class _DummyPredictiveModel:
    def __call__(self, batch):
        return {
            "pred_cell_prop": torch.tensor([[0.8, 0.2]], dtype=torch.float32),
            "mu": torch.tensor([[0.5]], dtype=torch.float32),
            "recon_x_all_types": torch.ones((1, 2, 2), dtype=torch.float32),
        }


def test_evaluate_model_saves_predicted_cell_prop_for_single_sample_batch(tmp_path):
    cell_type_fp = tmp_path / "cell_types.txt"
    cell_type_fp.write_text("Cancer Cells\nCD8 T\n")
    result_dir = tmp_path / "results"
    output_dir = tmp_path / "model"
    result_dir.mkdir()
    output_dir.mkdir()

    results = evaluate_model(
        trained_model=_DummyPredictiveModel(),
        test_set=_DummyPredictionDataset(),
        result_dir=str(result_dir),
        model_config=SimpleNamespace(
            predict_cell_prop=True,
            cell_type_fp=str(cell_type_fp),
        ),
        output_dir=str(output_dir),
        device="cpu",
        val_batch_size=1,
        save_reconstructed_geps=False,
        dataset_type="test",
        result_set_name="toy_test_set",
    )

    pred_fp = results["pred_cell_prop_file_path"]
    pred_df = pd.read_csv(pred_fp, index_col=0)

    assert pred_df.shape == (1, 2)
    assert list(pred_df.columns) == ["Cancer Cells", "CD8 T"]
    assert pred_df.index.tolist() == ["sample_1"]
    assert pred_df.iloc[0].tolist() == pytest.approx([0.8, 0.2])
