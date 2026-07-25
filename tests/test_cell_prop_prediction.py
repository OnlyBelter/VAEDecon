from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet, kl_divergence

from vaedecon.models.base import dirichlet_mean, has_usable_labels
from vaedecon.models.vae.vae_model import VAE
from vaedecon.workflow.workflow import evaluate_model


def _build_dummy_vae(
    cell_prop_weight: float = 2.0,
    kld_p_weight: float = 0.0,
    training: bool = True,
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
            gene_mean_weight=0.0,
            gene_std_weight=0.0,
        ),
        learn_gep_residual=False,
        predict_cell_prop=True,
    )
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
    dummy._cell_prop_dirichlet_loss = lambda y, dd_alpha, batch_size, device: VAE._cell_prop_dirichlet_loss(
        dummy,
        y=y,
        dd_alpha=dd_alpha,
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
        batch_size=dd_alpha.shape[0],
        device=torch.device("cpu"),
    )

    expected_kld_p = kl_divergence(Dirichlet(dd_alpha), Dirichlet(torch.ones_like(dd_alpha)))

    assert torch.allclose(kld_p, expected_kld_p)
    assert torch.allclose(cell_prop_loss, torch.zeros_like(cell_prop_loss))


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
