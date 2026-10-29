"""Independent correctness checks for losses and held-out evaluation."""

import numpy as np
import pytest
import torch
from torch import nn

from pinn_activation_comparison.config import ExperimentConfig
from pinn_activation_comparison.physics import breather
from pinn_activation_comparison.sampling import training_points


class ExactBreather(nn.Module):
    """Analytic reference used only to test the evaluator."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0., dtype=torch.float64))

    def forward(self, points):
        return breather(points) + 0 * self.dummy


def test_exact_field_has_zero_loss_and_heldout_error():
    from pinn_activation_comparison.losses import full_losses
    from pinn_activation_comparison.evaluate import evaluate_model

    cfg = ExperimentConfig(dtype="float64", n_f=31, n_ic=13, n_bc=11,
                           eval_nx=13, eval_nt=17, eval_batch_size=7)
    model = ExactBreather()
    losses = full_losses(model, training_points(cfg, 4), cfg)
    assert losses["total_loss"] < 1e-26
    metrics, arrays = evaluate_model(model, cfg)
    assert metrics["relative_l2"] == 0
    assert metrics["residual_rms"] < 1e-13
    assert arrays["prediction"].shape == (17, 13)
    np.testing.assert_allclose(arrays["center_exact"], arrays["center_prediction"])


def test_training_only_uses_exact_values_on_boundaries(monkeypatch):
    import pinn_activation_comparison.losses as losses

    cfg = ExperimentConfig(dtype="float64", n_f=9, n_ic=7, n_bc=5)
    visited = []

    def boundary_only(points, omega):
        assert torch.all((points[:, 0] == cfg.x_min) | (points[:, 0] == cfg.x_max))
        visited.append(points.shape[0])
        return breather(points, omega)

    monkeypatch.setattr(losses, "breather", boundary_only)
    terms = losses.loss_terms(ExactBreather(), training_points(cfg, 0), cfg)
    terms["total_loss"].backward()
    assert visited == [2 * cfg.n_bc]
    assert terms["ic_loss"] == terms["ic_value_loss"] + terms["ic_velocity_loss"]


def test_chunked_final_losses_match_full_batch():
    from pinn_activation_comparison.losses import full_losses, loss_terms
    from pinn_activation_comparison.models import PINN

    cfg = ExperimentConfig(dtype="float64", eval_batch_size=7, n_f=23, n_ic=13, n_bc=9,
                           lambda_pde=2, lambda_ic=3, lambda_bc=4)
    model = PINN(width=8, hidden_layers=2).double()
    samples = training_points(cfg, 2)
    direct = loss_terms(model, samples, cfg)
    chunks = full_losses(model, samples, cfg)
    for key in direct:
        assert chunks[key] == pytest.approx(direct[key].item(), rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("settings", [{"lr": True}, {"gammas": (False,)},
                                     {"x_min": False}, {"omega_first": "30"},
                                     {"lambda_pde": True}, {"omega": True}])
def test_config_rejects_boolean_and_nonnumeric_coefficients(settings):
    """JSON true must not silently become a physical coefficient or LR of one."""
    with pytest.raises(ValueError):
        ExperimentConfig(**settings)


def test_failed_checkpoint_can_be_reevaluated_as_strict_json(tmp_path, capsys):
    """Intentionally retained NaN checkpoints remain inspectable by the CLI."""
    import json
    from pinn_activation_comparison.evaluate import main
    from pinn_activation_comparison.models import PINN

    cfg = ExperimentConfig(width=4, hidden_layers=1, eval_nx=5, eval_nt=5)
    kwargs = {"width": 4, "hidden_layers": 1}
    model = PINN(**kwargs)
    with torch.no_grad():
        model.output_layer.bias.fill_(float("nan"))
    torch.save({"config": cfg.to_dict(), "model_kwargs": kwargs,
                "model_state_dict": model.state_dict(), "status": "nonfinite"},
               tmp_path / "checkpoint.pt")
    main(["--run-dir", str(tmp_path)])
    result = json.loads(capsys.readouterr().out)
    assert result["relative_l2"] is None
    assert result["residual_rms"] is None
    assert result["checkpoint_status"] == "nonfinite"
