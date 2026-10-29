"""Physical diagnostics must distinguish periodic breathers from small outputs."""

import numpy as np
import pytest
import torch
from torch import nn

from pinn_activation_comparison.config import ExperimentConfig


class AnalyticField(nn.Module):
    """Provide exact or zero fields with a device/dtype anchor."""

    def __init__(self, zero=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.zero = zero

    def forward(self, points):
        from pinn_activation_comparison.physics import breather
        return points[:, :1] * 0 + self.anchor if self.zero else breather(points) + self.anchor * 0


def test_exact_breather_period_amplitude_and_energy():
    from pinn_activation_comparison.tanh_diagnostics import physical_diagnostics
    cfg = ExperimentConfig(eval_nx=101, eval_nt=101, eval_batch_size=64, dtype="float64")
    metrics, arrays = physical_diagnostics(AnalyticField(), cfg)
    assert metrics['center_relative_l2'] == pytest.approx(0, abs=1e-14)
    assert metrics['period_prediction'] == pytest.approx(2 * np.pi / .8, rel=1e-4)
    assert metrics['positive_peak_count'] == 3
    assert metrics['center_peak_amplitude_ratio'] == pytest.approx(1)
    assert metrics['center_late_rms_ratio'] == pytest.approx(1)
    assert metrics['energy_initial'] == pytest.approx(9.6, rel=1e-3)
    assert metrics['energy_final_ratio'] == pytest.approx(1, rel=1e-3)
    assert metrics['near_zero_center'] is False
    assert metrics['meets_one_percent'] is True
    assert np.isfinite(arrays['energy_prediction']).all()


def test_zero_field_does_not_receive_a_fictitious_period():
    from pinn_activation_comparison.tanh_diagnostics import physical_diagnostics
    cfg = ExperimentConfig(eval_nx=21, eval_nt=21, eval_batch_size=64, dtype="float64")
    metrics, arrays = physical_diagnostics(AnalyticField(zero=True), cfg)
    assert metrics['center_relative_l2'] == pytest.approx(1)
    assert metrics['period_prediction'] is None
    assert metrics['positive_peak_count'] == 0
    assert metrics['center_peak_amplitude_ratio'] == 0
    assert metrics['near_zero_center'] is True
    assert metrics['meets_one_percent'] is False
    assert metrics['energy_initial'] == pytest.approx(0)
    assert metrics['energy_final_ratio'] is None
    assert np.all(arrays['energy_prediction'] == 0)


def test_diagnostics_preserve_training_mode_and_existing_gradients():
    from pinn_activation_comparison.tanh_diagnostics import physical_diagnostics
    from pinn_activation_comparison.models import PINN
    model = PINN(hidden_layers=1, width=4).double().train()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    cfg = ExperimentConfig(eval_nx=9, eval_nt=9, eval_batch_size=32, dtype="float64")
    metrics, arrays = physical_diagnostics(model, cfg)
    assert model.training
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    assert np.isfinite(metrics['center_relative_l2'])
