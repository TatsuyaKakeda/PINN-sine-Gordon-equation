"""Held-out initial velocity and center amplitude expose the zero-field trap."""

import math

import numpy as np
import pytest
import torch
from torch import nn

from pinn_activation_comparison.config import ExperimentConfig
from pinn_activation_comparison.evaluate import evaluate_model
from pinn_activation_comparison.models import PINN
from pinn_activation_comparison.physics import breather


class ReferenceField(nn.Module):
    def __init__(self, omega, zero=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.omega, self.zero = omega, zero

    def forward(self, points):
        field = points[:, :1] * 0.0 if self.zero else breather(points, self.omega)
        return field + self.anchor * 0.0


def evaluation_config(**overrides):
    values = dict(dtype="float64", omega=0.7, eval_nx=13, eval_nt=17,
                  eval_batch_size=5)
    values.update(overrides)
    return ExperimentConfig(**values)


def test_exact_field_has_zero_velocity_error_and_unit_center_amplitude():
    config = evaluation_config()
    model = ReferenceField(config.omega)

    metrics, arrays = evaluate_model(model, config)

    assert metrics["ic_velocity_relative_l2"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["center_amplitude_ratio"] == pytest.approx(1.0, abs=1e-14)
    x = np.linspace(config.x_min, config.x_max, config.eval_nx)
    k = math.sqrt(1.0 - config.omega**2)
    expected = 4.0 * k / np.cosh(k * x)
    np.testing.assert_array_equal(arrays["ic_x"], x)
    np.testing.assert_allclose(arrays["ic_velocity_exact"], expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(arrays["ic_velocity_prediction"], expected, rtol=1e-14, atol=1e-14)
    assert arrays["ic_velocity_prediction"].shape == (config.eval_nx,)


def test_zero_field_has_unit_velocity_error_and_zero_center_amplitude():
    config = evaluation_config()

    metrics, arrays = evaluate_model(ReferenceField(config.omega, zero=True), config)

    assert metrics["ic_velocity_relative_l2"] == pytest.approx(1.0)
    assert metrics["center_amplitude_ratio"] == 0.0
    assert metrics["relative_l2"] == pytest.approx(1.0)
    assert metrics["residual_rms"] == 0.0
    np.testing.assert_array_equal(arrays["ic_velocity_prediction"], np.zeros(config.eval_nx))


@pytest.mark.parametrize("kind", ["tanh", "siren", "gaussian"])
@pytest.mark.parametrize("training", [False, True])
def test_velocity_metrics_are_finite_and_preserve_parameter_gradients_and_mode(kind, training):
    config = evaluation_config(eval_nx=7, eval_nt=9)
    torch.manual_seed(23)
    model = PINN(kind, hidden_layers=2, width=5, gamma=0.02).double()
    model.train(training)
    gradients = []
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, index + 0.25)
        gradients.append(parameter.grad.clone())

    # Evaluation must explicitly enable coordinate differentiation when needed.
    with torch.no_grad():
        metrics, arrays = evaluate_model(model, config)

    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["ic_velocity_relative_l2"] >= 0.0
    assert metrics["center_amplitude_ratio"] >= 0.0
    assert np.isfinite(arrays["ic_velocity_prediction"]).all()
    assert model.training == training
    for parameter, expected in zip(model.parameters(), gradients, strict=True):
        torch.testing.assert_close(parameter.grad, expected, atol=0, rtol=0)
