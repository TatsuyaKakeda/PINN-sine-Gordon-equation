"""Numerical checks of the exact solution, PINN derivatives, and initialization."""

import math

import pytest
import torch
from torch import nn

from pinn_activation_comparison.activations import (
    GaussianSine,
    Sine,
    activation_derivatives,
)
from pinn_activation_comparison.models import PINN
from pinn_activation_comparison.physics import (
    breather,
    field_derivatives,
    initial_velocity,
    pde_residual,
)


def sample_points():
    generator = torch.Generator().manual_seed(81)
    return torch.rand(61, 2, generator=generator, dtype=torch.float64) * torch.tensor(
        [16.0, 20.0], dtype=torch.float64
    ) - torch.tensor([8.0, 0.0], dtype=torch.float64)


def test_exact_breather_satisfies_sine_gordon_in_float64():
    residual = pde_residual(breather, sample_points())
    assert residual.shape == (61, 1)
    assert residual.abs().max().item() < 2e-14


def test_exact_initial_displacement_and_velocity():
    x = torch.linspace(-10, 10, 101, dtype=torch.float64).unsqueeze(1)
    points = torch.cat((x, torch.zeros_like(x)), dim=1)
    derivatives = field_derivatives(breather, points)
    torch.testing.assert_close(derivatives["phi"], torch.zeros_like(x), atol=0, rtol=0)
    torch.testing.assert_close(derivatives["phi_x"], torch.zeros_like(x), atol=0, rtol=0)
    expected_velocity = 2.4 / torch.cosh(0.6 * x)
    torch.testing.assert_close(initial_velocity(x), expected_velocity, atol=2e-15, rtol=2e-15)
    torch.testing.assert_close(derivatives["phi_t"], expected_velocity, atol=2e-15, rtol=2e-15)


@pytest.mark.parametrize("kind", ["tanh", "siren", "gaussian"])
def test_pde_loss_backpropagates_through_second_derivatives(kind):
    torch.manual_seed(17)
    model = PINN(kind, hidden_layers=2, width=9, gamma=0.02).double()
    loss = pde_residual(model, sample_points()).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert any(parameter.grad.abs().max() > 0 for parameter in model.parameters())


def test_gamma_zero_has_identical_weights_outputs_and_two_derivatives():
    kwargs = dict(hidden_layers=3, width=12, omega_first=12.0, omega_hidden=7.0)
    torch.manual_seed(29)
    siren = PINN("siren", **kwargs).double()
    torch.manual_seed(29)
    gaussian = PINN("gaussian", gamma=0.0, **kwargs).double()
    for expected, actual in zip(siren.parameters(), gaussian.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    siren_fields = field_derivatives(siren, sample_points())
    gaussian_fields = field_derivatives(gaussian, sample_points())
    for name, expected in siren_fields.items():
        torch.testing.assert_close(gaussian_fields[name], expected, atol=0, rtol=0)


def test_normalization_keeps_derivatives_in_physical_coordinates():
    model = PINN(
        "siren", hidden_layers=1, width=1, omega_first=1.0,
        x_min=-2.0, x_max=6.0, t_max=10.0,
    ).double()
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    with torch.no_grad():
        linear_layers[0].weight.copy_(torch.tensor([[2.0, -3.0]], dtype=torch.float64))
        linear_layers[0].bias.zero_()
        linear_layers[1].weight.fill_(1.0)
        linear_layers[1].bias.zero_()
    points = torch.tensor([[-1.0, 2.0], [4.0, 8.0]], dtype=torch.float64)
    phase = 0.5 * (points[:, :1] - 2.0) - 0.6 * (points[:, 1:] - 5.0)
    result = field_derivatives(model, points)
    expected = {
        "phi": phase.sin(),
        "phi_x": 0.5 * phase.cos(),
        "phi_t": -0.6 * phase.cos(),
        "phi_xx": -0.25 * phase.sin(),
        "phi_tt": -0.36 * phase.sin(),
    }
    for name, field in expected.items():
        torch.testing.assert_close(result[name], field, atol=1e-14, rtol=1e-14)
    assert not points.requires_grad


@pytest.mark.parametrize("kind", ["siren", "gaussian"])
def test_siren_weight_initialization_uses_first_and_hidden_bounds(kind):
    torch.manual_seed(61)
    scale = 1.7
    omega_hidden = 11.0
    model = PINN(kind, hidden_layers=4, width=64, omega_first=30.0,
                 omega_hidden=omega_hidden, init_scale=scale)
    layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    assert len(layers) == 5
    for index, layer in enumerate(layers):
        bound = scale * (
            1.0 / layer.in_features if index == 0
            else math.sqrt(6.0 / layer.in_features) / omega_hidden
        )
        assert layer.weight.abs().max().item() <= bound + 1e-7
        assert layer.weight.abs().max().item() >= bound * 0.8
        assert layer.bias.abs().max().item() <= 1.0 / math.sqrt(layer.in_features) + 1e-7


def test_default_architecture_and_distinct_frequencies():
    model = PINN("gaussian", omega_first=17.0, omega_hidden=5.0, gamma=0.03)
    activations = [module for module in model.modules() if isinstance(module, GaussianSine)]
    layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in layers] == [
        (2, 64), (64, 64), (64, 64), (64, 64), (64, 1)
    ]
    assert [activation.omega for activation in activations] == [17.0, 5.0, 5.0, 5.0]
    assert all(activation.gamma == 0.03 for activation in activations)
    assert not any("gamma" in name for name, _ in model.named_parameters())


@pytest.mark.parametrize("kind", ["tanh", "siren", "gaussian"])
def test_activation_derivative_formula_matches_autograd(kind):
    z = torch.linspace(-2.0, 2.0, 21, dtype=torch.float64).requires_grad_(True)
    omega, gamma = 3.5, 0.07
    if kind == "tanh":
        value = torch.tanh(z)
    elif kind == "siren":
        value = Sine(omega)(z)
    else:
        value = GaussianSine(omega, gamma)(z)
    first = torch.autograd.grad(value.sum(), z, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), z)[0]
    for actual, expected in zip(
        activation_derivatives(z, kind, omega=omega, gamma=gamma),
        (value, first, second), strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("field", [
    lambda points: 2.0 * points[:, :1] + 3.0 * points[:, 1:],
    lambda points: torch.ones_like(points[:, :1]),
])
def test_linear_and_constant_fields_have_zero_second_derivatives(field):
    result = field_derivatives(field, sample_points())
    assert torch.count_nonzero(result["phi_xx"]) == 0
    assert torch.count_nonzero(result["phi_tt"]) == 0
    assert torch.isfinite(pde_residual(field, sample_points())).all()


@pytest.mark.parametrize("options", [
    {"model_type": "unknown"}, {"hidden_layers": 0}, {"hidden_layers": 2.5},
    {"width": 0}, {"width": True}, {"gamma": -0.1}, {"gamma": float("nan")},
    {"omega_first": 0}, {"omega_hidden": float("inf")}, {"init_scale": -1},
    {"x_min": 2, "x_max": 1}, {"t_max": 0}, {"x_min": float("nan")},
])
def test_invalid_model_hyperparameters_are_rejected(options):
    with pytest.raises((ValueError, TypeError)):
        PINN(**options)


@pytest.mark.parametrize("omega", [0.0, 1.0, -0.1, float("nan"), float("inf")])
def test_invalid_breather_frequency_is_rejected(omega):
    with pytest.raises(ValueError):
        breather(sample_points(), omega=omega)
    with pytest.raises(ValueError):
        initial_velocity(sample_points()[:, :1], omega=omega)


def test_normalization_can_be_disabled():
    model = PINN("siren", hidden_layers=1, width=1, omega_first=1.0, normalize=False).double()
    layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    with torch.no_grad():
        layers[0].weight.copy_(torch.tensor([[2.0, 3.0]], dtype=torch.float64))
        layers[0].bias.zero_()
        layers[1].weight.fill_(1.0)
        layers[1].bias.zero_()
    points = sample_points()
    torch.testing.assert_close(model(points), (2.0 * points[:, :1] + 3.0 * points[:, 1:]).sin())
