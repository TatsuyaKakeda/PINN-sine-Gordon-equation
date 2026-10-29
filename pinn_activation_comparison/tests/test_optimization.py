"""Full-pool L-BFGS gradients, repeatable state, and numerical failures."""

from copy import deepcopy
import io
import math

import pytest
import torch

from pinn_activation_comparison.config import ExperimentConfig
from pinn_activation_comparison.losses import full_losses, loss_terms
from pinn_activation_comparison.models import PINN
from pinn_activation_comparison.optimization import (
    NumericalFailure,
    full_batch_backward,
    lbfgs_step,
    make_lbfgs,
)
from pinn_activation_comparison.sampling import training_points


def tiny_problem(kind="tanh", **overrides):
    settings = dict(
        width=5, hidden_layers=2, n_f=11, n_ic=9, n_bc=7,
        dtype="float64", eval_batch_size=4, lambda_pde=0.7,
        lambda_ic=1.3, lambda_bc=2.1,
    )
    settings.update(overrides)
    config = ExperimentConfig(**settings)
    # These are explicit to make the unit tests independent of preset tuning.
    config.lbfgs_lr = 0.7
    config.lbfgs_max_iter = 3
    config.lbfgs_history_size = 5
    torch.manual_seed(17)
    model = PINN(kind, width=config.width, hidden_layers=config.hidden_layers,
                 gamma=0.0, omega_first=4.0, omega_hidden=3.0).double()
    return model, training_points(config, 7), config


def norm(parameters):
    return math.sqrt(sum(p.grad.norm().item() ** 2 for p in parameters if p.grad is not None))


@pytest.mark.parametrize("kind", ["tanh", "siren", "gaussian"])
@pytest.mark.parametrize("chunk_size", [1, 4, 64])
def test_chunked_backward_matches_full_pool_losses_and_gradients(kind, chunk_size):
    model, samples, config = tiny_problem(kind, eval_batch_size=chunk_size)
    reference = loss_terms(model, samples, config)
    reference["total_loss"].backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    rng_state = torch.get_rng_state().clone()

    actual = full_batch_backward(model, samples, config)

    assert actual == pytest.approx({key: value.item() for key, value in reference.items()},
                                  rel=1e-12, abs=1e-12)
    assert actual == pytest.approx(full_losses(model, samples, config), rel=1e-12, abs=1e-12)
    for parameter, gradient in zip(model.parameters(), gradients, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, rtol=1e-11, atol=1e-12)
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert all(not points.requires_grad for points in samples.values())


def test_lbfgs_uses_configured_strong_wolfe_and_full_pool_starting_diagnostics():
    model, samples, config = tiny_problem()
    expected_losses = full_batch_backward(model, samples, config)
    expected_norms = {
        "grad_norm": norm(model.parameters()),
        "grad_norm_first": norm(model.hidden_layers[0].parameters()),
        "grad_norm_last": norm(model.output_layer.parameters()),
    }
    optimizer = make_lbfgs(model, config)
    assert isinstance(optimizer, torch.optim.LBFGS)
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["line_search_fn"] == "strong_wolfe"
    assert group["lr"] == config.lbfgs_lr
    assert group["max_iter"] == config.lbfgs_max_iter
    assert group["history_size"] == config.lbfgs_history_size

    row = lbfgs_step(optimizer, model, samples, config)

    for key, value in {**expected_losses, **expected_norms}.items():
        assert row[key] == pytest.approx(value, rel=1e-12, abs=1e-12)
    assert row["closure_evaluations"] >= 2
    assert row["closure_grad_norm_max"] >= row["grad_norm"]
    assert optimizer.state[next(model.parameters())]["func_evals"] == row["closure_evaluations"]
    assert full_losses(model, samples, config)["total_loss"] < expected_losses["total_loss"]


def test_saved_lbfgs_state_continues_identically():
    model, samples, config = tiny_problem()
    optimizer = make_lbfgs(model, config)
    lbfgs_step(optimizer, model, samples, config)
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = deepcopy(model)
    restored.load_state_dict(saved["model"])
    restored_optimizer = make_lbfgs(restored, config)
    restored_optimizer.load_state_dict(saved["optimizer"])

    expected = lbfgs_step(optimizer, model, samples, config)
    actual = lbfgs_step(restored_optimizer, restored, samples, config)

    assert actual == expected
    for left, right in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


@pytest.mark.parametrize("threshold", ["divergence_threshold", "gradient_threshold"])
def test_threshold_failure_retains_first_and_latest_diagnostics(threshold):
    model, samples, config = tiny_problem()
    setattr(config, threshold, 1e-30)
    optimizer = make_lbfgs(model, config)
    before = [parameter.clone() for parameter in model.parameters()]

    with pytest.raises(NumericalFailure) as caught:
        lbfgs_step(optimizer, model, samples, config)

    assert caught.value.status == "diverged"
    diagnostics = caught.value.diagnostics
    assert diagnostics["closure_evaluations"] == 1
    assert diagnostics["first_losses"] == diagnostics["latest_losses"]
    assert diagnostics["total_loss"] == diagnostics["first_losses"]["total_loss"]
    assert diagnostics["closure_grad_norm_max"] == diagnostics["grad_norm"]
    for parameter, initial in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(parameter, initial, atol=0, rtol=0)


@pytest.mark.parametrize("corruption", ["loss", "gradient"])
def test_nonfinite_losses_and_gradients_are_detected(corruption):
    model, samples, config = tiny_problem()
    if corruption == "loss":
        with torch.no_grad():
            model.output_layer.bias.fill_(float("nan"))
    else:
        model.output_layer.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))

    with pytest.raises(NumericalFailure) as caught:
        lbfgs_step(make_lbfgs(model, config), model, samples, config)

    assert caught.value.status == "nonfinite"
    assert caught.value.diagnostics["closure_evaluations"] == 1
    assert not math.isfinite(caught.value.diagnostics["grad_norm"])


def test_gamma_zero_matches_siren_lbfgs_trajectory():
    siren, samples, config = tiny_problem("siren")
    gaussian, _, _ = tiny_problem("gaussian")
    siren_optimizer = make_lbfgs(siren, config)
    gaussian_optimizer = make_lbfgs(gaussian, config)
    for _ in range(2):
        assert lbfgs_step(siren_optimizer, siren, samples, config) == lbfgs_step(
            gaussian_optimizer, gaussian, samples, config)
    for left, right in zip(siren.parameters(), gaussian.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_failed_trial_restores_model_and_optimizer_history():
    model, samples, config = tiny_problem(eval_batch_size=64)
    optimizer = make_lbfgs(model, config)
    lbfgs_step(optimizer, model, samples, config)
    reference = deepcopy(model)
    reference_optimizer = make_lbfgs(reference, config)
    reference_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    calls = 0

    def fail_on_second_closure(gradient):
        nonlocal calls
        calls += 1
        # Each closure backpropagates once through each of the three pools.
        return torch.full_like(gradient, float("inf")) if calls == 4 else gradient

    hook = model.output_layer.weight.register_hook(fail_on_second_closure)
    with pytest.raises(NumericalFailure) as caught:
        lbfgs_step(optimizer, model, samples, config)
    hook.remove()

    diagnostics = caught.value.diagnostics
    assert caught.value.status == "nonfinite"
    assert diagnostics["closure_evaluations"] == 2
    assert math.isfinite(diagnostics["grad_norm"])
    assert not math.isfinite(diagnostics["latest_grad_norm"])
    assert not math.isfinite(diagnostics["closure_grad_norm_max"])
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Verify rollback of all quasi-Newton history, not only the weights.
    assert lbfgs_step(optimizer, model, samples, config) == lbfgs_step(
        reference_optimizer, reference, samples, config)
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_exact_field_is_only_used_at_dirichlet_boundaries(monkeypatch):
    import pinn_activation_comparison.optimization as optimization

    model, samples, config = tiny_problem()
    actual_breather = optimization.breather
    observed = []

    def boundary_only(points, omega):
        assert torch.all((points[:, 0] == config.x_min) | (points[:, 0] == config.x_max))
        observed.append(points.detach().clone())
        return actual_breather(points, omega)

    monkeypatch.setattr(optimization, "breather", boundary_only)
    full_batch_backward(model, samples, config)
    torch.testing.assert_close(torch.cat(observed), samples["bc"], atol=0, rtol=0)
