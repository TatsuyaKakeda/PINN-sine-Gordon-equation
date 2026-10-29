"""Deterministic, memory-bounded full-pool L-BFGS updates for the PINNs."""

from copy import deepcopy
import math

import torch

from .losses import combine_losses, initial_losses
from .physics import breather, pde_residual


class NumericalFailure(RuntimeError):
    """A terminal numerical failure with inspectable closure diagnostics.

    ``status`` is ``nonfinite`` or ``diverged``. ``diagnostics`` contains the
    first-closure history fields, closure count/maximum gradient norm, and
    ``first_losses`` / ``latest_losses`` plus the latest gradient norms.
    """

    def __init__(self, status, diagnostics, reason):
        """Attach machine-readable status and closure diagnostics."""
        super().__init__(reason)
        self.status = status
        self.diagnostics = diagnostics


def make_lbfgs(model, config, state_dict=None):
    """Create one persistent strong-Wolfe optimizer for a full-pool stage.

    ``lbfgs_max_iter`` limits inner quasi-Newton iterations in each outer
    ``step``; PyTorch's default ``max_eval`` additionally bounds evaluations.
    All parameters share the single group required by PyTorch L-BFGS. Optional
    state restores the quasi-Newton history. The explicitly configured stopping
    tolerance is reapplied after loading, so a refinement is not overwritten by
    the saved parameter group. No model weights or curvature history are reset.
    """
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=config.lbfgs_lr, max_iter=config.lbfgs_max_iter,
        history_size=config.lbfgs_history_size, line_search_fn="strong_wolfe",
        tolerance_change=config.lbfgs_tolerance_change,
    )
    if state_dict is not None:
        optimizer.load_state_dict(state_dict)
        for group in optimizer.param_groups:
            group['tolerance_change'] = config.lbfgs_tolerance_change
    return optimizer


def full_batch_backward(model, samples, config):
    """Reset gradients and accumulate the exact fixed-pool weighted objective.

    Each chunk contributes its actual sample fraction within its own pool,
    including the two-sided boundary pool. Backpropagating immediately frees
    the derivative graph before the next chunk. Returned losses are detached
    Python floats; gradients remain on model parameters. No points are drawn
    here, and analytic field values supervise only the boundary pool.
    """
    model.zero_grad(set_to_none=True)
    parts = dict.fromkeys(("pde_loss", "ic_value_loss", "ic_velocity_loss", "bc_loss"), 0.0)
    for group in ("f", "ic", "bc"):
        points = samples[group]
        for chunk in points.split(config.eval_batch_size):
            weight = len(chunk) / len(points)
            if group == "f":
                loss = pde_residual(model, chunk).square().mean()
                parts["pde_loss"] += loss.detach().item() * weight
                objective = config.lambda_pde * weight * loss
            elif group == "ic":
                value, velocity = initial_losses(model, chunk, config.omega)
                parts["ic_value_loss"] += value.detach().item() * weight
                parts["ic_velocity_loss"] += velocity.detach().item() * weight
                objective = config.lambda_ic * weight * (value + velocity)
            else:
                target = breather(chunk, config.omega).detach()
                loss = (model(chunk) - target).square().mean()
                parts["bc_loss"] += loss.detach().item() * weight
                objective = config.lambda_bc * weight * loss
            objective.backward()
    return combine_losses(parts, config)


def _gradient_norm(parameters):
    """Return an unclipped Euclidean norm, avoiding scalar-square overflow."""
    return math.hypot(*(torch.linalg.vector_norm(p.grad.detach()).item()
                        for p in parameters if p.grad is not None))


def lbfgs_step(optimizer, model, samples, config):
    """Run one outer step and report first-closure, pre-update diagnostics.

    Every closure uses the same full pools, validates losses and gradients,
    and enforces both configured divergence limits. ``closure_grad_norm_max``
    includes all trial points. PyTorch returns the starting loss from ``step``;
    it is deliberately ignored. Callers must evaluate final losses separately.

    A numerical failure restores parameters and optimizer state to before the
    outer step, then raises ``NumericalFailure`` with the failed trial details.
    The rollback snapshot adds one model copy and one optimizer-state copy to
    peak memory. Failures remain terminal and must not be marked completed.
    """
    parameters = tuple(model.parameters())
    saved_parameters = [p.detach().clone() for p in parameters]
    saved_optimizer = deepcopy(optimizer.state_dict())
    first = latest = None
    closure_evaluations = 0
    closure_grad_norm_max = 0.0

    def diagnostics():
        """Package first and failed trial measurements for the caller."""
        return {
            **first,
            "closure_evaluations": closure_evaluations,
            "closure_grad_norm_max": closure_grad_norm_max,
            "first_losses": {key: value for key, value in first.items() if key.endswith("_loss")},
            "latest_losses": {key: value for key, value in latest.items() if key.endswith("_loss")},
            **{"latest_" + key: latest[key]
               for key in ("grad_norm", "grad_norm_first", "grad_norm_last")},
        }

    def closure():
        """Evaluate the deterministic full-pool objective and its gradients."""
        nonlocal first, latest, closure_evaluations, closure_grad_norm_max
        closure_evaluations += 1
        losses = full_batch_backward(model, samples, config)
        latest = {
            **losses,
            "grad_norm": _gradient_norm(parameters),
            "grad_norm_first": _gradient_norm(model.hidden_layers[0].parameters()),
            "grad_norm_last": _gradient_norm(model.output_layer.parameters()),
        }
        if first is None:
            first = latest.copy()
        norm = latest["grad_norm"]
        closure_grad_norm_max = (float("nan") if math.isnan(norm)
                                 else max(closure_grad_norm_max, norm))
        if not all(math.isfinite(value) for value in latest.values()):
            raise NumericalFailure("nonfinite", diagnostics(),
                                   "L-BFGS closure produced a nonfinite loss or gradient")
        if losses["total_loss"] > config.divergence_threshold or norm > config.gradient_threshold:
            raise NumericalFailure("diverged", diagnostics(),
                                   "L-BFGS closure exceeded a loss or gradient threshold")
        return losses["total_loss"]

    try:
        optimizer.step(closure)
        if not all(torch.isfinite(p).all().item() for p in parameters):
            raise NumericalFailure("nonfinite", diagnostics(),
                                   "L-BFGS update produced nonfinite parameters")
    except NumericalFailure:
        with torch.no_grad():
            for parameter, saved in zip(parameters, saved_parameters, strict=True):
                parameter.copy_(saved)
        optimizer.load_state_dict(saved_optimizer)
        model.zero_grad(set_to_none=True)
        raise
    return {**first, "closure_evaluations": closure_evaluations,
            "closure_grad_norm_max": closure_grad_norm_max}
