"""Exact sine-Gordon breather and differentiable physical PDE operators."""

import math
from collections.abc import Callable
from numbers import Real

import torch
from torch import Tensor


def _breather_scale(omega: float) -> float:
    if isinstance(omega, bool) or not isinstance(omega, Real):
        raise ValueError("omega must be a finite real number strictly between 0 and 1")
    if not math.isfinite(omega) or not 0.0 < omega < 1.0:
        raise ValueError("omega must be a finite real number strictly between 0 and 1")
    return math.sqrt(1.0 - float(omega) ** 2)


def _validate_points(points: Tensor) -> None:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2), with columns [x, t]")
    if not points.is_floating_point():
        raise TypeError("points must be floating-point physical coordinates")


def breather(points: Tensor, omega: float = 0.8) -> Tensor:
    """Exact ``4 atan(k sin(omega t) / (omega cosh(k x)))``, k=sqrt(1-omega²)."""
    _validate_points(points)
    k = _breather_scale(omega)
    x, t = points[:, :1], points[:, 1:]
    return 4.0 * torch.atan(k * torch.sin(omega * t) / (omega * torch.cosh(k * x)))


def initial_velocity(x: Tensor, omega: float = 0.8) -> Tensor:
    """Exact phi_t(x, 0) = 4 sqrt(1-omega²) / cosh(sqrt(1-omega²) x)."""
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError("x must have shape (N, 1)")
    if not x.is_floating_point():
        raise TypeError("x must contain floating-point physical coordinates")
    k = _breather_scale(omega)
    return 4.0 * k / torch.cosh(k * x)


def _gradient(values: Tensor, points: Tensor) -> Tensor:
    """Differentiate pointwise fields, retaining a graph even for constants."""
    gradient = None
    if values.requires_grad:
        gradient = torch.autograd.grad(
            values, points, grad_outputs=torch.ones_like(values),
            create_graph=True, retain_graph=True, allow_unused=True,
        )[0]
    zeros = points * 0.0
    return zeros if gradient is None else gradient + zeros


def field_derivatives(model: Callable[[Tensor], Tensor], points: Tensor) -> dict[str, Tensor]:
    """Return phi, phi_x, phi_t, phi_xx, phi_tt, each shaped (N, 1).

    Derivatives are with respect to the physical input columns, and retain
    the higher-order graph required for backpropagating a PINN PDE loss.
    The caller's ``requires_grad`` flag is left unchanged. This pointwise
    operator assumes the model does not couple different batch samples.
    """
    _validate_points(points)
    if not points.requires_grad:
        points = points.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        phi = model(points)
        if phi.shape != (points.shape[0], 1):
            raise ValueError("model must return a scalar field with shape (N, 1)")
        first = _gradient(phi, points)
        phi_x, phi_t = first[:, :1], first[:, 1:]
        phi_xx = _gradient(phi_x, points)[:, :1]
        phi_tt = _gradient(phi_t, points)[:, 1:]
    return {"phi": phi, "phi_x": phi_x, "phi_t": phi_t, "phi_xx": phi_xx, "phi_tt": phi_tt}


def pde_residual(model: Callable[[Tensor], Tensor], points: Tensor) -> Tensor:
    """Return the sine-Gordon residual phi_tt - phi_xx + sin(phi)."""
    fields = field_derivatives(model, points)
    return fields["phi_tt"] - fields["phi_xx"] + torch.sin(fields["phi"])
