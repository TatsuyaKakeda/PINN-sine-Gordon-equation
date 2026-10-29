"""Smooth activations and their first two preactivation derivatives."""

import math
from numbers import Real

import torch
from torch import Tensor, nn


def _finite_real(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return value


class Sine(nn.Module):
    """SIREN activation ``sin(omega * z)`` with a fixed frequency."""

    def __init__(self, omega: float = 30.0) -> None:
        super().__init__()
        self.omega = _finite_real("omega", omega, positive=True)

    def forward(self, z: Tensor) -> Tensor:
        return torch.sin(self.omega * z)

    def extra_repr(self) -> str:
        return f"omega={self.omega:g}"


class GaussianSine(nn.Module):
    """``exp(-gamma * (omega * z)**2) * sin(omega * z)``.

    Both gamma and omega are fixed hyperparameters. Gamma zero uses the exact
    same computation as :class:`Sine`, including its automatic derivatives.
    """

    def __init__(self, omega: float = 30.0, gamma: float = 0.0) -> None:
        super().__init__()
        self.omega = _finite_real("omega", omega, positive=True)
        self.gamma = _finite_real("gamma", gamma)

    def forward(self, z: Tensor) -> Tensor:
        phase = self.omega * z
        if self.gamma == 0.0:
            return torch.sin(phase)
        return torch.exp(-self.gamma * phase.square()) * torch.sin(phase)

    def extra_repr(self) -> str:
        return f"omega={self.omega:g}, gamma={self.gamma:g}"


def activation_derivatives(
    z: Tensor, kind: str, omega: float = 1.0, gamma: float = 0.0
) -> tuple[Tensor, Tensor, Tensor]:
    """Return activation value, first derivative, and second derivative in z.

    ``kind`` is ``tanh``, ``siren``, or ``gaussian``. Tanh is the conventional
    unscaled tanh; omega and gamma affect only the sinusoidal activations.
    Analytic formulas make activation plots independent of autograd state.
    """
    omega = _finite_real("omega", omega, positive=True)
    gamma = _finite_real("gamma", gamma)
    if kind == "tanh":
        value = torch.tanh(z)
        first = 1.0 - value.square()
        return value, first, -2.0 * value * first
    if kind not in {"siren", "gaussian"}:
        raise ValueError("kind must be 'tanh', 'siren', or 'gaussian'")

    phase = omega * z
    sine, cosine = torch.sin(phase), torch.cos(phase)
    if kind == "siren" or gamma == 0.0:
        return sine, omega * cosine, -(omega ** 2) * sine
    envelope = torch.exp(-gamma * phase.square())
    value = envelope * sine
    first = omega * envelope * (cosine - 2.0 * gamma * phase * sine)
    second = omega ** 2 * envelope * (
        (4.0 * gamma ** 2 * phase.square() - 2.0 * gamma - 1.0) * sine
        - 4.0 * gamma * phase * cosine
    )
    return value, first, second
