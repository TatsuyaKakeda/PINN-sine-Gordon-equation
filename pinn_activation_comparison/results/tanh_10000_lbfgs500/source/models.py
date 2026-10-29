"""Comparable fully connected PINNs with physical-coordinate inputs."""

import math
from numbers import Integral, Real

import torch
from torch import Tensor, nn

from .activations import GaussianSine, Sine, _finite_real


class PINN(nn.Module):
    """A scalar network mapping physical coordinates ``[x, t]`` to phi.

    Inputs are optionally mapped from the physical rectangle to [-1, 1]^2
    inside ``forward``, preserving the chain rule for PDE derivatives.

    SIREN and Gaussian-sine use identical initialization: the first weights
    are uniform in +/-1/fan_in; subsequent weights (including the output)
    are uniform in +/-sqrt(6/fan_in)/omega_hidden. ``init_scale`` multiplies
    these weight bounds. Tanh uses Xavier-uniform weights with the same scale.
    All biases retain PyTorch's ``nn.Linear`` default initialization, matching
    the official SIREN Colab's SineLayer and linear-output conventions.
    """

    def __init__(
        self,
        model_type: str = "tanh",
        hidden_layers: int = 4,
        width: int = 64,
        omega_first: float = 30.0,
        omega_hidden: float = 30.0,
        gamma: float = 0.0,
        x_min: float = -10.0,
        x_max: float = 10.0,
        t_max: float = 20.0,
        normalize: bool = True,
        init_scale: float = 1.0,
    ) -> None:
        """Validate architecture and initialize the requested activation family."""
        super().__init__()
        if model_type not in {"tanh", "siren", "gaussian"}:
            raise ValueError("model_type must be 'tanh', 'siren', or 'gaussian'")
        for name, value in (("hidden_layers", hidden_layers), ("width", width)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a boolean")
        for name, value in (("x_min", x_min), ("x_max", x_max)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        if x_max <= x_min:
            raise ValueError("x_max must be greater than x_min")

        self.model_type = model_type
        self.width = int(width)
        self.num_hidden_layers = int(hidden_layers)
        self.omega_first = _finite_real("omega_first", omega_first, positive=True)
        self.omega_hidden = _finite_real("omega_hidden", omega_hidden, positive=True)
        self.gamma = _finite_real("gamma", gamma)
        self.init_scale = _finite_real("init_scale", init_scale, positive=True)
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.t_max = _finite_real("t_max", t_max, positive=True)
        self.normalize = normalize

        self.hidden_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        for index in range(self.num_hidden_layers):
            layer = nn.Linear(2 if index == 0 else self.width, self.width)
            self._initialize_weights(layer, first=index == 0)
            self.hidden_layers.append(layer)
            omega = self.omega_first if index == 0 else self.omega_hidden
            if model_type == "tanh":
                activation = nn.Tanh()
            elif model_type == "siren":
                activation = Sine(omega)
            else:
                activation = GaussianSine(omega, self.gamma)
            self.activations.append(activation)
        self.output_layer = nn.Linear(self.width, 1)
        self._initialize_weights(self.output_layer, first=False)

    def _initialize_weights(self, layer: nn.Linear, *, first: bool) -> None:
        """Apply Xavier or SIREN weight bounds while retaining default biases."""
        with torch.no_grad():
            if self.model_type == "tanh":
                nn.init.xavier_uniform_(layer.weight, gain=self.init_scale)
            else:
                bound = 1.0 / layer.in_features if first else (
                    math.sqrt(6.0 / layer.in_features) / self.omega_hidden
                )
                layer.weight.uniform_(-self.init_scale * bound, self.init_scale * bound)

    def forward(self, points: Tensor) -> Tensor:
        """Map physical [x, t] inputs to phi, preserving coordinate gradients."""
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2), with columns [x, t]")
        if not points.is_floating_point():
            raise TypeError("points must be floating-point physical coordinates")
        if self.normalize:
            # Compute scales in the input dtype, including after model.double().
            # Storing precomputed float32 scales would lose float64 precision.
            hidden = torch.stack((
                2.0 * (points[:, 0] - self.x_min) / (self.x_max - self.x_min) - 1.0,
                2.0 * points[:, 1] / self.t_max - 1.0,
            ), dim=1)
        else:
            hidden = points
        for layer, activation in zip(self.hidden_layers, self.activations, strict=True):
            hidden = activation(layer(hidden))
        return self.output_layer(hidden)
