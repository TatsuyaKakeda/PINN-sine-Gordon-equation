"""Validated JSON/CLI settings and matched quick/full experiment presets."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
import math
from numbers import Real


@dataclass
class ExperimentConfig:
    """Shared settings; an epoch denotes one Adam update on a PDE minibatch."""

    mode: str = "quick"
    models: tuple[str, ...] = ("tanh", "siren", "gaussian")
    gammas: tuple[float, ...] = (0.0, 0.001, 0.01, 0.05)
    seeds: tuple[int, ...] = (0,)
    epochs: int = 100
    hidden_layers: int = 4
    width: int = 64
    lr: float = 1e-3
    n_f: int = 256
    n_ic: int = 64
    n_bc: int = 64
    batch_size: int = 128
    lambda_pde: float = 1.0
    lambda_ic: float = 1.0
    lambda_bc: float = 1.0
    omega: float = 0.8
    omega_first: float = 30.0
    omega_hidden: float = 30.0
    init_scale: float = 1.0
    normalize: bool = True
    x_min: float = -10.0
    x_max: float = 10.0
    t_max: float = 20.0
    device: str = "cpu"
    dtype: str = "float32"
    eval_nx: int = 81
    eval_nt: int = 81
    eval_batch_size: int = 512
    eval_every: int = 20
    threads: int = 1
    output_dir: str = "pinn_activation_comparison/results/quick"
    plots: bool = True
    comparison: str = "controlled"
    target_error: float = 0.1
    divergence_threshold: float = 1e12
    gradient_threshold: float = 1e8

    def __post_init__(self):
        """Reject invalid settings before allocating networks or output files."""
        self.models, self.gammas, self.seeds = map(tuple, (self.models, self.gammas, self.seeds))
        for name in ("lr", "omega", "omega_first", "omega_hidden", "init_scale",
                     "t_max", "x_min", "x_max", "target_error", "divergence_threshold",
                     "gradient_threshold", "lambda_pde", "lambda_ic", "lambda_bc"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number, not a boolean")
        for name in ("epochs", "hidden_layers", "width", "n_f", "n_ic", "n_bc",
                     "eval_batch_size", "eval_every", "threads", "eval_nx", "eval_nt"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.eval_nx < 3 or self.eval_nt < 3:
            raise ValueError("Evaluation grids need at least 3 points per axis")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 0:
            raise ValueError("batch_size must be an integer >= 0 (0 means full batch)")
        for name in ("lr", "omega_first", "omega_hidden", "init_scale", "t_max",
                     "target_error", "divergence_threshold", "gradient_threshold"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("lambda_pde", "lambda_ic", "lambda_bc"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.lambda_pde + self.lambda_ic + self.lambda_bc == 0:
            raise ValueError("At least one loss weight must be positive")
        if not math.isfinite(self.omega) or not 0 < self.omega < 1:
            raise ValueError("omega must lie strictly between 0 and 1")
        if not all(math.isfinite(x) for x in (self.x_min, self.x_max)) or self.x_min >= self.x_max:
            raise ValueError("x_min and x_max must be finite with x_min < x_max")
        if not self.x_min <= 0 <= self.x_max:
            raise ValueError("The domain must include x=0 for the requested center diagnostic")
        for name, options in (("mode", ("quick", "full")), ("device", ("cpu", "mps", "auto")),
                              ("dtype", ("float32", "float64")),
                              ("comparison", ("controlled", "tuned"))):
            if getattr(self, name) not in options:
                raise ValueError(f"Invalid {name}: {getattr(self, name)}")
        if not self.models or any(m not in ("tanh", "siren", "gaussian") for m in self.models):
            raise ValueError("models must contain tanh, siren, and/or gaussian")
        if not self.gammas or any(isinstance(g, bool) or not isinstance(g, Real) or not math.isfinite(g) or g < 0 for g in self.gammas):
            raise ValueError("gammas must contain finite nonnegative values")
        if not self.seeds or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in self.seeds):
            raise ValueError("seeds must be integers in [0, 2**32)")
        for name in ("models", "gammas", "seeds"):
            if len(set(getattr(self, name))) != len(getattr(self, name)):
                raise ValueError(f"Duplicate {name} would overwrite runs")
        if not isinstance(self.normalize, bool) or not isinstance(self.plots, bool):
            raise ValueError("normalize and plots must be booleans")
        self.output_dir = str(self.output_dir)
        if not self.output_dir.strip():
            raise ValueError("output_dir must be nonempty")

    def to_dict(self):
        """Return JSON-compatible settings."""
        return asdict(self)


def parse_args(argv=None):
    """Apply preset < explicit JSON values < explicit CLI values, in that order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON object using ExperimentConfig field names")
    parser.add_argument("--mode", choices=("quick", "full"))
    parser.add_argument("--model", choices=("all", "tanh", "siren", "gaussian"))
    parser.add_argument("--gamma", type=float, nargs="+", dest="gammas")
    parser.add_argument("--seed", type=int, nargs="+", dest="seeds")
    default = ExperimentConfig()
    skip = {"mode", "models", "gammas", "seeds", "normalize", "plots"}
    for field in fields(default):
        if field.name not in skip:
            parser.add_argument("--" + field.name.replace("_", "-"), type=type(getattr(default, field.name)))
    parser.add_argument("--no-plots", action="store_false", dest="plots", default=None)
    parser.add_argument("--no-normalize", action="store_false", dest="normalize", default=None)
    args = vars(parser.parse_args(argv))
    config_file = args.pop("config")
    data = {}
    if config_file:
        try:
            with open(config_file, encoding="utf-8") as stream:
                data = json.load(stream)
            if not isinstance(data, dict):
                raise ValueError("Config JSON must be an object")
            unknown = data.keys() - default.to_dict().keys()
            if unknown:
                raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        except (OSError, ValueError) as error:
            parser.error(str(error))
    mode = args.get("mode") or data.get("mode", "quick")
    values = default.to_dict()
    if mode == "full":
        values.update(epochs=4000, n_f=4096, n_ic=256, n_bc=256, batch_size=512,
                      eval_nx=201, eval_nt=201, eval_every=200,
                      output_dir="pinn_activation_comparison/results/full")
    values.update(data)
    model = args.pop("model")
    values.update({key: value for key, value in args.items() if value is not None})
    if model:
        values["models"] = default.models if model == "all" else (model,)
    try:
        return ExperimentConfig(**values)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
