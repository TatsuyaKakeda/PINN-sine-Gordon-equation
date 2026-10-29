"""Chunked diagnostics on a fixed grid independent of random training points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .physics import breather, pde_residual


def evaluate_model(model, config):
    """Return relative field error, physical residual RMS, and plot arrays.

    All grid values include both endpoints. Global L2 is an unweighted discrete
    norm across the same grid for every model, as specified in the experiment.
    Graphs are retained only within a residual chunk and never backpropagated
    into optimizer state. Exact grid data are evaluation-only.
    """
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    x = np.linspace(config.x_min, config.x_max, config.eval_nx)
    t = np.linspace(0, config.t_max, config.eval_nt)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    coordinates = np.column_stack((xx.ravel(), tt.ravel()))

    def predict(points, with_residual=False):
        """Evaluate a coordinate array without retaining cross-chunk graphs."""
        predictions, exacts, residuals = [], [], []
        for start in range(0, len(points), config.eval_batch_size):
            chunk = torch.as_tensor(points[start:start + config.eval_batch_size], device=device, dtype=dtype)
            with torch.no_grad():
                predictions.append(model(chunk).cpu().numpy().ravel())
                exacts.append(breather(chunk, config.omega).cpu().numpy().ravel())
            if with_residual:
                residuals.append(pde_residual(model, chunk).detach().cpu().numpy().ravel())
        return np.concatenate(predictions), np.concatenate(exacts), (np.concatenate(residuals) if with_residual else None)

    was_training = model.training
    model.eval()
    try:
        predicted, exact, residual = predict(coordinates, with_residual=True)
        denominator = np.linalg.norm(exact.astype(np.float64))
        if denominator == 0:
            raise ValueError("Evaluation grid samples only zeros of the exact solution; change eval_nt")
        difference = predicted.astype(np.float64) - exact.astype(np.float64)
        metrics = {"relative_l2": float(np.linalg.norm(difference) / denominator),
                   "residual_rms": float(np.sqrt(np.mean(residual.astype(np.float64)**2)))}
        center_coordinates = np.column_stack((np.zeros_like(t), t))
        center_predicted, center_exact, _ = predict(center_coordinates)
        # Include nonzero phases of the breather, plus endpoints and long-time behavior.
        profile_times = np.unique(np.clip(np.array([0, np.pi / (2 * config.omega),
                                                    3 * np.pi / (2 * config.omega),
                                                    0.75 * config.t_max, config.t_max]), 0, config.t_max))
        px, pt = np.meshgrid(x, profile_times, indexing="xy")
        profiles, profiles_exact, _ = predict(np.column_stack((px.ravel(), pt.ravel())))
        arrays = {"x": x, "t": t, "exact": exact.reshape(tt.shape),
                  "prediction": predicted.reshape(tt.shape),
                  "abs_error": np.abs(predicted - exact).reshape(tt.shape),
                  "residual": residual.reshape(tt.shape),
                  "center_t": t, "center_exact": center_exact, "center_prediction": center_predicted,
                  "profile_times": profile_times, "profile_x": x,
                  "profile_exact": profiles_exact.reshape(pt.shape),
                  "profile_prediction": profiles.reshape(pt.shape)}
        return metrics, arrays
    finally:
        model.train(was_training)


def main(argv=None):
    """Recompute held-out metrics from one saved checkpoint on CPU."""
    from .models import PINN

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    checkpoint = torch.load(args.run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    config = ExperimentConfig(**checkpoint["config"])
    torch.set_num_threads(config.threads)
    model = PINN(**checkpoint["model_kwargs"]).to(dtype=getattr(torch, config.dtype))
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, _ = evaluate_model(model, config)
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
