"""Evaluate tanh breather amplitude, period, localization and physical energy.

This module is evaluation-only: no exact interior value enters an optimizer.
Run ``python -m pinn_activation_comparison.tanh_diagnostics --run-dir RUN``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .evaluate import evaluate_model
from .models import PINN
from .physics import breather
from .reporting import write_json


def _relative_error(prediction, exact):
    """Compute a discrete relative norm with double-precision accumulation."""
    prediction, exact = np.asarray(prediction, dtype=float), np.asarray(exact, dtype=float)
    denominator = np.linalg.norm(exact)
    return float(np.linalg.norm(prediction - exact) / denominator) if denominator > 0 else None


def _positive_peaks(t, values, threshold):
    """Locate positive maxima, interpolating each peak through three samples."""
    indices = np.flatnonzero((values[1:-1] > values[:-2]) &
                             (values[1:-1] > values[2:]) &
                             (values[1:-1] > threshold)) + 1
    times, heights = [], []
    for index in indices:
        left, middle, right = values[index - 1:index + 2]
        curvature = left - 2 * middle + right
        offset = float(np.clip(.5 * (left - right) / curvature, -1, 1)) if curvature else 0.
        times.append(t[index] + offset * (t[index + 1] - t[index]))
        heights.append(middle - .25 * (left - right) * offset)
    return np.asarray(times), np.asarray(heights)


def physical_diagnostics(model, config):
    """Return held-out field, oscillation and autograd energy measurements.

    Positive peaks must exceed 10% of the exact center peak to avoid assigning
    a period to a nearly zero trace. ``near_zero_center`` uses a descriptive
    center RMS ratio < 0.05, not a convergence certificate. Energy integrates
    0.5*(phi_t^2 + phi_x^2) + 1-cos(phi) on the finite physical domain; exact
    finite-domain energy is computed independently for comparison.
    """
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    was_training = model.training
    model.eval()
    try:
        metrics, arrays = evaluate_model(model, config)
        metrics.update(validation_target=0.01, meets_one_percent=bool(metrics['relative_l2'] < 0.01))
        center_t = np.linspace(0, config.t_max, max(1601, config.eval_nt))
        center_points = np.column_stack((np.zeros_like(center_t), center_t))
        predicted, exact = [], []
        with torch.no_grad():
            for start in range(0, len(center_t), config.eval_batch_size):
                point = torch.as_tensor(center_points[start:start + config.eval_batch_size], device=device, dtype=dtype)
                predicted.append(model(point).cpu().numpy().ravel())
                exact.append(breather(point, config.omega).cpu().numpy().ravel())
        predicted, exact = np.concatenate(predicted), np.concatenate(exact)
        peak_times, peak_values = _positive_peaks(center_t, predicted, .1 * np.max(np.abs(exact)))
        period = float(np.mean(np.diff(peak_times))) if len(peak_times) >= 2 else None
        late = center_t >= .75 * config.t_max
        center_ratio = float(np.linalg.norm(predicted) / np.linalg.norm(exact))
        metrics.update(center_relative_l2=_relative_error(predicted, exact),
                       period_exact=2 * np.pi / config.omega, period_prediction=period,
                       period_relative_error=abs(period / (2 * np.pi / config.omega) - 1) if period else None,
                       positive_peak_count=len(peak_times),
                       center_peak_amplitude_ratio=float(np.max(np.abs(predicted)) / np.max(np.abs(exact))),
                       center_late_rms_ratio=float(np.linalg.norm(predicted[late]) / np.linalg.norm(exact[late])),
                       near_zero_center=bool(center_ratio < .05))
        x = np.linspace(config.x_min, config.x_max, max(201, config.eval_nx))
        energy_t = np.linspace(0, config.t_max, max(101, config.eval_nt))
        xx, tt = np.meshgrid(x, energy_t)
        coordinates = np.column_stack((xx.ravel(), tt.ravel()))
        densities = {"prediction": [], "exact": []}
        for start in range(0, len(coordinates), config.eval_batch_size):
            point = torch.as_tensor(coordinates[start:start + config.eval_batch_size], device=device, dtype=dtype)
            for name, field in (("prediction", model), ("exact", lambda p: breather(p, config.omega))):
                with torch.enable_grad():
                    differentiable = point.detach().requires_grad_(True)
                    value = field(differentiable)
                    gradient = (torch.autograd.grad(value, differentiable, torch.ones_like(value), allow_unused=True)[0]
                                if value.requires_grad else None)
                    if gradient is None:
                        gradient = torch.zeros_like(differentiable)
                    density = .5 * gradient.square().sum(dim=1) + 1 - torch.cos(value[:, 0])
                densities[name].append(density.detach().cpu().numpy())
        energies = {name: np.trapezoid(np.concatenate(values).reshape(xx.shape), x, axis=1)
                    for name, values in densities.items()}
        initial, final = float(energies['prediction'][0]), float(energies['prediction'][-1])
        metrics.update(energy_initial=initial, energy_final=final,
                       energy_final_ratio=final / initial if initial > 1e-20 else None,
                       energy_relative_l2=_relative_error(energies['prediction'], energies['exact']))
        for index, (lower, upper) in enumerate(zip(np.linspace(0, config.t_max, 5)[:-1],
                                                np.linspace(0, config.t_max, 5)[1:])):
            mask = (arrays['t'] >= lower) & ((arrays['t'] <= upper) if index == 3 else (arrays['t'] < upper))
            metrics[f'field_relative_l2_quarter_{index + 1}'] = _relative_error(arrays['prediction'][mask], arrays['exact'][mask])
        arrays.update(center_fine_t=center_t, center_fine_prediction=predicted, center_fine_exact=exact,
                      positive_peak_times=peak_times, positive_peak_values=peak_values,
                      energy_t=energy_t, energy_prediction=energies['prediction'], energy_exact=energies['exact'])
        return metrics, arrays
    finally:
        model.train(was_training)


def main(argv=None):
    """Recompute independent diagnostics from saved weights without training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--eval-nx', type=int, default=401)
    parser.add_argument('--eval-nt', type=int, default=401)
    args = parser.parse_args(argv)
    checkpoint = torch.load(args.run_dir / 'checkpoint.pt', weights_only=True, map_location='cpu')
    config = replace(ExperimentConfig(**checkpoint['config']), eval_nx=args.eval_nx, eval_nt=args.eval_nt)
    torch.set_num_threads(config.threads)
    model = PINN(**checkpoint['model_kwargs']).to(dtype=getattr(torch, config.dtype))
    model.load_state_dict(checkpoint['model_state_dict'])
    metrics, arrays = physical_diagnostics(model, config)
    write_json(args.run_dir / 'physical_diagnostics.json', dict(eval_nx=args.eval_nx, eval_nt=args.eval_nt, **metrics))
    np.savez_compressed(args.run_dir / 'physical_diagnostics.npz', **arrays)
    from .plot_results import plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout='constrained')
    axes[0].plot(arrays['center_fine_t'], arrays['center_fine_exact'], 'k', label='Exact')
    axes[0].plot(arrays['center_fine_t'], arrays['center_fine_prediction'], '--', label='PINN')
    axes[0].set(xlabel='t', ylabel='phi(0,t)', title='Center oscillation')
    axes[1].plot(arrays['energy_t'], arrays['energy_exact'], 'k', label='Exact (finite domain)')
    axes[1].plot(arrays['energy_t'], arrays['energy_prediction'], '--', label='PINN')
    axes[1].set(xlabel='t', ylabel='E(t)', title='Energy from physical first derivatives')
    for axis in axes:
        axis.legend()
        axis.grid(alpha=.2)
    fig.savefig(args.run_dir / 'physical_diagnostics.png', dpi=160)
    plt.close(fig)
    print(f"L2={metrics['relative_l2']:.8g}; center L2={metrics['center_relative_l2']:.8g}; "
          f"period={metrics['period_prediction']}; energy ratio={metrics['energy_final_ratio']}")


if __name__ == '__main__':
    main()
