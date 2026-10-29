"""Reproducible Adam comparisons with shared physics points and diagnostics."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import platform
import random
import shutil
import time

import numpy as np
import torch

from .config import ExperimentConfig, parse_args
from .evaluate import evaluate_model
from .losses import full_losses, loss_terms
from .models import PINN
from .reporting import aggregate_metrics, write_csv, write_json
from .sampling import training_points


def variants(config):
    """Yield independent model conditions; gamma is meaningful only for Gaussian."""
    for kind in config.models:
        for gamma in config.gammas if kind == "gaussian" else (None,):
            yield kind, gamma


def model_arguments(config, kind, gamma):
    """Map shared settings into the network constructor without hidden tuning."""
    return dict(model_type=kind, hidden_layers=config.hidden_layers, width=config.width,
                omega_first=config.omega_first, omega_hidden=config.omega_hidden,
                gamma=gamma if gamma is not None else 0.0,
                x_min=config.x_min, x_max=config.x_max, t_max=config.t_max,
                normalize=config.normalize, init_scale=config.init_scale)


def seed_everything(seed):
    """Reset parameter RNGs per variant; sampling has separate local generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synchronize(device):
    """Synchronize accelerator work so reported training times are comparable."""
    if str(device) == "mps":
        torch.mps.synchronize()


def choose_device(config):
    """Probe second derivatives AND their parameter backward before using MPS.

    A failed probe selects CPU for the entire comparison. Float64 is deliberately
    run on CPU because MPS does not support double precision.
    """
    if config.device == "cpu":
        return "cpu", None
    if config.dtype == "float64":
        return "cpu", "MPS does not support float64; using CPU for the entire comparison."
    if not torch.backends.mps.is_available():
        return "cpu", "MPS is unavailable; using CPU for the entire comparison."
    try:
        probe_config = replace(config, n_f=4, n_ic=4, n_bc=4)
        samples = {key: value.to("mps") for key, value in training_points(probe_config, 0).items()}
        with torch.random.fork_rng(devices=[]):
            for kind, gamma in variants(config):
                model = PINN(**model_arguments(config, kind, gamma)).to("mps")
                optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
                terms = loss_terms(model, samples, config)
                terms["total_loss"].backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()):
                    raise RuntimeError("MPS derivative probe produced nonfinite gradients")
                optimizer.step()
                synchronize("mps")
        return "mps", None
    except (RuntimeError, NotImplementedError) as error:
        return "cpu", f"MPS derivative/backward probe failed: {type(error).__name__}: {error}"


def gradient_norm(parameters):
    """Compute the unclipped Euclidean gradient norm across a parameter group."""
    return math.sqrt(sum(float(torch.linalg.vector_norm(p.grad.detach()).item())**2
                         for p in parameters if p.grad is not None))


def run_one(config, kind, gamma, seed, samples, run_dir, device, fallback_reason):
    """Train one condition and persist final weights, history, and diagnostics."""
    run_dir.mkdir(parents=True)
    seed_everything(seed)
    kwargs = model_arguments(config, kind, gamma)
    # Initialize on CPU before moving to the device: common initial draws.
    model = PINN(**kwargs).to(device=device, dtype=getattr(torch, config.dtype))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    points = {key: value.to(device) for key, value in samples.items()}
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
    history = []
    status, has_nonfinite, divergence_detected = "completed", False, False
    epochs_completed, spike_count = 0, 0
    train_seconds, evaluation_seconds = 0.0, 0.0
    first_target_epoch, first_target_seconds = None, None
    eval_start = time.perf_counter()
    initial_metrics, _ = evaluate_model(model, config)
    evaluation_seconds += time.perf_counter() - eval_start
    if initial_metrics["relative_l2"] <= config.target_error:
        first_target_epoch, first_target_seconds = 0, 0.0
    write_json(run_dir / "config.json", {**config.to_dict(), "model": kind,
                                        "gamma": gamma, "seed": seed, "actual_device": device})

    for epoch in range(1, config.epochs + 1):
        synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if 0 < config.batch_size < config.n_f:
            indices = torch.randperm(config.n_f, generator=batch_generator)[:config.batch_size].to(device)
            batch = {**points, "f": points["f"][indices]}
        else:
            batch = points
        losses = loss_terms(model, batch, config)
        losses["total_loss"].backward()
        norm = gradient_norm(model.parameters())
        first_norm = gradient_norm(model.hidden_layers[0].parameters())
        last_norm = gradient_norm(model.output_layer.parameters())
        row = {"epoch": epoch, **{name: loss.item() for name, loss in losses.items()},
               "grad_norm": norm, "grad_norm_first": first_norm, "grad_norm_last": last_norm,
               "relative_l2": None, "residual_rms": None}
        if history and row["total_loss"] > 10 * max(history[-1]["total_loss"], 1e-30):
            spike_count += 1
        finite_step = all(math.isfinite(row[key]) for key in (*losses.keys(), "grad_norm", "grad_norm_first", "grad_norm_last"))
        if not finite_step:
            status, has_nonfinite = "nonfinite", True
        elif row["total_loss"] > config.divergence_threshold or norm > config.gradient_threshold:
            status, divergence_detected = "diverged", True
        else:
            optimizer.step()
            epochs_completed = epoch
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                status, has_nonfinite = "nonfinite", True
        synchronize(device)
        train_seconds += time.perf_counter() - start
        row["train_seconds"] = train_seconds
        row["step_status"] = status
        history.append(row)
        if status != "completed":
            break
        if epoch % config.eval_every == 0 or epoch == config.epochs:
            start = time.perf_counter()
            evaluation, _ = evaluate_model(model, config)
            evaluation_seconds += time.perf_counter() - start
            row.update(evaluation)
            if not all(math.isfinite(value) for value in evaluation.values()):
                status, has_nonfinite = "nonfinite", True
                row["step_status"] = status
                break
            if first_target_epoch is None and evaluation["relative_l2"] <= config.target_error:
                first_target_epoch, first_target_seconds = epoch, train_seconds

    start = time.perf_counter()
    final_losses = full_losses(model, points, config)
    final_evaluation, arrays = evaluate_model(model, config)
    evaluation_seconds += time.perf_counter() - start
    if not all(math.isfinite(value) for value in (*final_losses.values(), *final_evaluation.values())):
        status, has_nonfinite = "nonfinite", True
    elif final_losses["total_loss"] > config.divergence_threshold:
        status, divergence_detected = "diverged", True
    norms = [row["grad_norm"] for row in history]
    finite_norms = [value for value in norms if math.isfinite(value)]
    metrics = {"model": kind, "gamma": gamma, "seed": seed, "comparison": config.comparison,
               "status": status, "has_nonfinite": has_nonfinite,
               "divergence_detected": divergence_detected, "epochs_completed": epochs_completed,
               "epochs_requested": config.epochs, "device": device, "fallback_reason": fallback_reason,
               "train_seconds": train_seconds, "evaluation_seconds": evaluation_seconds,
               **final_losses, **final_evaluation,
               "initial_relative_l2": initial_metrics["relative_l2"],
               "initial_residual_rms": initial_metrics["residual_rms"],
               "grad_norm_initial": norms[0], "grad_norm_final": norms[-1],
               "grad_norm_min": min(finite_norms, default=float("nan")),
               "grad_norm_max": max(finite_norms, default=float("nan")),
               "loss_spike_count": spike_count,
               "near_zero_gradient_steps": sum(value < 1e-12 for value in finite_norms),
               "target_error": config.target_error, "first_target_epoch": first_target_epoch,
               "first_target_train_seconds": first_target_seconds,
               "parameter_count": sum(p.numel() for p in model.parameters())}
    write_csv(run_dir / "history.csv", history)
    write_json(run_dir / "metrics.json", metrics)
    # Keep nonfinite arrays too: failures remain inspectable and are excluded
    # from aggregate success statistics and finite plot scale calculations.
    np.savez_compressed(run_dir / "predictions.npz", **arrays)
    torch.save({"model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "model_kwargs": kwargs, "config": config.to_dict(),
                "seed": seed, "status": status, "epochs_completed": epochs_completed}, run_dir / "checkpoint.pt")
    print(f"{run_dir.name}: {status}; L2={metrics['relative_l2']:.6g}, "
          f"RMS={metrics['residual_rms']:.6g}, train={train_seconds:.2f}s", flush=True)
    return metrics


def run_experiment(config):
    """Run all selected conditions sequentially; never overwrite an experiment."""
    output = Path(config.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {output}. Choose a new --output-dir.")
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    device, fallback_reason = choose_device(config)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sample_sets = {seed: training_points(config, seed) for seed in config.seeds}
    for seed, points in sample_sets.items():
        np.savez_compressed(output / f"points_seed_{seed}.npz", **{key: value.numpy() for key, value in points.items()})

    def run_suite(actual_device, reason):
        """Keep every condition on one backend, including after a restart."""
        records = []
        for seed in config.seeds:
            for kind, gamma in variants(config):
                label = f"{kind}" + (f"_gamma_{gamma!r}" if gamma is not None else "") + f"_seed_{seed}"
                records.append(run_one(config, kind, gamma, seed, sample_sets[seed],
                                       output / "runs" / label, actual_device, reason))
        return records

    try:
        records = run_suite(device, fallback_reason)
    except (RuntimeError, NotImplementedError) as error:
        if device != "mps":
            raise
        # Preserve attempted MPS files; restart ALL conditions from the same seeds.
        fallback_reason = f"MPS runtime failure; complete suite restarted on CPU: {type(error).__name__}: {error}"
        if (output / "runs").exists():
            (output / "runs").rename(output / "mps_failed_attempt")
        device = "cpu"
        records = run_suite(device, fallback_reason)
    write_csv(output / "summary.csv", records)
    write_csv(output / "aggregate.csv", aggregate_metrics(records))
    source_dir = output / "source"
    source_dir.mkdir()
    source_root = Path(__file__).parent
    hashes = {}
    for source in sorted(source_root.glob("*.py")) + [source_root / "requirements.txt", source_root / "README.md"]:
        if source.is_file():
            shutil.copy2(source, source_dir / source.name)
            hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {"config": config.to_dict(), "actual_device": device, "fallback_reason": fallback_reason,
                "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                "numpy": np.__version__, "platform": platform.platform(),
                                "machine": platform.machine(), "threads": config.threads},
                "source_sha256": hashes,
                "loss_history_convention": "pre-update batch loss; evaluation columns are post-update",
                "final_loss_convention": "post-update full training pool, weighted by actual sample count",
                "gradient_convention": "unclipped pre-update global/first/output-layer parameter norms",
                "timing_convention": "train_seconds excludes model setup, evaluation, serialization and plotting; includes per-step synchronization",
                "std_convention": "sample standard deviation ddof=1; one completed seed reports descriptive zero",
                "failure_convention": "nonfinite or threshold-exceeding runs retained; only completed finite values aggregated",
                "wall_seconds_before_plots": time.perf_counter() - started}
    write_json(output / "experiment.json", metadata)
    if config.plots:
        from .plot_results import plot_all
        plot_all(output)
    metadata["wall_seconds_total"] = time.perf_counter() - started
    write_json(output / "experiment.json", metadata)
    return output


def main(argv=None):
    """CLI entry point; numerical failures remain visible in saved status counts."""
    config = parse_args(argv)
    try:
        result = run_experiment(config)
    except (ValueError, FileExistsError) as error:
        raise SystemExit(str(error)) from error
    print(f"Results: {result.resolve()}")


if __name__ == "__main__":
    main()
