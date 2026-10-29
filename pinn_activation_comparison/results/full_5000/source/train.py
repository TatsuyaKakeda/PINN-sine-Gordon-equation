"""Matched PINN comparisons with periodic diagnostics and resumable optimizers."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import time

import numpy as np
import torch

from .config import ExperimentConfig, parse_args
from .checkpoints import copy_prior_snapshots, load_resume, run_label, save_checkpoint
from .evaluate import evaluate_model
from .losses import full_losses, loss_terms
from .models import PINN
from .optimization import NumericalFailure, lbfgs_step, make_lbfgs
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
    """Train or continue one condition with coherent optimizer checkpoints."""
    payload, source = load_resume(config, kind, gamma, seed)
    run_dir.mkdir(parents=True)
    seed_everything(seed)
    kwargs = model_arguments(config, kind, gamma)
    model = PINN(**kwargs).to(device=device, dtype=getattr(torch, config.dtype))
    adam = torch.optim.Adam(model.parameters(), lr=config.lr)
    lbfgs = None
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
    history, evaluations = [], []
    state = dict(adam_epoch=0, lbfgs_step=0, adam_train_seconds=0.0,
                 lbfgs_train_seconds=0.0, evaluation_seconds=0.0, loss_spike_count=0,
                 first_target_epoch=None, first_target_train_seconds=None,
                 first_target_stage=None, first_target_stage_step=None,
                 lbfgs_closure_evaluations=0, initial_metrics={})
    if payload is not None:
        model.load_state_dict(payload["model_state_dict"])
        adam.load_state_dict(payload["adam_state_dict"])
        batch_generator.set_state(payload["batch_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        state.update(payload["run_state"])
        history, evaluations = payload["history"], payload["evaluations"]
        samples = payload["points"]
        copy_prior_snapshots(source, run_dir, payload)
        if state["lbfgs_step"]:
            lbfgs = make_lbfgs(model, config)
            lbfgs.load_state_dict(payload["lbfgs_state_dict"])
    points = {key: value.to(device) for key, value in samples.items()}
    status, has_nonfinite, divergence_detected = "completed", False, False
    latest_arrays, latest_values = None, None
    write_json(run_dir / "config.json", {**config.to_dict(), "model": kind,
                                        "gamma": gamma, "seed": seed, "actual_device": device,
                                        "resume_checkpoint": str(source) if source else None})

    def training_seconds():
        """Return accumulated optimizer time across both stages and resumes."""
        return state["adam_train_seconds"] + state["lbfgs_train_seconds"]

    def assess(values):
        """Record numerical failure without discarding its artifacts."""
        nonlocal status, has_nonfinite, divergence_detected
        if not all(math.isfinite(value) for value in values.values()):
            status, has_nonfinite = "nonfinite", True
        elif values.get("total_loss", 0) > config.divergence_threshold:
            status, divergence_detected = "diverged", True

    def evaluate(stage, step):
        """Save post-update full-pool losses and independent validation traces."""
        nonlocal latest_arrays, latest_values
        start = time.perf_counter()
        losses = full_losses(model, points, config)
        metrics, latest_arrays = evaluate_model(model, config)
        state["evaluation_seconds"] += time.perf_counter() - start
        latest_values = {**losses, **metrics}
        assess(latest_values)
        global_step = state["adam_epoch"] + state["lbfgs_step"]
        row = dict(epoch=global_step, stage=stage, stage_step=step,
                   train_seconds=training_seconds(), **latest_values)
        # A resumed endpoint may be refreshed, but is never duplicated.
        if evaluations and (evaluations[-1]["stage"], evaluations[-1]["stage_step"]) == (stage, step):
            evaluations[-1] = row
        else:
            evaluations.append(row)
        if not state["initial_metrics"]:
            state["initial_metrics"] = metrics.copy()
        if state["first_target_epoch"] is None and metrics["relative_l2"] <= config.target_error:
            state.update(first_target_epoch=global_step, first_target_train_seconds=training_seconds(),
                         first_target_stage=stage, first_target_stage_step=step)
        snapshot_dir = run_dir / "snapshots"
        snapshot_dir.mkdir(exist_ok=True)
        np.savez_compressed(snapshot_dir / f"{stage}_{step:06d}.npz",
                            **{key: value for key, value in latest_arrays.items()
                               if key.startswith("center_") or key.startswith("ic_")})
        if config.plots:
            from .plot_results import save_center_snapshot
            save_center_snapshot(run_dir, latest_arrays, stage, step, run_dir.name)
        write_csv(run_dir / "evaluations.csv", evaluations)
        write_csv(run_dir / "history.csv", history)
        print(f"{run_dir.name}: {stage} {step}; L2={metrics['relative_l2']:.6g}, "
              f"IC velocity L2={metrics['ic_velocity_relative_l2']:.6g}", flush=True)
        return metrics

    def checkpoint(path, checkpoint_status="running"):
        """Save all state needed to continue the exact update sequence."""
        stored = dict(format_version=2, condition=dict(model=kind, gamma=gamma, seed=seed),
                      model_kwargs=kwargs, config=config.to_dict(), seed=seed, status=checkpoint_status,
                      epochs_completed=state["adam_epoch"], actual_device=device,
                      model_state_dict={name: value.detach().cpu() for name, value in model.state_dict().items()},
                      adam_state_dict=adam.state_dict(),
                      lbfgs_state_dict=lbfgs.state_dict() if lbfgs is not None else None,
                      batch_rng_state=batch_generator.get_state(), torch_rng_state=torch.get_rng_state(),
                      points={key: value.detach().cpu() for key, value in samples.items()},
                      history=history, evaluations=evaluations, run_state=state.copy())
        save_checkpoint(path, stored)

    def periodic_checkpoint(stage, step):
        """Retain a named checkpoint and atomically advance the latest pointer."""
        checkpoint(run_dir / "checkpoints" / f"{stage}_{step:06d}.pt")
        checkpoint(run_dir / "checkpoints" / "latest.pt")
        write_csv(run_dir / "history.csv", history)
        write_csv(run_dir / "evaluations.csv", evaluations)

    def summarize(protocol, stage):
        """Describe the accepted weights, optimizer work, and observed stability."""
        norms = [row["grad_norm"] for row in history]
        finite_norms = [value for value in norms if math.isfinite(value)]
        return dict(model=kind, gamma=gamma, seed=seed, comparison=config.comparison,
                    optimizer=protocol, stage=stage, status=status, has_nonfinite=has_nonfinite,
                    divergence_detected=divergence_detected, epochs_completed=state["adam_epoch"],
                    epochs_requested=config.epochs, lbfgs_steps_completed=state["lbfgs_step"],
                    lbfgs_steps_requested=config.lbfgs_steps if protocol == "adam_lbfgs" else 0,
                    optimizer_steps_completed=state["adam_epoch"] + state["lbfgs_step"],
                    lbfgs_closure_evaluations=state["lbfgs_closure_evaluations"],
                    device=device, fallback_reason=fallback_reason,
                    train_seconds=training_seconds(), adam_train_seconds=state["adam_train_seconds"],
                    lbfgs_train_seconds=state["lbfgs_train_seconds"], evaluation_seconds=state["evaluation_seconds"],
                    **latest_values, initial_relative_l2=state["initial_metrics"]["relative_l2"],
                    initial_residual_rms=state["initial_metrics"]["residual_rms"],
                    grad_norm_initial=norms[0] if norms else float("nan"),
                    grad_norm_final=norms[-1] if norms else float("nan"),
                    grad_norm_min=min(finite_norms, default=float("nan")),
                    grad_norm_max=max(finite_norms, default=float("nan")),
                    loss_spike_count=state["loss_spike_count"],
                    near_zero_gradient_steps=sum(value < 1e-12 for value in finite_norms),
                    target_error=config.target_error, first_target_epoch=state["first_target_epoch"],
                    first_target_stage=state["first_target_stage"],
                    first_target_stage_step=state["first_target_stage_step"],
                    first_target_train_seconds=state["first_target_train_seconds"],
                    parameter_count=sum(p.numel() for p in model.parameters()))

    # Refresh the restored checkpoint to keep endpoint artifacts self-contained.
    stage = "lbfgs" if state["lbfgs_step"] else "adam"
    evaluate(stage, state["lbfgs_step"] if stage == "lbfgs" else state["adam_epoch"])
    if status == "completed":
        periodic_checkpoint(stage, state["lbfgs_step"] if stage == "lbfgs" else state["adam_epoch"])
    for epoch in range(state["adam_epoch"] + 1, config.epochs + 1):
        if status != "completed":
            break
        synchronize(device)
        start = time.perf_counter()
        adam.zero_grad(set_to_none=True)
        if 0 < config.batch_size < config.n_f:
            indices = torch.randperm(config.n_f, generator=batch_generator)[:config.batch_size].to(device)
            batch = {**points, "f": points["f"][indices]}
        else:
            batch = points
        losses = loss_terms(model, batch, config)
        losses["total_loss"].backward()
        row = dict(epoch=epoch, stage="adam", stage_step=epoch,
                   **{name: loss.item() for name, loss in losses.items()},
                   grad_norm=gradient_norm(model.parameters()),
                   grad_norm_first=gradient_norm(model.hidden_layers[0].parameters()),
                   grad_norm_last=gradient_norm(model.output_layer.parameters()),
                   relative_l2=None, residual_rms=None)
        assess({key: row[key] for key in (*losses, "grad_norm", "grad_norm_first", "grad_norm_last")})
        if status == "completed" and row["grad_norm"] > config.gradient_threshold:
            status, divergence_detected = "diverged", True
        if status == "completed":
            adam.step()
            state["adam_epoch"] = epoch
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                status, has_nonfinite = "nonfinite", True
        synchronize(device)
        state["adam_train_seconds"] += time.perf_counter() - start
        if history and history[-1]["stage"] == "adam" and row["total_loss"] > 10 * max(history[-1]["total_loss"], 1e-30):
            state["loss_spike_count"] += 1
        row.update(train_seconds=training_seconds(), step_status=status)
        history.append(row)
        latest_values = None  # The weights may have changed since evaluation.
        if status != "completed":
            break
        if epoch % config.eval_every == 0 or epoch == config.epochs:
            row.update(evaluate("adam", epoch))
            row["step_status"] = status
        if status == "completed" and (epoch % config.checkpoint_every == 0 or epoch == config.epochs):
            periodic_checkpoint("adam", epoch)

    if status == "completed" and config.optimizer == "adam_lbfgs":
        if state["lbfgs_step"] == 0:
            if latest_values is None:
                evaluate("adam", state["adam_epoch"])
            write_json(run_dir / "adam_metrics.json", summarize("adam", "adam"))
            np.savez_compressed(run_dir / "adam_predictions.npz", **latest_arrays)
            checkpoint(run_dir / "adam_checkpoint.pt", "completed")
            lbfgs = make_lbfgs(model, config)
            evaluate("lbfgs", 0)
        for step in range(state["lbfgs_step"] + 1, config.lbfgs_steps + 1):
            if status != "completed":
                break
            synchronize(device)
            start = time.perf_counter()
            try:
                diagnostics = lbfgs_step(lbfgs, model, points, config)
                state["lbfgs_step"] = step
            except NumericalFailure as error:
                status = error.status
                has_nonfinite = status == "nonfinite"
                divergence_detected = status == "diverged"
                diagnostics = {key: value for key, value in error.diagnostics.items()
                               if isinstance(value, (float, int))}
                write_json(run_dir / "failure.json", dict(stage="lbfgs", step=step,
                                                          reason=str(error), **error.diagnostics))
            synchronize(device)
            state["lbfgs_train_seconds"] += time.perf_counter() - start
            state["lbfgs_closure_evaluations"] += diagnostics["closure_evaluations"]
            row = dict(epoch=state["adam_epoch"] + step, stage="lbfgs", stage_step=step,
                       **diagnostics, relative_l2=None, residual_rms=None,
                       train_seconds=training_seconds(), step_status=status)
            if history and history[-1]["stage"] == "lbfgs" and row["total_loss"] > 10 * max(history[-1]["total_loss"], 1e-30):
                state["loss_spike_count"] += 1
            history.append(row)
            latest_values = None
            if status != "completed":
                break
            if step % config.lbfgs_eval_every == 0 or step == config.lbfgs_steps:
                row.update(evaluate("lbfgs", step))
                row["step_status"] = status
                if status == "completed":
                    periodic_checkpoint("lbfgs", step)
        stage = "lbfgs"
    else:
        stage = "adam"
    if latest_values is None:
        evaluate(stage, state["lbfgs_step"] if stage == "lbfgs" else state["adam_epoch"])
    metrics = summarize(config.optimizer, stage)
    write_csv(run_dir / "history.csv", history)
    write_csv(run_dir / "evaluations.csv", evaluations)
    write_json(run_dir / "metrics.json", metrics)
    np.savez_compressed(run_dir / "predictions.npz", **latest_arrays)
    checkpoint(run_dir / "checkpoint.pt", status)
    print(f"{run_dir.name}: {status}; L2={metrics['relative_l2']:.6g}, "
          f"RMS={metrics['residual_rms']:.6g}, train={training_seconds():.2f}s", flush=True)
    return metrics


def run_experiment(config):
    """Run matched conditions; preserve source experiments on continuation."""
    from .checkpoints import validate_resume_source

    output = Path(config.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {output}. Choose a new --output-dir.")
    validate_resume_source(config)
    # Validate every selected run before writing any new artifacts. Stored pools
    # are authoritative, including for not-yet-started variants of the same seed.
    sample_sets = {}
    for seed in config.seeds:
        for kind, gamma in variants(config):
            saved, _ = load_resume(config, kind, gamma, seed)
            if saved is not None:
                if seed in sample_sets and any(not torch.equal(sample_sets[seed][key], saved["points"][key])
                                               for key in ("f", "ic", "bc")):
                    raise ValueError(f"Resume runs have different training points for seed {seed}")
                sample_sets[seed] = saved["points"]
        if seed not in sample_sets:
            sample_sets[seed] = training_points(config, seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    device, fallback_reason = choose_device(config)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for seed, points in sample_sets.items():
        np.savez_compressed(output / f"points_seed_{seed}.npz", **{key: value.numpy() for key, value in points.items()})
    source_dir = output / "source"
    source_dir.mkdir()
    source_root = Path(__file__).parent
    hashes = {}
    for source in sorted(source_root.glob("*.py")) + [source_root / "requirements.txt", source_root / "README.md"]:
        if source.is_file():
            shutil.copy2(source, source_dir / source.name)
            hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {"config": config.to_dict(), "status": "running", "actual_device": device,
                "fallback_reason": fallback_reason,
                "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                "numpy": np.__version__, "platform": platform.platform(),
                                "machine": platform.machine(), "threads": config.threads},
                "source_sha256": hashes,
                "loss_history_convention": "pre-update Adam minibatch / first L-BFGS closure; evaluation columns post-update",
                "evaluation_convention": "evaluations.csv: post-update entire fixed training pools and held-out grid",
                "final_loss_convention": "post-update full training pool, weighted by actual sample count",
                "step_convention": "epochs_completed counts Adam only; history epoch counts all updates; stage_step separates Adam epochs and L-BFGS outer steps",
                "gradient_convention": "unclipped pre-update global/first/output-layer norms; L-BFGS also tracks max norm across closures",
                "timing_convention": "train_seconds excludes setup, evaluation, serialization and plotting; includes synchronization and L-BFGS rollback snapshots; cumulative across resumes",
                "std_convention": "sample standard deviation ddof=1; one completed seed reports descriptive zero",
                "failure_convention": "failed runs retained; only completed finite values aggregated; interruption resumes last complete checkpoint"}
    write_json(output / "experiment.json", metadata)

    def run_suite(actual_device, reason):
        """Keep every condition on one backend, including after a restart."""
        records = []
        for seed in config.seeds:
            for kind, gamma in variants(config):
                records.append(run_one(config, kind, gamma, seed, sample_sets[seed],
                                       output / "runs" / run_label(kind, gamma, seed), actual_device, reason))
                write_csv(output / "summary.csv", records)
                write_csv(output / "aggregate.csv", aggregate_metrics(records))
        return records

    try:
        try:
            records = run_suite(device, fallback_reason)
        except (RuntimeError, NotImplementedError) as error:
            if device != "mps":
                raise
            fallback_reason = f"MPS runtime failure; complete suite restarted on CPU: {type(error).__name__}: {error}"
            if (output / "runs").exists():
                (output / "runs").rename(output / "mps_failed_attempt")
            device = "cpu"
            metadata.update(actual_device=device, fallback_reason=fallback_reason)
            write_json(output / "experiment.json", metadata)
            records = run_suite(device, fallback_reason)
    except BaseException as error:
        metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        failure=f"{type(error).__name__}: {error}",
                        wall_seconds_before_plots=time.perf_counter() - started)
        write_json(output / "experiment.json", metadata)
        raise
    adam_records = [json.loads(path.read_text()) for path in sorted((output / "runs").glob("*/adam_metrics.json"))]
    if adam_records:
        write_csv(output / "summary_adam.csv", adam_records)
        write_csv(output / "aggregate_adam.csv", aggregate_metrics(adam_records))
    metadata.update(status="completed" if all(row.get("status") == "completed" for row in records) else "completed_with_failures",
                    wall_seconds_before_plots=time.perf_counter() - started)
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
