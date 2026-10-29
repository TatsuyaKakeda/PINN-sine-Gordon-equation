"""Atomic training checkpoints and validation for reproducible continuation."""

from __future__ import annotations

from pathlib import Path
import json
import re
import shutil
import tempfile

import torch

from .config import ExperimentConfig


FORMAT_VERSION = 2
REQUIRED_KEYS = {
    "condition", "model_state_dict", "adam_state_dict", "batch_rng_state",
    "torch_rng_state", "history", "evaluations", "run_state", "points", "config",
    "model_kwargs",
}
MUTABLE_SETTINGS = {
    "epochs", "lbfgs_steps", "output_dir", "resume", "plots", "device", "threads",
    "models", "gammas", "seeds", "optimizer",
}


def run_label(kind, gamma, seed):
    """Keep the established experiment/run directory naming convention."""
    return kind + (f"_gamma_{gamma!r}" if gamma is not None else "") + f"_seed_{seed}"


def _validate_settings(saved_config, config):
    """Validate common conditions, even before any variant has started."""
    saved = ExperimentConfig(**saved_config).to_dict()
    current = config.to_dict()
    for name in current.keys() - MUTABLE_SETTINGS:
        if current[name] != saved[name]:
            raise ValueError(f"Cannot change {name} when resuming: {saved[name]!r} -> {current[name]!r}")
    for name in ("models", "gammas", "seeds"):
        if not set(current[name]).issubset(saved[name]):
            raise ValueError(f"Resume {name} must be a subset of the saved conditions")
    return saved


def validate_resume_source(config):
    """Require valid experiment metadata for a directory continuation."""
    if not config.resume:
        return
    source = Path(config.resume)
    if source.is_dir():
        try:
            metadata = json.loads((source / "experiment.json").read_text())
            _validate_settings(metadata["config"], config)
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid resume manifest {source / 'experiment.json'}: {error}") from error


def validate_resume(payload, config):
    """Reject missing state and changes that would invalidate continuation."""
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            "Resume requires checkpoint format_version=2 with optimizer and RNG state; "
            "legacy weight-only checkpoints can be evaluated but cannot resume training."
        )
    missing = REQUIRED_KEYS - payload.keys()
    if missing:
        raise ValueError(f"Resume checkpoint is missing required state: {sorted(missing)}")
    if payload.get("status") in ("nonfinite", "diverged"):
        raise ValueError(f"Cannot resume a {payload['status']} checkpoint")
    state = payload["run_state"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint run_state must be an object")
    for key in ("adam_epoch", "lbfgs_step"):
        value = state.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Checkpoint run_state.{key} must be a nonnegative integer")
    adam_epoch, lbfgs_step = state["adam_epoch"], state["lbfgs_step"]
    if lbfgs_step > 0 and payload.get("lbfgs_state_dict") is None:
        raise ValueError("A started L-BFGS stage requires lbfgs_state_dict")
    try:
        saved = _validate_settings(payload["config"], config)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid saved checkpoint config: {error}") from error
    if config.optimizer != saved["optimizer"]:
        allowed = saved["optimizer"] == "adam" and config.optimizer == "adam_lbfgs" and lbfgs_step == 0
        if not allowed:
            raise ValueError("Cannot change optimizer protocol after the L-BFGS stage has begun or revert to Adam")
    if config.epochs < adam_epoch:
        raise ValueError(f"epochs ({config.epochs}) cannot be less than completed Adam updates ({adam_epoch})")
    if config.lbfgs_steps < lbfgs_step:
        raise ValueError(f"lbfgs_steps ({config.lbfgs_steps}) cannot be less than completed L-BFGS steps ({lbfgs_step})")
    if lbfgs_step > 0 and config.epochs != saved["epochs"]:
        raise ValueError("Cannot change epochs after L-BFGS has begun")


def load_resume(config, kind, gamma, seed):
    """Load one matching run, or return no state for an unstarted directory run."""
    if not config.resume:
        return None, None
    source = Path(config.resume)
    if source.is_dir():
        run = source / "runs" / run_label(kind, gamma, seed)
        source = run / "checkpoint.pt"
        if not source.is_file():
            source = run / "checkpoints" / "latest.pt"
        if not source.is_file():
            return None, None
    payload = torch.load(source, map_location="cpu", weights_only=True)
    validate_resume(payload, config)
    expected = {"model": kind, "gamma": gamma, "seed": seed}
    if payload["condition"] != expected:
        raise ValueError(f"Checkpoint condition {payload['condition']!r} does not match requested condition {expected!r}")
    return payload, source


def save_checkpoint(path, payload):
    """Replace a checkpoint only after its complete temporary file is written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def copy_prior_snapshots(checkpoint_path, newrun, payload):
    """Copy only artifacts at or before the loaded checkpoint's completed work."""
    source = Path(checkpoint_path).parent
    if source.name == "checkpoints":
        source = source.parent
    destination = Path(newrun)
    state = payload["run_state"]
    limits = {"adam": state["adam_epoch"]}
    if state["lbfgs_step"] > 0:
        limits["lbfgs"] = state["lbfgs_step"]
    for snapshot in (source / "snapshots").glob("*"):
        match = re.fullmatch(r"(?:center_)?(adam|lbfgs)_(\d+)\.(npz|png)", snapshot.name)
        if match and match[1] in limits and int(match[2]) <= limits[match[1]]:
            target = destination / "snapshots" / snapshot.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshot, target)
    if state["lbfgs_step"] > 0:
        for name in ("adam_checkpoint.pt", "adam_metrics.json", "adam_predictions.npz"):
            artifact = source / name
            if artifact.is_file():
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(artifact, destination / name)
