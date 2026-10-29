"""End-to-end continuation and optimizer-stage monitoring contracts."""

import csv
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from pinn_activation_comparison.config import ExperimentConfig, parse_args
from pinn_activation_comparison.train import run_experiment


def tiny(path, **changes):
    """Build a cheap experiment with genuinely stochastic PDE minibatches."""
    values = dict(models=("siren", "gaussian"), gammas=(0.0,), seeds=(0, 1, 2),
                  epochs=4, hidden_layers=2, width=8, n_f=17, n_ic=7, n_bc=5,
                  batch_size=8, eval_nx=9, eval_nt=9, eval_batch_size=13,
                  eval_every=2, checkpoint_every=2, plots=False, output_dir=str(path))
    values.update(changes)
    return ExperimentConfig(**values)


def rows(path):
    """Read a saved table."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def checkpoint(run, name="checkpoint.pt"):
    """Load a portable checkpoint without arbitrary pickle classes."""
    return torch.load(run / name, weights_only=True, map_location="cpu")


def assert_same_state(left, right):
    """Check bitwise model and sampling RNG equivalence."""
    for key in left["model_state_dict"]:
        torch.testing.assert_close(left["model_state_dict"][key], right["model_state_dict"][key], rtol=0, atol=0)
    assert torch.equal(left["batch_rng_state"], right["batch_rng_state"])
    for key in left["points"]:
        assert torch.equal(left["points"][key], right["points"][key])


def test_three_seed_adam_resume_is_identical_and_monitoring_is_saved(tmp_path):
    config = tiny(tmp_path / "continuous")
    original = run_experiment(config)
    first = run_experiment(replace(config, epochs=2, output_dir=str(tmp_path / "first")))
    continued = run_experiment(replace(config, resume=str(first), output_dir=str(tmp_path / "continued")))
    for source in sorted((original / "runs").iterdir()):
        resumed = continued / "runs" / source.name
        assert_same_state(checkpoint(source), checkpoint(resumed))
        a, b = rows(source / "history.csv"), rows(resumed / "history.csv")
        for x, y in zip(a, b, strict=True):
            for key in ("epoch", "stage", "stage_step", "total_loss", "grad_norm"):
                assert x[key] == y[key]
        evaluated = rows(resumed / "evaluations.csv")
        assert [int(r["stage_step"]) for r in evaluated] == [0, 2, 4]
        for row in evaluated:
            for key in ("relative_l2", "pde_loss", "ic_loss", "bc_loss", "ic_velocity_loss",
                        "ic_velocity_relative_l2", "center_amplitude_ratio"):
                assert np.isfinite(float(row[key]))
        for step in (0, 2, 4):
            assert (resumed / "snapshots" / f"adam_{step:06d}.npz").is_file()
        assert (resumed / "checkpoints" / "adam_000004.pt").is_file()
        assert checkpoint(resumed)["run_state"]["adam_epoch"] == 4
    assert len(rows(continued / "aggregate.csv")) == 2
    assert {r["n_completed"] for r in rows(continued / "aggregate.csv")} == {"3"}


def test_resume_intermediate_file_discards_future_history(tmp_path):
    original = run_experiment(tiny(tmp_path / "source", models=("siren",), seeds=(0,)))
    run = original / "runs" / "siren_seed_0"
    config = parse_args(["--resume", str(run / "checkpoints" / "adam_000002.pt"),
                         "--epochs", "3", "--output-dir", str(tmp_path / "resumed")])
    resumed = run_experiment(config) / "runs" / "siren_seed_0"
    assert [int(r["epoch"]) for r in rows(resumed / "history.csv")] == [1, 2, 3]
    assert not (resumed / "snapshots" / "adam_000004.npz").exists()


def test_hybrid_keeps_adam_endpoint_and_resumes_lbfgs(tmp_path):
    config = tiny(tmp_path / "hybrid", models=("siren",), seeds=(0,), epochs=2,
                  optimizer="adam_lbfgs", lbfgs_steps=4, lbfgs_max_iter=2,
                  lbfgs_eval_every=1, lbfgs_history_size=4)
    full = run_experiment(config)
    first = run_experiment(replace(config, lbfgs_steps=2, output_dir=str(tmp_path / "first")))
    continued = run_experiment(replace(config, resume=str(first), output_dir=str(tmp_path / "continued")))
    full_run = full / "runs" / "siren_seed_0"
    continued_run = continued / "runs" / "siren_seed_0"
    assert_same_state(checkpoint(full_run), checkpoint(continued_run))
    stages = [r["stage"] for r in rows(continued_run / "history.csv")]
    assert stages == ["adam"] * 2 + ["lbfgs"] * 4
    pure = run_experiment(replace(config, optimizer="adam", output_dir=str(tmp_path / "pure")))
    assert_same_state(checkpoint(full_run, "adam_checkpoint.pt"), checkpoint(pure / "runs" / "siren_seed_0"))
    adam_metrics = json.loads((continued_run / "adam_metrics.json").read_text())
    assert adam_metrics["optimizer"] == "adam"
    final = json.loads((continued_run / "metrics.json").read_text())
    assert final["optimizer"] == "adam_lbfgs"
    assert final["lbfgs_steps_completed"] == 4
    assert final["lbfgs_closure_evaluations"] >= 4
    assert (continued / "summary_adam.csv").is_file()
    assert (continued / "aggregate_adam.csv").is_file()
    with pytest.raises(ValueError, match="Adam|epochs"):
        run_experiment(replace(config, resume=str(first), epochs=3, output_dir=str(tmp_path / "invalid")))


def test_changed_training_condition_is_rejected_before_creating_output(tmp_path):
    config = tiny(tmp_path / "first", models=("siren",), seeds=(0,), epochs=2)
    source = run_experiment(config)
    destination = tmp_path / "wrong"
    with pytest.raises(ValueError, match="lr"):
        run_experiment(replace(config, resume=str(source), lr=0.002, output_dir=str(destination)))
    assert not destination.exists()
