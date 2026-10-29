"""Interrupted trials never replace coherent periodic optimizer checkpoints."""

from dataclasses import replace
import json

import pytest
import torch

from pinn_activation_comparison.config import ExperimentConfig, parse_args
import pinn_activation_comparison.train as training


def tiny(path, **changes):
    values = dict(models=("tanh",), seeds=(0,), epochs=4, hidden_layers=1, width=4,
                  n_f=9, n_ic=5, n_bc=3, batch_size=3, eval_nx=5, eval_nt=7,
                  eval_batch_size=4, eval_every=2, checkpoint_every=2,
                  dtype="float64", plots=False, output_dir=str(path))
    values.update(changes)
    return ExperimentConfig(**values)


def load(path):
    return torch.load(path, weights_only=True, map_location="cpu")


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for x, y in zip(left, right, strict=True):
            assert_nested_equal(x, y)
    else:
        assert left == right


def assert_same_trajectory(left, right):
    for key in ("model_state_dict", "adam_state_dict", "lbfgs_state_dict",
                "batch_rng_state", "torch_rng_state", "points"):
        assert_nested_equal(left[key], right[key])
    assert len(left["history"]) == len(right["history"])
    for a, b in zip(left["history"], right["history"], strict=True):
        for key in ("stage", "stage_step", "total_loss", "grad_norm"):
            assert a[key] == b[key]


@pytest.mark.parametrize("interrupt_at,checkpoint_epoch", [(1, 0), (3, 2)])
def test_adam_interrupt_resumes_last_atomic_checkpoint(tmp_path, monkeypatch, interrupt_at, checkpoint_epoch):
    config = tiny(tmp_path / "interrupted")
    reference_root = training.run_experiment(replace(config, output_dir=str(tmp_path / "reference")))
    original_step = torch.optim.Adam.step
    calls = 0

    def interrupted_step(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        result = original_step(optimizer, *args, **kwargs)
        if calls == interrupt_at:
            # A changed model from this unfinished update must not be saved.
            raise KeyboardInterrupt("simulated interruption during Adam")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.Adam, "step", interrupted_step)
        with pytest.raises(KeyboardInterrupt):
            training.run_experiment(config)

    output = tmp_path / "interrupted"
    run = output / "runs" / "tanh_seed_0"
    assert json.loads((output / "experiment.json").read_text())["status"] == "interrupted"
    assert not (run / "checkpoint.pt").exists()
    latest = load(run / "checkpoints" / "latest.pt")
    assert latest["status"] == "running"
    assert latest["run_state"]["adam_epoch"] == checkpoint_epoch
    assert len(latest["history"]) == checkpoint_epoch
    resumed_config = parse_args(["--resume", str(output), "--output-dir", str(tmp_path / "resumed")])
    resumed = training.run_experiment(resumed_config)
    assert_same_trajectory(load(reference_root / "runs" / "tanh_seed_0" / "checkpoint.pt"),
                           load(resumed / "runs" / "tanh_seed_0" / "checkpoint.pt"))


def test_lbfgs_interruption_discards_uncheckpointed_update(tmp_path, monkeypatch):
    config = tiny(tmp_path / "interrupted", epochs=2, optimizer="adam_lbfgs",
                  lbfgs_steps=4, lbfgs_max_iter=2, lbfgs_history_size=4, lbfgs_eval_every=1)
    reference_root = training.run_experiment(replace(config, output_dir=str(tmp_path / "reference")))
    original_step = training.lbfgs_step
    calls = 0

    def interrupted_step(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original_step(*args, **kwargs)
        if calls == 3:
            raise KeyboardInterrupt("simulated interruption during L-BFGS")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(training, "lbfgs_step", interrupted_step)
        with pytest.raises(KeyboardInterrupt):
            training.run_experiment(config)

    output = tmp_path / "interrupted"
    run = output / "runs" / "tanh_seed_0"
    assert json.loads((output / "experiment.json").read_text())["status"] == "interrupted"
    latest = load(run / "checkpoints" / "latest.pt")
    assert latest["run_state"]["lbfgs_step"] == 2
    assert len(latest["history"]) == 4
    assert not (run / "checkpoint.pt").exists()
    resumed = training.run_experiment(replace(config, resume=str(output), output_dir=str(tmp_path / "resumed")))
    assert_same_trajectory(load(reference_root / "runs" / "tanh_seed_0" / "checkpoint.pt"),
                           load(resumed / "runs" / "tanh_seed_0" / "checkpoint.pt"))


def test_failed_lbfgs_retains_failure_and_prior_resumable_checkpoint(tmp_path, monkeypatch):
    config = tiny(tmp_path / "failed", epochs=2, optimizer="adam_lbfgs",
                  lbfgs_steps=4, lbfgs_max_iter=2, lbfgs_history_size=4, lbfgs_eval_every=1)
    reference_root = training.run_experiment(replace(config, output_dir=str(tmp_path / "reference")))
    original_step = training.lbfgs_step
    calls = 0

    def failing_step(optimizer, model, samples, step_config):
        nonlocal calls
        calls += 1
        if calls == 3:
            step_config = replace(step_config, gradient_threshold=1e-30)
        return original_step(optimizer, model, samples, step_config)

    with monkeypatch.context() as patch:
        patch.setattr(training, "lbfgs_step", failing_step)
        failed_root = training.run_experiment(config)

    run = failed_root / "runs" / "tanh_seed_0"
    final = load(run / "checkpoint.pt")
    latest_path = run / "checkpoints" / "latest.pt"
    latest = load(latest_path)
    metrics = json.loads((run / "metrics.json").read_text())
    failure = json.loads((run / "failure.json").read_text())
    assert final["status"] == metrics["status"] == "diverged"
    assert metrics["divergence_detected"] is True
    assert final["run_state"]["lbfgs_step"] == latest["run_state"]["lbfgs_step"] == 2
    assert latest["status"] == "running"
    assert failure["step"] == 3
    assert failure["closure_evaluations"] == 1
    assert "first_losses" in failure and "latest_losses" in failure
    assert final["history"][-1]["step_status"] == "diverged"
    assert_nested_equal(final["model_state_dict"], latest["model_state_dict"])
    assert_nested_equal(final["lbfgs_state_dict"], latest["lbfgs_state_dict"])
    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match="diverged"):
        training.run_experiment(replace(config, resume=str(failed_root), output_dir=str(rejected)))
    assert not rejected.exists()
    resumed = training.run_experiment(replace(config, resume=str(latest_path), output_dir=str(tmp_path / "resumed")))
    assert_same_trajectory(load(reference_root / "runs" / "tanh_seed_0" / "checkpoint.pt"),
                           load(resumed / "runs" / "tanh_seed_0" / "checkpoint.pt"))
