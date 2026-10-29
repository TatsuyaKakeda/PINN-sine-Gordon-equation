"""Backend fallback and inspectable failed-run artifacts without MPS hardware."""

import csv
from functools import wraps
import json

import numpy as np
import pytest
import torch
from torch import nn

from pinn_activation_comparison.config import ExperimentConfig
from pinn_activation_comparison.sampling import training_points
import pinn_activation_comparison.train as train


def tiny_config(tmp_path, **overrides):
    values = dict(
        models=("tanh",), epochs=2, hidden_layers=1, width=4,
        n_f=8, n_ic=5, n_bc=4, batch_size=4,
        eval_nx=5, eval_nt=5, eval_every=1, eval_batch_size=7,
        plots=False, output_dir=str(tmp_path / "results"),
    )
    values.update(overrides)
    return ExperimentConfig(**values)


def test_unavailable_mps_selects_cpu_for_the_comparison(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    for requested in ("mps", "auto"):
        device, reason = train.choose_device(tiny_config(tmp_path, device=requested))
        assert device == "cpu"
        assert "unavailable" in reason
        assert "entire comparison" in reason


def test_float64_selects_cpu_before_the_mps_probe(monkeypatch, tmp_path):
    def unexpected_probe():
        pytest.fail("MPS availability should not be checked for float64")

    monkeypatch.setattr(torch.backends.mps, "is_available", unexpected_probe)
    device, reason = train.choose_device(tiny_config(tmp_path, device="mps", dtype="float64"))
    assert device == "cpu"
    assert "float64" in reason


def test_cpu_request_never_probes_mps(monkeypatch, tmp_path):
    def unexpected_probe():
        pytest.fail("An explicit CPU request must not probe MPS")

    monkeypatch.setattr(torch.backends.mps, "is_available", unexpected_probe)
    assert train.choose_device(tiny_config(tmp_path)) == ("cpu", None)


def test_mps_derivative_probe_failure_has_a_recorded_cpu_reason(monkeypatch, tmp_path):
    class DeviceNeutralModel(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def to(self, *args, **kwargs):
            return self

    def unsupported_derivative(model, samples, config):
        raise NotImplementedError("simulated unsupported second derivative")

    # Adam lazily imports Dynamo, which inspects the availability wrapper.
    @wraps(torch.backends.mps.is_available)
    def available_mps():
        return True

    monkeypatch.setattr(torch.backends.mps, "is_available", available_mps)
    monkeypatch.setattr(train, "PINN", DeviceNeutralModel)
    monkeypatch.setattr(train, "training_points", lambda config, seed: {})
    monkeypatch.setattr(train, "loss_terms", unsupported_derivative)
    device, reason = train.choose_device(tiny_config(tmp_path, device="mps"))
    assert device == "cpu"
    assert "derivative/backward probe failed" in reason
    assert "simulated unsupported second derivative" in reason


def test_runtime_mps_failure_preserves_attempts_and_restarts_every_variant(monkeypatch, tmp_path):
    config = tiny_config(tmp_path, models=("tanh", "siren"), device="mps", seeds=(3,))
    monkeypatch.setattr(train, "choose_device", lambda config: ("mps", None))
    attempts = []
    sample_ids = []

    def simulated_run(config, kind, gamma, seed, samples, run_dir, device, reason):
        attempts.append((kind, device))
        sample_ids.append(id(samples))
        run_dir.mkdir(parents=True)
        (run_dir / "attempt.txt").write_text(device)
        if kind == "siren" and device == "mps":
            raise RuntimeError("simulated late MPS failure")
        if device == "cpu":
            assert "complete suite restarted on CPU" in reason
        return {"model": kind, "gamma": gamma, "seed": seed, "status": "completed",
                "comparison": config.comparison, "device": device, "relative_l2": 0.5}

    monkeypatch.setattr(train, "run_one", simulated_run)
    output = train.run_experiment(config)
    assert attempts == [("tanh", "mps"), ("siren", "mps"), ("tanh", "cpu"), ("siren", "cpu")]
    assert len(set(sample_ids)) == 1
    for kind in config.models:
        name = f"{kind}_seed_3"
        assert (output / "mps_failed_attempt" / name / "attempt.txt").read_text() == "mps"
        assert (output / "runs" / name / "attempt.txt").read_text() == "cpu"
    metadata = json.loads((output / "experiment.json").read_text())
    assert metadata["actual_device"] == "cpu"
    assert "simulated late MPS failure" in metadata["fallback_reason"]
    with (output / "summary.csv").open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 2
    assert {record["device"] for record in summary} == {"cpu"}


def test_divergent_cpu_run_saves_status_and_last_checkpoint(tmp_path):
    config = tiny_config(tmp_path, divergence_threshold=1e-30)
    run_dir = tmp_path / "diverged"
    metrics = train.run_one(config, "tanh", None, 0, training_points(config, 0), run_dir, "cpu", None)
    assert metrics["status"] == "diverged"
    assert metrics["divergence_detected"] is True
    assert metrics["has_nonfinite"] is False
    assert metrics["epochs_completed"] == 0
    stored = json.loads((run_dir / "metrics.json").read_text())
    assert stored["status"] == "diverged"
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["status"] == "diverged"
    assert checkpoint["epochs_completed"] == 0
    with np.load(run_dir / "predictions.npz") as arrays:
        assert np.isfinite(arrays["prediction"]).all()


def test_nonfinite_cpu_run_saves_null_metrics_and_inspectable_arrays(monkeypatch, tmp_path):
    real_pinn = train.PINN

    def nonfinite_model(**kwargs):
        model = real_pinn(**kwargs)
        with torch.no_grad():
            model.output_layer.bias.fill_(float("nan"))
        return model

    monkeypatch.setattr(train, "PINN", nonfinite_model)
    config = tiny_config(tmp_path)
    run_dir = tmp_path / "nonfinite"
    metrics = train.run_one(config, "tanh", None, 0, training_points(config, 0), run_dir, "cpu", None)
    assert metrics["status"] == "nonfinite"
    assert metrics["has_nonfinite"] is True
    assert metrics["epochs_completed"] == 0
    stored = json.loads((run_dir / "metrics.json").read_text())
    assert stored["status"] == "nonfinite"
    assert stored["relative_l2"] is None
    assert stored["total_loss"] is None
    with np.load(run_dir / "predictions.npz") as arrays:
        assert np.isnan(arrays["prediction"]).all()
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["status"] == "nonfinite"
