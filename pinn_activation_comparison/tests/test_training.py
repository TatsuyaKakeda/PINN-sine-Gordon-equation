"""Behavioral checks for the CLI, shared samples, and reproducible runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOSS_COLUMNS = (
    "total_loss", "pde_loss", "ic_loss", "ic_value_loss", "ic_velocity_loss",
    "bc_loss", "grad_norm",
)


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _quick_command(output):
    return [
        sys.executable, "-m", "pinn_activation_comparison.train",
        "--mode", "quick", "--epochs", "2", "--width", "8",
        "--hidden-layers", "2", "--n-f", "16", "--n-ic", "8",
        "--n-bc", "8", "--batch-size", "8", "--eval-nx", "9",
        "--eval-nt", "9", "--eval-every", "1", "--no-plots",
        "--device", "cpu", "--output-dir", str(output),
    ]


@pytest.fixture(scope="module")
def quick_runs(tmp_path_factory):
    roots = [tmp_path_factory.mktemp("cli") / name for name in ("first", "repeat")]
    for root in roots:
        result = subprocess.run(
            _quick_command(root), cwd=PROJECT_ROOT, capture_output=True,
            text=True, timeout=180, check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    return roots


def _runs_by_variant(root):
    runs = {}
    for metrics_file in sorted((root / "runs").glob("*/metrics.json")):
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        key = (metrics["model"], metrics["gamma"], metrics["seed"])
        assert key not in runs, f"Duplicate run for {key}"
        runs[key] = (metrics_file.parent, metrics)
    return runs


def test_quick_cli_saves_all_six_variants_and_finite_diagnostics(quick_runs):
    root = quick_runs[0]
    runs = _runs_by_variant(root)
    expected = {("tanh", None, 0), ("siren", None, 0)}
    expected.update(("gaussian", gamma, 0) for gamma in (0.0, 0.001, 0.01, 0.05))
    assert set(runs) == expected
    assert (root / "experiment.json").is_file()
    assert len(_read_csv(root / "summary.csv")) == 6
    assert len(_read_csv(root / "aggregate.csv")) == 6
    assert (root / "points_seed_0.npz").is_file()

    finite_metrics = (
        "relative_l2", "residual_rms", "total_loss", "pde_loss", "ic_loss",
        "ic_value_loss", "ic_velocity_loss", "bc_loss", "train_seconds",
        "grad_norm_initial", "grad_norm_final", "grad_norm_min", "grad_norm_max",
    )
    for run_dir, metrics in runs.values():
        assert metrics["status"] == "completed"
        assert metrics["epochs_completed"] == 2
        assert metrics["has_nonfinite"] is False
        assert metrics["divergence_detected"] is False
        assert metrics["device"] == "cpu"
        assert "fallback_reason" in metrics
        assert isinstance(metrics["loss_spike_count"], int)
        for name in finite_metrics:
            assert math.isfinite(metrics[name]), (run_dir.name, name, metrics[name])
            assert metrics[name] >= 0, (run_dir.name, name)
        assert metrics["ic_loss"] == pytest.approx(
            metrics["ic_value_loss"] + metrics["ic_velocity_loss"]
        )
        assert (run_dir / "config.json").is_file()
        assert (run_dir / "checkpoint.pt").is_file()
        history = _read_csv(run_dir / "history.csv")
        assert len(history) == 2
        assert [int(row["epoch"]) for row in history] == [1, 2]
        for row in history:
            assert all(math.isfinite(float(row[name])) for name in LOSS_COLUMNS)
        with np.load(run_dir / "predictions.npz") as prediction:
            for name in ("exact", "prediction", "abs_error"):
                assert prediction[name].shape == (9, 9)
                assert np.isfinite(prediction[name]).all()
            np.testing.assert_allclose(
                prediction["abs_error"],
                np.abs(prediction["prediction"] - prediction["exact"]),
            )
            measured = np.linalg.norm(prediction["prediction"] - prediction["exact"])
            measured /= np.linalg.norm(prediction["exact"])
            assert metrics["relative_l2"] == pytest.approx(measured, rel=2e-6)


def _state_dict(run_dir):
    return torch.load(
        run_dir / "checkpoint.pt", map_location="cpu", weights_only=True
    )["model_state_dict"]


def _assert_identical_training(first, second):
    first_dir, first_metrics = first
    second_dir, second_metrics = second
    first_state = _state_dict(first_dir)
    second_state = _state_dict(second_dir)
    assert first_state.keys() == second_state.keys()
    for name in first_state:
        torch.testing.assert_close(first_state[name], second_state[name], rtol=0, atol=0)
    with np.load(first_dir / "predictions.npz") as a, np.load(second_dir / "predictions.npz") as b:
        assert set(a.files) == set(b.files)
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name])
    a_history = _read_csv(first_dir / "history.csv")
    b_history = _read_csv(second_dir / "history.csv")
    assert len(a_history) == len(b_history)
    for a, b in zip(a_history, b_history):
        for name in ("epoch", *LOSS_COLUMNS):
            assert a[name] == b[name], name
    for name in ("relative_l2", "residual_rms", *LOSS_COLUMNS[:-1]):
        assert first_metrics[name] == second_metrics[name], name


def test_gamma_zero_is_exact_siren_training_control(quick_runs):
    runs = _runs_by_variant(quick_runs[0])
    _assert_identical_training(runs[("siren", None, 0)], runs[("gaussian", 0.0, 0)])


def test_cpu_runs_reproduce_weights_predictions_and_losses(quick_runs):
    first, repeat = map(_runs_by_variant, quick_runs)
    assert first.keys() == repeat.keys()
    for variant in first:
        _assert_identical_training(first[variant], repeat[variant])
    with np.load(quick_runs[0] / "points_seed_0.npz") as a, np.load(quick_runs[1] / "points_seed_0.npz") as b:
        for name in ("f", "ic", "bc"):
            np.testing.assert_array_equal(a[name], b[name])


def test_existing_output_is_refused_without_mutating_results(quick_runs):
    root = quick_runs[0]
    summary_before = (root / "summary.csv").read_bytes()
    result = subprocess.run(
        _quick_command(root), cwd=PROJECT_ROOT, capture_output=True,
        text=True, timeout=30, check=False,
    )
    assert result.returncode != 0
    assert (root / "summary.csv").read_bytes() == summary_before


def test_final_metrics_describe_saved_weights_on_entire_training_pool(quick_runs):
    from pinn_activation_comparison.config import ExperimentConfig
    from pinn_activation_comparison.losses import loss_terms
    from pinn_activation_comparison.models import PINN

    root = quick_runs[0]
    run_dir, metrics = _runs_by_variant(root)[("siren", None, 0)]
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    config = ExperimentConfig(**checkpoint["config"])
    model = PINN(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"])
    with np.load(root / "points_seed_0.npz") as saved:
        samples = {name: torch.from_numpy(saved[name]) for name in ("f", "ic", "bc")}
    losses = loss_terms(model, samples, config)
    for name, value in losses.items():
        assert metrics[name] == pytest.approx(value.item(), rel=2e-6, abs=1e-9), name


def test_training_objective_uses_analytic_field_only_on_boundaries(monkeypatch):
    from pinn_activation_comparison.config import ExperimentConfig
    from pinn_activation_comparison import losses
    from pinn_activation_comparison.models import PINN
    from pinn_activation_comparison.sampling import training_points

    config = ExperimentConfig(n_f=17, n_ic=11, n_bc=7, hidden_layers=2, width=8)
    samples = training_points(config, 123)
    analytic_field = losses.breather
    requested_points = []

    def boundary_only(points, omega):
        assert torch.all((points[:, 0] == config.x_min) | (points[:, 0] == config.x_max))
        requested_points.append(len(points))
        return analytic_field(points, omega)

    monkeypatch.setattr(losses, "breather", boundary_only)
    model = PINN(hidden_layers=2, width=8)
    parts = losses.loss_terms(model, samples, config)
    parts["total_loss"].backward()
    assert sum(requested_points) == 2 * config.n_bc
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_shared_training_points_are_reproducible_and_respect_boundaries():
    from pinn_activation_comparison.config import ExperimentConfig
    from pinn_activation_comparison.sampling import training_points

    config = ExperimentConfig(n_f=17, n_ic=11, n_bc=7)
    first = training_points(config, 123)
    repeat = training_points(config, 123)
    other_seed = training_points(config, 124)
    assert set(first) == {"f", "ic", "bc"}
    assert first["f"].shape == (17, 2)
    assert first["ic"].shape == (11, 2)
    assert first["bc"].shape == (14, 2)
    for name in first:
        torch.testing.assert_close(first[name], repeat[name], rtol=0, atol=0)
        assert torch.all((first[name][:, 0] >= -10) & (first[name][:, 0] <= 10))
        assert torch.all((first[name][:, 1] >= 0) & (first[name][:, 1] <= 20))
    assert not torch.equal(first["f"], other_seed["f"])
    assert torch.all(first["ic"][:, 1] == 0)
    assert int((first["bc"][:, 0] == -10).sum()) == 7
    assert int((first["bc"][:, 0] == 10).sum()) == 7


def test_full_preset_keeps_architecture_and_changes_compute_budget():
    from pinn_activation_comparison.config import parse_args

    quick = parse_args([])
    full = parse_args(["--mode", "full"])
    assert (quick.hidden_layers, quick.width) == (full.hidden_layers, full.width) == (4, 64)
    assert (full.epochs, full.n_f, full.n_ic, full.n_bc, full.batch_size) == (5000, 4096, 256, 256, 512)
    assert (full.eval_nx, full.eval_nt, full.eval_every) == (201, 201, 500)
    assert full.output_dir.endswith("results/full")


def test_json_mode_defaults_then_explicit_json_then_cli(tmp_path):
    from pinn_activation_comparison.config import parse_args

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"mode": "full", "epochs": 7, "width": 12, "gammas": [0, 0.02]}))
    config = parse_args(["--config", str(path), "--epochs", "9", "--seed", "2", "3"])
    assert config.mode == "full"
    assert config.epochs == 9
    assert config.width == 12
    assert config.n_f == 4096
    assert config.eval_nx == 201
    assert tuple(config.gammas) == (0, 0.02)
    assert tuple(config.seeds) == (2, 3)
    # A CLI mode override also changes unspecified preset values.
    quick = parse_args(["--config", str(path), "--mode", "quick"])
    assert quick.n_f == 256
    assert quick.epochs == 7
    assert quick.output_dir.endswith("results/quick")


@pytest.mark.parametrize("args", [
    ["--omega", "0"], ["--omega", "1"], ["--gamma", "-0.01"],
    ["--gamma", "nan"], ["--gamma", "inf"], ["--n-f", "0"],
    ["--epochs", "0"], ["--batch-size", "-1"], ["--eval-nx", "1"],
    ["--t-max", "0"], ["--x-min", "10", "--x-max", "-10"],
    ["--hidden-layers", "0"], ["--width", "0"],
])
def test_invalid_configuration_is_rejected(args):
    from pinn_activation_comparison.config import parse_args

    with pytest.raises((SystemExit, ValueError)):
        parse_args(args)


def test_unknown_json_keys_are_rejected(tmp_path):
    from pinn_activation_comparison.config import parse_args

    path = tmp_path / "typo.json"
    path.write_text('{"epohcs": 10}')
    with pytest.raises((SystemExit, ValueError)):
        parse_args(["--config", str(path)])
