"""Check saved-result plotting with small, deterministic synthetic artifacts."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pinn_activation_comparison import plot_results


def _write_run(root: Path, name: str, model: str, gamma: float | None,
               seed: int = 17, prediction: bool = True, offset: float = 0.1) -> Path:
    """Write the same artifact schema as the training runner."""
    run_dir = root / "runs" / name
    run_dir.mkdir(parents=True)
    metrics = {
        "model": model, "gamma": gamma, "seed": seed,
        "status": "completed" if prediction else "nonfinite",
        "relative_l2": 0.1 + offset, "residual_rms": 0.2,
        "total_loss": 0.5, "pde_loss": 0.3, "ic_loss": 0.1,
        "bc_loss": 0.1, "train_seconds": 1.2,
        "grad_norm_min": 0.1, "grad_norm_max": 8.0,
        "grad_norm_final": 0.5, "loss_spike_count": 1,
        "epochs_completed": 3, "has_nonfinite": not prediction,
        "divergence_detected": False,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    (run_dir / "config.json").write_text(json.dumps({"model": model, "gamma": gamma}))
    fields = ["epoch", "total_loss", "pde_loss", "ic_loss", "bc_loss", "grad_norm",
              "grad_norm_first", "grad_norm_last", "train_seconds", "relative_l2",
              "residual_rms"]
    with (run_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, 4):
            writer.writerow({
                "epoch": epoch, "total_loss": 1 / epoch, "pde_loss": 0.5 / epoch,
                "ic_loss": 0.2 / epoch, "bc_loss": 0.3 / epoch,
                "grad_norm": 2 / epoch, "grad_norm_first": 0.5 / epoch,
                "grad_norm_last": 0.8 / epoch, "train_seconds": epoch * 0.4,
                "relative_l2": 0.1 / epoch if epoch != 2 else "",
                "residual_rms": 0.2 / epoch if epoch != 2 else "",
            })
    if prediction:
        x, t = np.linspace(-2, 2, 13), np.linspace(0, 1, 9)
        exact = np.cos(x)[None, :] * np.cos(t)[:, None]
        predicted = exact + offset
        np.savez_compressed(
            run_dir / "predictions.npz", x=x, t=t, exact=exact,
            prediction=predicted, abs_error=np.abs(predicted - exact),
            center_t=t, center_exact=np.cos(t), center_prediction=np.cos(t) + offset,
            profile_times=t[[0, 4, 8]], profile_x=x,
            profile_exact=exact[[0, 4, 8]], profile_prediction=predicted[[0, 4, 8]],
        )
    return run_dir


@pytest.fixture
def result_artifacts(tmp_path: Path) -> Path:
    """Provide all six activation variants, including one failed run."""
    (tmp_path / "experiment.json").write_text(json.dumps({"preset": "quick"}))
    _write_run(tmp_path, "tanh_seed17", "tanh", None)
    _write_run(tmp_path, "siren_seed17", "siren", None, offset=0.2)
    for gamma in (0.0, 0.001, 0.01, 0.05):
        _write_run(tmp_path, f"gaussian_{gamma}_seed17", "gaussian", gamma,
                   prediction=gamma != 0.05, offset=gamma + 0.1)
    return tmp_path


@pytest.mark.parametrize("kind,gamma", [("tanh", 0.0), ("siren", 0.0),
                                        ("gaussian", 0.0), ("gaussian", 0.05)])
def test_activation_derivatives_include_frequency_chain_rule(kind: str, gamma: float) -> None:
    """Compare analytical derivatives to independent central differences."""
    z = np.linspace(-1.2, 1.3, 31)
    omega = 2.3
    value, first, second = plot_results.activation_derivatives(z, kind, omega, gamma)
    step = 1e-4
    plus = plot_results.activation_derivatives(z + step, kind, omega, gamma)[0]
    minus = plot_results.activation_derivatives(z - step, kind, omega, gamma)[0]
    np.testing.assert_allclose(first, (plus - minus) / (2 * step), atol=2e-7)
    np.testing.assert_allclose(second, (plus - 2 * value + minus) / step**2, atol=5e-7)
    assert value.shape == first.shape == second.shape == z.shape


def test_heatmaps_share_limits_across_all_runs(result_artifacts: Path) -> None:
    """One prediction outlier must expand every run's common field scale."""
    outlier_dir = _write_run(result_artifacts, "outlier", "siren", None, seed=99, offset=12.0)
    with np.load(outlier_dir / "predictions.npz") as archive:
        expected_maximum = float(np.max(archive["prediction"]))
    runs = plot_results.load_runs(result_artifacts)
    minimum, maximum, error_maximum = plot_results.heatmap_limits(runs)
    assert minimum < 0
    assert maximum == pytest.approx(expected_maximum)
    assert error_maximum == pytest.approx(12.0)


def test_sample_statistics_do_not_imply_uncertainty_for_one_seed() -> None:
    """Use ddof=1 for repeated seeds and zero SD when only one is available."""
    assert plot_results.sample_statistics([2.0]) == (2.0, 0.0, 1)
    mean, std, count = plot_results.sample_statistics([1.0, 3.0, float("nan")])
    assert mean == 2.0
    assert std == pytest.approx(np.sqrt(2))
    assert count == 2


def test_summary_excludes_failed_numeric_metrics_but_counts_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finite partial result from a diverged run must not bias success means."""
    fields = ("relative_l2", "total_loss", "train_seconds", "loss_spike_count",
              "grad_norm_min", "grad_norm_final", "grad_norm_max")
    runs = []
    for seed, status, value in [(1, "completed", 1.0), (2, "completed", 3.0),
                                (3, "diverged", 100.0)]:
        metrics = dict.fromkeys(fields, value)
        metrics.update(model="siren", gamma=None, seed=seed, status=status,
                       has_nonfinite=False, divergence_detected=status == "diverged")
        runs.append(plot_results.RunArtifacts(str(seed), metrics, {}, None))
    captured = []

    def capture_figure(fig: object, path: Path) -> Path:
        """Retain the genuine matplotlib figure to inspect plotted statistics."""
        captured.append(fig)
        return path

    monkeypatch.setattr(plot_results, "_save", capture_figure)
    plot_results._summary_figure(runs, tmp_path)
    fig = captured[0]
    try:
        for index in (0, 1, 2, 3, 5):
            for errorbar in fig.axes[index].containers:
                np.testing.assert_allclose(np.asarray(errorbar.lines[0].get_ydata(), dtype=float), [2.0])
                segment = errorbar.lines[2][0].get_segments()[0]
                np.testing.assert_allclose(segment[:, 1], [2 - np.sqrt(2), 2 + np.sqrt(2)])
        assert [patch.get_height() for patch in fig.axes[4].patches] == [1, 0, 1]
        assert any(text.get_text() == "N=3" for text in fig.axes[4].texts)
        assert "completed" in fig._suptitle.get_text()
    finally:
        plot_results.plt.close(fig)


def test_plot_all_saves_complete_figure_suite_and_handles_failed_run(result_artifacts: Path) -> None:
    """Generate plots from files without requiring a trained model or torch."""
    paths = plot_results.plot_all(result_artifacts)
    expected = {
        "activation_functions.png", "learning_curves.png", "gradient_norms.png",
        "error_convergence.png", "gamma_summary.png", "center_trace_seed_17.png",
        "spatial_profiles_seed_17.png",
    }
    assert expected <= {path.name for path in paths}
    assert len([path for path in paths if path.name.startswith("heatmap_")]) == 6
    for path in paths:
        assert path.parent == result_artifacts / "figures"
        assert path.is_file() and path.stat().st_size > 5000
    assert not plot_results.plt.get_fignums()


def test_plot_all_rejects_missing_run_artifacts(tmp_path: Path) -> None:
    """An empty results directory must fail with a useful explanation."""
    with pytest.raises(ValueError, match="No run artifacts"):
        plot_results.plot_all(tmp_path)


def test_plot_all_handles_failure_before_first_epoch(tmp_path: Path) -> None:
    """An early failure must still produce labeled diagnostics and summary."""
    run_dir = _write_run(tmp_path, "early_failure", "gaussian", 0.05, prediction=False)
    (run_dir / "history.csv").unlink()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    for key in ("total_loss", "relative_l2", "grad_norm_min", "grad_norm_max", "grad_norm_final"):
        metrics[key] = None
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    paths = plot_results.plot_all(tmp_path)
    assert any(path.name == "heatmap_early_failure.png" for path in paths)
    assert all(path.exists() for path in paths)


def test_activation_figure_includes_custom_trained_gamma(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A custom run's damping must appear alongside the canonical references."""
    _write_run(tmp_path, "custom_gamma", "gaussian", 0.02)
    activation_labels = []
    captions = []
    save = plot_results._save

    def inspect_saved_figure(fig: object, path: Path) -> Path:
        """Read actual plotted legends before the normal save closes the figure."""
        if path.name == "activation_functions.png":
            activation_labels.extend(fig.axes[0].get_legend_handles_labels()[1])
            captions.append(fig._suptitle.get_text())
        return save(fig, path)

    monkeypatch.setattr(plot_results, "_save", inspect_saved_figure)
    plot_results.plot_all(tmp_path)
    assert "Gaussian γ=0.02" in activation_labels
    for gamma in (0.0, 0.001, 0.01, 0.05):
        assert f"Gaussian γ={gamma:g}" in activation_labels
    assert "custom" in captions[0]


def test_summary_scale_follows_observed_means_without_clipping_sd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large relative SD keeps its true negative endpoint on a readable scale."""
    fields = ("relative_l2", "total_loss", "train_seconds", "loss_spike_count",
              "grad_norm_min", "grad_norm_final", "grad_norm_max")
    runs = []
    for model, values in [("tanh", (0.01, 0.15)), ("siren", (20.0, 200.0))]:
        for seed, value in enumerate(values):
            metrics = dict.fromkeys(fields, value)
            metrics.update(model=model, gamma=None, seed=seed, status="completed")
            runs.append(plot_results.RunArtifacts(f"{model}_{seed}", metrics, {}, None))
    captured = []

    def capture_figure(fig: object, path: Path) -> Path:
        """Inspect the real figure before it is closed by the saving helper."""
        captured.append(fig)
        return path

    monkeypatch.setattr(plot_results, "_save", capture_figure)
    plot_results._summary_figure(runs, tmp_path)
    fig = captured[0]
    try:
        expected_sd = np.std([0.01, 0.15], ddof=1)
        for index in (0, 1, 3):
            axis = fig.axes[index]
            assert axis.get_yscale() == "symlog"
            assert axis.yaxis.get_transform().linthresh == pytest.approx(0.008)
            lower, upper = axis.get_ylim()
            visible_ticks = [tick for tick in axis.get_yticks() if lower <= tick <= upper]
            assert len(visible_ticks) <= 9
            assert all(tick == 0 or abs(tick) >= 0.008 for tick in visible_ticks)
            segment = axis.containers[0].lines[2][0].get_segments()[0]
            np.testing.assert_allclose(segment[:, 1], [0.08 - expected_sd, 0.08 + expected_sd])
            assert lower < 0.08 - expected_sd < 0
        assert "SD bars may extend below zero" in fig._suptitle.get_text()
    finally:
        plot_results.plt.close(fig)
