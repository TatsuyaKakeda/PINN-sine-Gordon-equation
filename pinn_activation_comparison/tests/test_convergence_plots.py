"""Check post-update convergence data and optimizer-aware snapshot plots."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pinn_activation_comparison import plot_results


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _snapshot(offset: float = 0.0) -> dict[str, np.ndarray]:
    t = np.linspace(0.0, 10.0, 31)
    x = np.linspace(-5.0, 5.0, 21)
    return {
        "center_t": t, "center_exact": np.sin(t), "center_prediction": np.sin(t) + offset,
        "ic_x": x, "ic_velocity_exact": np.exp(-x**2),
        "ic_velocity_prediction": np.exp(-x**2) + offset,
    }


@pytest.fixture
def hybrid_artifacts(tmp_path: Path) -> Path:
    for model in ("tanh", "siren"):
        run_dir = tmp_path / "runs" / f"{model}_seed7"
        run_dir.mkdir(parents=True)
        metrics = {
            "model": model, "gamma": None, "seed": 7, "status": "completed",
            "optimizer": "adam_lbfgs", "stage": "lbfgs", "relative_l2": 0.02,
            "total_loss": 0.01, "train_seconds": 8.0,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
        rows = []
        for epoch, stage, stage_step, seconds, value in (
            (0, "adam", 0, 0.0, 1.0), (20, "adam", 20, 4.0, 0.4),
            (21, "lbfgs", 1, 6.0, 0.2), (22, "lbfgs", 2, 8.0, 0.1),
        ):
            rows.append({
                "epoch": epoch, "stage": stage, "stage_step": stage_step,
                "train_seconds": seconds, "total_loss": value,
                "pde_loss": value / 2, "ic_loss": value / 3,
                "ic_value_loss": value / 8, "ic_velocity_loss": value / 6,
                "bc_loss": value / 4, "relative_l2": value / 5,
                "residual_rms": value / 7, "ic_velocity_relative_l2": value / 9,
                "center_amplitude_ratio": 1 - value / 2,
            })
        _write_csv(run_dir / "evaluations.csv", rows)
        history = []
        for row in rows[1:]:
            history.append(dict(row, total_loss=99.0, grad_norm=2.0,
                                grad_norm_first=1.0, grad_norm_last=0.5))
        _write_csv(run_dir / "history.csv", history)
        snapshot_dir = run_dir / "snapshots"
        snapshot_dir.mkdir()
        for stage, step in (("adam", 0), ("adam", 20), ("lbfgs", 2)):
            np.savez_compressed(snapshot_dir / f"{stage}_{step:06d}.npz", **_snapshot(0.1))
    return tmp_path


def _capture_figures(monkeypatch: pytest.MonkeyPatch) -> dict:
    figures = {}

    def capture(fig: object, path: Path) -> Path:
        figures[path.name] = fig
        return path

    monkeypatch.setattr(plot_results, "_save", capture)
    return figures


def test_loader_preserves_optimizer_stages_and_post_update_evaluations(hybrid_artifacts: Path) -> None:
    runs = plot_results.load_runs(hybrid_artifacts)
    run = runs[0]
    assert run.history["stage"].tolist() == ["adam", "lbfgs", "lbfgs"]
    assert run.evaluations["total_loss"].tolist() == [1.0, 0.4, 0.2, 0.1]
    assert run.optimizer == "adam_lbfgs"
    assert set(run.snapshots) == {("adam", 0), ("adam", 20), ("lbfgs", 2)}
    np.testing.assert_array_equal(run.snapshots[("adam", 20)]["center_t"], _snapshot()["center_t"])


def test_convergence_uses_full_pool_evaluations_and_separate_stage_steps(
    hybrid_artifacts: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    figures = _capture_figures(monkeypatch)
    try:
        plot_results.plot_all(hybrid_artifacts)
        adam = figures["learning_curves.png"]
        for line in adam.axes[0].lines:
            np.testing.assert_array_equal(line.get_xdata(), [0, 20])
            np.testing.assert_allclose(line.get_ydata(), [1.0, 0.4])
        assert adam.axes[-1].get_xlabel() == "Adam epoch"
        assert "full training pool" in adam._suptitle.get_text()
        assert "post-update" in adam._suptitle.get_text()
        lbfgs = figures["lbfgs_learning_curves.png"]
        for line in lbfgs.axes[0].lines:
            np.testing.assert_array_equal(line.get_xdata(), [1, 2])
            np.testing.assert_allclose(line.get_ydata(), [0.2, 0.1])
        assert lbfgs.axes[-1].get_xlabel() == "L-BFGS outer step"
        gradients = figures["gradient_norms.png"]
        for line in gradients.axes[0].lines:
            np.testing.assert_array_equal(line.get_xdata(), [20])
        assert "first closure" in figures["lbfgs_gradient_norms.png"]._suptitle.get_text()
    finally:
        plot_results.plt.close("all")


def test_initial_velocity_loss_and_heldout_error_are_separate_panels(
    hybrid_artifacts: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    figures = _capture_figures(monkeypatch)
    try:
        plot_results.plot_all(hybrid_artifacts)
        fig = figures["ic_velocity_convergence.png"]
        labels = [axis.get_ylabel() for axis in fig.axes]
        assert "IC velocity loss (training pool)" in labels
        assert "IC velocity relative L2 (held-out)" in labels
        velocity_axis = fig.axes[labels.index("IC velocity loss (training pool)")]
        heldout_axis = fig.axes[labels.index("IC velocity relative L2 (held-out)")]
        np.testing.assert_allclose(velocity_axis.lines[0].get_ydata(), np.array([1, 0.4]) / 6)
        np.testing.assert_allclose(heldout_axis.lines[0].get_ydata(), np.array([1, 0.4]) / 9)
    finally:
        plot_results.plt.close("all")


def test_time_convergence_labels_stages_without_connecting_them(
    hybrid_artifacts: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    figures = _capture_figures(monkeypatch)
    try:
        plot_results.plot_all(hybrid_artifacts)
        fig = figures["training_time_convergence.png"]
        assert fig.axes[-1].get_xlabel() == "Cumulative training time (s)"
        lines = fig.axes[0].lines
        assert len(lines) == 4
        assert all(len(line.get_xdata()) == 2 for line in lines)
        assert any("Adam" in line.get_label() for line in lines)
        assert any("L-BFGS" in line.get_label() for line in lines)
    finally:
        plot_results.plt.close("all")


def test_periodic_snapshot_overlays_include_all_variants_at_same_stage_and_step(
    hybrid_artifacts: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    figures = _capture_figures(monkeypatch)
    try:
        paths = plot_results.plot_all(hybrid_artifacts)
        name = "center_trace_seed_7_lbfgs_000002.png"
        assert name in {path.name for path in paths}
        axis = figures[name].axes[0]
        assert axis.get_legend_handles_labels()[1] == ["Exact", "tanh", "SIREN"]
        assert "L-BFGS outer step 2" in axis.get_title()
        assert "Adam → L-BFGS" in axis.get_title()
    finally:
        plot_results.plt.close("all")


def test_save_center_snapshot_creates_training_time_png(tmp_path: Path) -> None:
    path = plot_results.save_center_snapshot(tmp_path, _snapshot(0.05), "adam", 500, "tanh | Seed 7")
    assert path == tmp_path / "snapshots" / "center_adam_000500.png"
    assert path.is_file() and path.stat().st_size > 5000
    assert not plot_results.plt.get_fignums()


def test_summary_does_not_pool_adam_and_hybrid_protocols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = []
    for optimizer, loss in (("adam", 1.0), ("adam_lbfgs", 0.1)):
        metrics = dict(model="tanh", gamma=None, seed=1, status="completed",
                       optimizer=optimizer, relative_l2=loss, total_loss=loss)
        runs.append(plot_results.RunArtifacts(optimizer, metrics, {}, None))
    figures = _capture_figures(monkeypatch)
    try:
        plot_results._summary_figure(runs, tmp_path)
        fig = figures["gamma_summary.png"]
        assert len(fig.axes[0].containers) == 2
        labels = [tick.get_text() for tick in fig.axes[0].get_xticklabels()]
        assert any("Adam → L-BFGS" in label for label in labels)
        assert any(label.endswith("Adam") for label in labels)
        counts = [text.get_text() for text in fig.axes[4].texts]
        assert counts == ["N=1", "N=1"]
    finally:
        plot_results.plt.close("all")
