"""Analyze the completed 5,000-Adam-epoch experiment without training.

From the project root:
    .venv/bin/python pinn_activation_comparison/results/full_5000/analyze_convergence.py --expected-runs 18

Only saved evaluation data are read; no model, optimizer, or training module is
imported. The experiment must have a terminal status and every expected run's
metrics. Failed runs remain in diagnostic_summary.csv; convergence statistics
use completed runs and report their denominators explicitly. Sample SD uses
ddof=1 and is blank for fewer than two finite observations.

Temporal windows are [0,5), [5,10), [10,15), and [15,20]. The first and last
windows define early/late center RMS ratios. These are discrete-grid norms,
without spatial/time quadrature weights. RMS ratios do not establish phase
accuracy. Mean +/- SD bands describe seed variation, not confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


EVALUATION_FIELDS = (
    "relative_l2", "residual_rms", "total_loss", "pde_loss", "ic_loss",
    "ic_value_loss", "ic_velocity_loss", "bc_loss", "ic_velocity_relative_l2",
    "center_amplitude_ratio", "train_seconds",
)
WINDOWS = ((0, 5), (5, 10), (10, 15), (15, 20))


def _number(value):
    """Return a float or NaN for unavailable serialized diagnostics."""
    try:
        return float(value) if not isinstance(value, bool) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _csv_rows(path):
    """Read a CSV without pandas or training-library dependencies."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _variant_order(key):
    """Keep tanh and SIREN before increasing Gaussian damping."""
    return {"tanh": 0, "siren": 1, "gaussian": 2}.get(key[0], 3), key[1] or 0.0


def _load_runs(root, expected_runs, expected_epochs):
    """Reject an unfinished/mismatched suite before creating any output."""
    metadata = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if metadata.get("status") not in {"completed", "completed_with_failures"}:
        raise ValueError("Experiment is not completed; wait until every run has finished.")
    config = metadata["config"]
    if config.get("optimizer") != "adam" or config.get("epochs") != expected_epochs:
        raise ValueError(f"Expected Adam-only experiment requesting {expected_epochs} epochs.")
    if not math.isclose(float(config.get("t_max", 20.0)), 20.0):
        raise ValueError("Temporal diagnostics require the saved [0,20] time domain.")
    conditions = {(model, gamma, int(seed)) for model in config["models"]
                  for gamma in (config["gammas"] if model == "gaussian" else [None])
                  for seed in config["seeds"]}
    paths = sorted((root / "runs").glob("*/metrics.json"))
    if len(conditions) != expected_runs or len(paths) != expected_runs:
        raise ValueError(f"Expected all {expected_runs} run metrics; found {len(paths)} "
                         f"files and {len(conditions)} manifest conditions.")
    interval = int(config["eval_every"])
    expected_grid = sorted({0, expected_epochs, *range(interval, expected_epochs + 1, interval)})
    runs, seen = [], set()
    for path in paths:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        key = metrics["model"], metrics.get("gamma"), int(metrics["seed"])
        if key in seen or key not in conditions:
            raise ValueError(f"Duplicate or unexpected run condition in {path}.")
        seen.add(key)
        completed = metrics.get("status") == "completed"
        if metrics.get("status") not in {"completed", "nonfinite", "diverged"}:
            raise ValueError(f"Unfinished or unknown run status in {path}.")
        if metrics.get("optimizer") != "adam" or metrics.get("epochs_requested") != expected_epochs:
            raise ValueError(f"Unexpected optimizer or requested epochs in {path}.")
        if completed and metrics.get("epochs_completed") != expected_epochs:
            raise ValueError(f"Completed run did not finish {expected_epochs} epochs: {path}.")
        evaluations = {}
        if completed:
            for row in _csv_rows(path.parent / "evaluations.csv"):
                epoch = int(row["epoch"])
                if row.get("stage") != "adam" or int(row["stage_step"]) != epoch or epoch in evaluations:
                    raise ValueError(f"Invalid/duplicate evaluation stage or epoch in {path.parent}.")
                values = {field: _number(row.get(field)) for field in EVALUATION_FIELDS}
                if not all(math.isfinite(value) for value in values.values()):
                    raise ValueError(f"Completed-run evaluation must be finite: {path.parent}, epoch {epoch}.")
                evaluations[epoch] = values
            if sorted(evaluations) != expected_grid:
                raise ValueError(f"Incomplete or unmatched evaluation grid in {path.parent}.")
        runs.append({"path": path.parent, "metrics": metrics, "evaluations": evaluations})
    if seen != conditions:
        raise ValueError("Saved runs do not cover all configured seeds and activation variants.")
    runs.sort(key=lambda run: (_variant_order((run["metrics"]["model"], run["metrics"].get("gamma"))),
                               run["metrics"]["seed"]))
    return runs, expected_grid


def _convergence_rows(runs, epochs):
    """Aggregate matching epochs across successful seeds, retaining failures."""
    groups = {}
    for run in runs:
        metrics = run["metrics"]
        groups.setdefault((metrics["model"], metrics.get("gamma")), []).append(run)
    rows = []
    for (model, gamma), group in sorted(groups.items(), key=lambda item: _variant_order(item[0])):
        completed = [run for run in group if run["metrics"]["status"] == "completed"]
        for epoch in epochs:
            row = {"model": model, "gamma": gamma, "optimizer": "adam", "epoch": epoch,
                   "seed_count": len(completed), "n_total": len(group), "n_completed": len(completed),
                   "n_failed": len(group) - len(completed),
                   "seeds_in_mean": ";".join(str(run["metrics"]["seed"]) for run in completed)}
            for field in EVALUATION_FIELDS:
                values = np.asarray([run["evaluations"][epoch][field] for run in completed], dtype=float)
                values = values[np.isfinite(values)]
                row[field + "_count"] = int(values.size)
                row[field + "_mean"] = float(values.mean()) if values.size else None
                row[field + "_std"] = float(values.std(ddof=1)) if values.size > 1 else None
            rows.append(row)
    return rows


def _relative_l2(predicted, exact):
    """Evaluate a discrete relative error without changing either input."""
    denominator = np.linalg.norm(exact)
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("Exact reference has zero or nonfinite norm in a diagnostic window.")
    return float(np.linalg.norm(predicted - exact) / denominator)


def _window_mask(times, start, stop):
    """Assign each timestamp to one window, including the final endpoint."""
    mask = (times >= start) & ((times <= stop) if stop == 20 else (times < stop))
    if not mask.any():
        raise ValueError(f"No evaluation samples in time window {start} to {stop}.")
    return mask


def _prediction_diagnostics(path):
    """Measure global-center and per-window errors from saved predictions."""
    with np.load(path, allow_pickle=False) as archive:
        keys = ("t", "x", "exact", "prediction", "center_t", "center_exact", "center_prediction")
        arrays = {key: np.asarray(archive[key], dtype=float) for key in keys}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"Nonfinite saved prediction data in {path}.")
    t, center_t = arrays["t"], arrays["center_t"]
    for coordinates in (t, center_t):
        if coordinates.ndim != 1 or np.any(np.diff(coordinates) <= 0):
            raise ValueError(f"Invalid temporal coordinates in {path}.")
        if not np.isclose(coordinates[0], 0.0) or not np.isclose(coordinates[-1], 20.0):
            raise ValueError(f"Temporal coordinates must cover [0,20] in {path}.")
    exact, predicted = arrays["exact"], arrays["prediction"]
    center_exact, center = arrays["center_exact"], arrays["center_prediction"]
    if exact.shape != predicted.shape or exact.shape != (len(t), len(arrays["x"])):
        raise ValueError(f"Invalid (t,x) prediction array shapes in {path}.")
    if center.shape != center_exact.shape or center.shape != center_t.shape:
        raise ValueError(f"Invalid center-trace array shapes in {path}.")
    row = {"center_relative_l2": _relative_l2(center, center_exact)}
    for start, stop in WINDOWS:
        mask, center_mask = _window_mask(t, start, stop), _window_mask(center_t, start, stop)
        suffix = f"t{start}_{stop}"
        row["time_samples_" + suffix] = int(mask.sum())
        row["relative_l2_" + suffix] = _relative_l2(predicted[mask], exact[mask])
        row["center_relative_l2_" + suffix] = _relative_l2(center[center_mask], center_exact[center_mask])
        denominator = np.linalg.norm(center_exact[center_mask])
        row["center_rms_ratio_" + suffix] = float(np.linalg.norm(center[center_mask]) / denominator)
    row["center_early_rms_ratio"] = row["center_rms_ratio_t0_5"]
    row["center_late_rms_ratio"] = row["center_rms_ratio_t15_20"]
    return row


def _diagnostic_rows(runs):
    """Keep each successful or failed outcome, with finite gradient summaries."""
    rows = []
    prediction_fields = ["center_relative_l2", "center_early_rms_ratio", "center_late_rms_ratio"]
    prediction_fields += [prefix + f"t{start}_{stop}" for start, stop in WINDOWS
                          for prefix in ("time_samples_", "relative_l2_", "center_relative_l2_", "center_rms_ratio_")]
    for run in runs:
        metrics, path = run["metrics"], run["path"]
        completed = metrics["status"] == "completed"
        row = {"run_id": path.name, "model": metrics["model"], "gamma": metrics.get("gamma"),
               "seed": metrics["seed"], "status": metrics["status"], "optimizer": metrics["optimizer"],
               "epochs_requested": metrics["epochs_requested"], "epochs_completed": metrics["epochs_completed"],
               "reached_requested_epochs": metrics["epochs_completed"] == metrics["epochs_requested"],
               "included_in_convergence_means": completed}
        row.update({field: metrics.get(field) for field in (*EVALUATION_FIELDS, "grad_norm_initial",
                    "grad_norm_final", "grad_norm_min", "grad_norm_max", "loss_spike_count")})
        row.update(dict.fromkeys(prediction_fields))
        try:
            row.update(_prediction_diagnostics(path / "predictions.npz"))
            row["prediction_diagnostic_status"] = "available"
        except (OSError, KeyError, ValueError) as error:
            if completed:
                raise ValueError(f"Invalid completed-run predictions in {path}: {error}") from error
            row["prediction_diagnostic_status"] = f"unavailable: {error}"
        history_path = path / "history.csv"
        history = _csv_rows(history_path) if history_path.exists() else []
        history = [item for item in history if item.get("stage") == "adam"]
        row["gradient_history_rows"] = len(history)
        for field in ("grad_norm", "grad_norm_first", "grad_norm_last"):
            values = np.asarray([_number(item.get(field)) for item in history])
            finite = values[np.isfinite(values)]
            row[field + "_finite_count"] = int(finite.size)
            row[field + "_nonfinite_count"] = int(values.size - finite.size)
            row[field + "_mean"] = float(finite.mean()) if finite.size else None
            row[field + "_std"] = float(finite.std(ddof=1)) if finite.size > 1 else None
            for quantile, name in ((0.05, "p05"), (0.5, "median"), (0.95, "p95")):
                row[field + "_" + name] = float(np.quantile(finite, quantile)) if finite.size else None
            row[field + "_near_zero_fraction"] = float(np.mean(finite < 1e-12)) if finite.size else None
        rows.append(row)
    return rows


def _write_csv(path, rows):
    """Write portable CSVs, leaving absent/nonfinite measurements blank."""
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None or isinstance(value, float) and not math.isfinite(value)
                             else value for key, value in row.items()})


def _plot_convergence(rows, path):
    """Plot true mean +/- sample SD bands, retaining signed SD endpoints."""
    cache = Path(tempfile.gettempdir()) / "pinn-activation-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {}
    for row in rows:
        groups.setdefault((row["model"], row["gamma"]), []).append(row)
    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#B07AA1", "#E15759", "#76B7B2"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharex=True, layout="constrained")
    try:
        for index, ((model, gamma), group) in enumerate(groups.items()):
            label = f"Gaussian gamma={gamma:g}" if model == "gaussian" else "SIREN" if model == "siren" else model
            label += f" (n={group[0]['n_completed']}/{group[0]['n_total']})"
            epochs = np.asarray([row["epoch"] for row in group])
            for axis, field in zip(axes, ("relative_l2", "ic_velocity_relative_l2")):
                mean = np.asarray([_number(row[field + "_mean"]) for row in group])
                sd = np.asarray([_number(row[field + "_std"]) for row in group])
                color = colors[index % len(colors)]
                axis.plot(epochs, mean, label=label, color=color, marker="o", markersize=3,
                          linestyle="--" if model == "siren" else "-", linewidth=1.7)
                axis.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.13)
        axes[0].set(title="Independent field error", ylabel="Relative L2 error")
        positive = [_number(row["ic_velocity_relative_l2_mean"]) for row in rows]
        positive = [value for value in positive if math.isfinite(value) and value > 0]
        threshold = min(positive) * 0.1 if positive else 1e-3
        axes[1].set_yscale("symlog", linthresh=max(threshold, 1e-12))
        axes[1].yaxis.get_major_locator().set_params(numticks=7)
        axes[1].set(title="Independent initial-velocity error", ylabel="IC velocity relative L2 (symlog)")
        for axis in axes:
            axis.set_xlabel("Adam epoch")
            axis.axhline(0, color="#888888", linewidth=0.6)
            axis.grid(alpha=0.2, which="both")
            axis.legend(fontsize=8, loc="best")
        fig.suptitle("Adam convergence: mean +/- sample SD (ddof=1) across completed seeds\n"
                     "n = completed / attempted; failed runs excluded from means and retained in diagnostic_summary.csv\n"
                     "SD bands describe seed variation, not confidence intervals; signed endpoints are not clipped.", fontsize=11)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)


def analyze(results, *, expected_runs=18, expected_epochs=5000):
    """Validate the entire suite, then write two CSVs and one convergence PNG."""
    root = Path(results)
    if expected_runs < 1 or expected_epochs < 1:
        raise ValueError("Expected run and epoch counts must be positive.")
    runs, epochs = _load_runs(root, expected_runs, expected_epochs)
    convergence = _convergence_rows(runs, epochs)
    diagnostics = _diagnostic_rows(runs)
    paths = [root / "convergence_summary.csv", root / "diagnostic_summary.csv",
             root / "figures" / "convergence_mean_std.png"]
    _write_csv(paths[0], convergence)
    _write_csv(paths[1], diagnostics)
    _plot_convergence(convergence, paths[2])
    return paths


def main(argv=None):
    """Expose strict expected-count guards without invoking any training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--expected-runs", type=int, default=18)
    parser.add_argument("--expected-epochs", type=int, default=5000)
    args = parser.parse_args(argv)
    try:
        paths = analyze(args.results, expected_runs=args.expected_runs, expected_epochs=args.expected_epochs)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(2, f"Analysis refused: {error}\n")
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
