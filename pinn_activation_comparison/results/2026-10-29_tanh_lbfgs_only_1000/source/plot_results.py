"""Rebuild all comparison figures from saved artifacts without loading a model.

Run ``python -m pinn_activation_comparison.plot_results --results OUTPUT_DIR``.
The plotted exact solution is evaluation data only; this module never trains.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any

_CONFIG_DIR = Path(tempfile.gettempdir()) / "pinn-activation-matplotlib"
_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np


@dataclass
class RunArtifacts:
    """Numeric artifacts for one activation, gamma, and random seed."""

    run_id: str
    metrics: dict[str, Any]
    history: dict[str, np.ndarray]
    predictions: dict[str, np.ndarray] | None
    evaluations: dict[str, np.ndarray] = field(default_factory=dict)
    snapshots: dict[tuple[str, int], dict[str, np.ndarray]] = field(default_factory=dict)

    @property
    def optimizer(self) -> str:
        """Identify the optimizer protocol; historical runs used Adam only."""
        return str(self.metrics.get("optimizer", "adam"))

    @property
    def model(self) -> str:
        """Return the activation family identifier."""
        return str(self.metrics.get("model", "unknown"))

    @property
    def gamma(self) -> float | None:
        """Return Gaussian damping, or None for baseline families."""
        value = self.metrics.get("gamma")
        return None if value is None else float(value)

    @property
    def seed(self) -> int:
        """Return the experiment seed."""
        return int(self.metrics.get("seed", 0))

    @property
    def status(self) -> str:
        """Return the recorded completion or failure status."""
        return str(self.metrics.get("status", "unknown"))

    @property
    def label(self) -> str:
        """Build a readable activation label including failed-run status."""
        label = _variant_label((self.model, self.gamma))
        return label if self.status in {"completed", "ok", "success"} else f"{label} [{self.status}]"


def _number(value: Any) -> float:
    """Convert blank, absent, or nonnumeric metric entries to NaN."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def load_runs(output_dir: Path) -> list[RunArtifacts]:
    """Read saved run files, retaining failures that have no predictions."""
    runs: list[RunArtifacts] = []
    for metrics_path in sorted((Path(output_dir) / "runs").glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        history = _read_series(metrics_path.parent / "history.csv")
        evaluations = _read_series(metrics_path.parent / "evaluations.csv")
        predictions = None
        prediction_path = metrics_path.parent / "predictions.npz"
        if prediction_path.exists():
            with np.load(prediction_path, allow_pickle=False) as archive:
                predictions = {key: np.asarray(archive[key]) for key in archive.files}
        snapshots = {}
        for snapshot_path in sorted((metrics_path.parent / "snapshots").glob("*.npz")):
            stage, _, step = snapshot_path.stem.rpartition("_")
            if stage not in {"adam", "lbfgs"} or not step.isdigit():
                continue
            with np.load(snapshot_path, allow_pickle=False) as archive:
                snapshots[(stage, int(step))] = {key: np.asarray(archive[key]) for key in archive.files}
        runs.append(RunArtifacts(metrics_path.parent.name, metrics, history, predictions,
                                 evaluations, snapshots))
    return sorted(runs, key=lambda run: (run.seed, _variant_order((run.model, run.gamma))))


def _read_series(path: Path) -> dict[str, np.ndarray]:
    """Load CSV numeric diagnostics while retaining optimizer stage names."""
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return {key: np.asarray([row.get(key, "") if key == "stage" else _number(row.get(key))
                                 for row in rows]) for key in (reader.fieldnames or [])}


def _protocol_label(optimizer: str) -> str:
    """Display the optimization protocol in figure labels."""
    return {"adam": "Adam", "adam_lbfgs": "Adam → L-BFGS", "lbfgs": "L-BFGS only"}.get(optimizer, optimizer)


def _protocol_caption(runs: list[RunArtifacts]) -> str:
    """Describe all protocols represented in a figure."""
    return "Protocol: " + "; ".join(_protocol_label(value) for value in sorted({run.optimizer for run in runs}))


def _run_label(run: RunArtifacts, runs: list[RunArtifacts]) -> str:
    """Disambiguate otherwise identical activations across protocols."""
    if len({item.optimizer for item in runs}) > 1:
        return f"{run.label} | {_protocol_label(run.optimizer)}"
    return run.label


def _stage_label(stage: str) -> str:
    """Distinguish Adam updates from L-BFGS outer steps."""
    return "L-BFGS outer step" if stage == "lbfgs" else "Adam epoch"


def activation_derivatives(z: np.ndarray, kind: str, omega: float = 1.0,
                           gamma: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate activation and its first two derivatives with respect to z.

    Tanh is tanh(z), SIREN is sin(omega*z), and Gaussian is
    exp(-gamma*(omega*z)**2)*sin(omega*z). The sine-family derivatives include
    both frequency chain-rule factors, omega and omega**2.
    """
    z = np.asarray(z, dtype=float)
    if kind == "tanh":
        value = np.tanh(z)
        first = 1.0 - value**2
        return value, first, -2.0 * value * first
    u = omega * z
    if kind == "siren":
        return np.sin(u), omega * np.cos(u), -(omega**2) * np.sin(u)
    if kind != "gaussian":
        raise ValueError(f"Unknown activation: {kind}")
    envelope = np.exp(-gamma * u**2)
    value = envelope * np.sin(u)
    first = omega * envelope * (np.cos(u) - 2.0 * gamma * u * np.sin(u))
    second = omega**2 * envelope * (
        (4.0 * gamma**2 * u**2 - 2.0 * gamma - 1.0) * np.sin(u)
        - 4.0 * gamma * u * np.cos(u)
    )
    return value, first, second


def sample_statistics(values: list[float]) -> tuple[float, float, int]:
    """Return finite-value mean, sample SD (ddof=1), and sample count.

    One observation receives SD=0 for plotting only; it gives no estimate of
    seed-to-seed uncertainty. Missing values are not replaced by zero.
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return float("nan"), float("nan"), 0
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0, int(array.size)


def heatmap_limits(runs: list[RunArtifacts]) -> tuple[float, float, float]:
    """Find shared field and absolute-error limits over every available run."""
    minima, maxima, errors = [], [], []
    for run in runs:
        if run.predictions is None:
            continue
        for key in ("exact", "prediction"):
            field = run.predictions.get(key, np.array([]))
            finite = field[np.isfinite(field)]
            if finite.size:
                minima.append(float(finite.min()))
                maxima.append(float(finite.max()))
        error = run.predictions.get("abs_error", np.array([]))
        finite_error = error[np.isfinite(error)]
        if finite_error.size:
            errors.append(float(np.max(np.abs(finite_error))))
    minimum, maximum = (min(minima), max(maxima)) if minima else (-1.0, 1.0)
    if minimum == maximum:
        padding = max(abs(minimum) * 0.01, 0.01)
        minimum, maximum = minimum - padding, maximum + padding
    return minimum, maximum, max(max(errors, default=0.0), 1e-12)


def _variant_order(variant: tuple[str, float | None]) -> tuple[int, float]:
    """Sort baselines before Gaussian variants in increasing gamma."""
    model, gamma = variant
    return {"tanh": 0, "siren": 1, "gaussian": 2}.get(model, 3), gamma or 0.0


def _variant_label(variant: tuple[str, float | None]) -> str:
    """Return the same activation name in every plot."""
    model, gamma = variant
    if model == "gaussian":
        return f"Gaussian γ={gamma:g}" if gamma is not None else "Gaussian"
    return {"tanh": "tanh", "siren": "SIREN"}.get(model, model)


def _style(variant: tuple[str, float | None]) -> dict[str, Any]:
    """Assign fixed, distinct colors and line styles to activation variants."""
    model, gamma = variant
    if model == "tanh":
        return {"color": "#4E79A7", "linestyle": "-"}
    if model == "siren":
        return {"color": "#F28E2B", "linestyle": "--"}
    colors = {0.0: "#59A14F", 0.001: "#B07AA1", 0.01: "#E15759", 0.05: "#76B7B2"}
    return {"color": colors.get(gamma, "#9C755F"), "linestyle": "-." if gamma == 0 else "-"}


def _save(fig: Any, destination: Path) -> Path:
    """Save a self-contained PNG and release matplotlib figure resources."""
    fig.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination


def _activation_figure(figures_dir: Path, omega: float = 1.0,
                       run_gammas: set[float] | None = None) -> Path:
    """Compare canonical and run-specific activations at explicit omega."""
    z = np.linspace(-12, 12, 1601)
    canonical_gammas = {0.0, 0.001, 0.01, 0.05}
    custom_gammas = (run_gammas or set()) - canonical_gammas
    gammas = sorted(canonical_gammas | custom_gammas)
    variants = [("tanh", None), ("siren", None)] + [("gaussian", gamma) for gamma in gammas]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharex=True, layout="constrained")
    for variant in variants:
        values = activation_derivatives(z, variant[0], omega, variant[1] or 0.0)
        for axis, value in zip(axes, values):
            axis.plot(z, value, label=_variant_label(variant), linewidth=1.6, **_style(variant))
    for axis, title, ylabel in zip(axes, ("Activation", "First derivative", "Second derivative"),
                                  ("f(z)", "df/dz", "d²f/dz²")):
        axis.set(title=title, xlabel="z", ylabel=ylabel)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="upper left")
    title = (f"Activation comparison: ω={omega:g}; tanh(z), sin(ωz), exp[−γ(ωz)²] sin(ωz)\n"
             "Derivatives are with respect to z, including ω and ω² chain factors")
    if custom_gammas:
        title += ("\nCanonical reference γ values plus custom run γ values: "
                  + ", ".join(f"{gamma:g}" for gamma in sorted(custom_gammas)))
    fig.suptitle(title, fontsize=12)
    return _save(fig, figures_dir / "activation_functions.png")


def _history_figure(runs: list[RunArtifacts], figures_dir: Path, fields: list[tuple[str, str]],
                    filename: str, title: str, *, source: str = "history",
                    stage: str | None = "adam", elapsed_time: bool = False) -> Path:
    """Plot diagnostics without equating Adam epochs and L-BFGS outer steps.

    Evaluation CSVs contain post-update full-pool losses and independent-grid
    errors. Missing evaluation CSVs fall back to explicitly labeled legacy
    histories. Time plots use separate lines for the two optimizer stages.
    """
    seeds = sorted({run.seed for run in runs})
    fig, axes = plt.subplots(len(fields), len(seeds), figsize=(5.6 * len(seeds), 3.0 * len(fields) + 1),
                             squeeze=False, sharex="col", sharey="row", layout="constrained")
    for column, seed in enumerate(seeds):
        seed_runs = [run for run in runs if run.seed == seed]
        for row, (field, label) in enumerate(fields):
            axis = axes[row, column]
            for run in seed_runs:
                use_evaluations = source == "evaluations" and bool(run.evaluations)
                series = run.evaluations if use_evaluations else run.history
                x_field = "train_seconds" if elapsed_time else "stage_step" if "stage_step" in series else "epoch"
                coordinates = series.get(x_field, np.array([]))
                values = series.get(field, np.array([]))
                if coordinates.shape != values.shape:
                    continue
                stages = series.get("stage", np.full(coordinates.shape, "adam"))
                for selected_stage in (("adam", "lbfgs") if stage is None else (stage,)):
                    mask = np.isfinite(coordinates) & np.isfinite(values) & (stages == selected_stage)
                    if field != "center_amplitude_ratio":
                        mask &= values > 0
                    if not mask.any():
                        continue
                    line_label = _run_label(run, runs)
                    if stage is None:
                        line_label += " | " + ("Adam" if selected_stage == "adam" else "L-BFGS")
                    if source == "evaluations" and not use_evaluations:
                        line_label += " [legacy history]"
                    style = _style((run.model, run.gamma))
                    if stage is None and selected_stage == "lbfgs":
                        style = {**style, "linestyle": ":"}
                    axis.plot(coordinates[mask], values[mask], label=line_label, linewidth=1.6,
                              marker="." if use_evaluations else None, **style)
            if field != "center_amplitude_ratio":
                axis.set_yscale("log")
            axis.set_ylabel(label)
            axis.grid(alpha=0.2, which="both")
            if row == 0:
                axis.set_title(f"Seed {seed}")
                if axis.lines:
                    axis.legend(fontsize=8, loc="best")
            if not axis.lines:
                axis.text(0.5, 0.5, "No positive finite observations", ha="center", transform=axis.transAxes)
            if row == len(fields) - 1:
                axis.set_xlabel("Cumulative training time (s)" if elapsed_time else _stage_label(stage or "adam"))
    if source == "evaluations":
        detail = ("post-update: full training pool losses; independent evaluation grid / initial line"
                  if any(run.evaluations for run in runs) else
                  "Legacy history: sampled pre-update losses; recorded evaluation epochs only")
        if any(not run.evaluations for run in runs) and any(run.evaluations for run in runs):
            detail += "; legacy histories labeled separately"
    else:
        detail = ("L-BFGS first closure diagnostics (before the outer-step update)" if stage == "lbfgs" else
                  "Adam minibatch diagnostics (before the update)")
    fig.suptitle(f"{title}\n{detail}\n{_protocol_caption(runs)}; nonpositive log values / nonfinite values omitted",
                 fontsize=11)
    return _save(fig, figures_dir / filename)


def save_center_snapshot(run_dir: Path, arrays: dict[str, np.ndarray], stage: str,
                         step: int, label: str) -> Path:
    """Save a per-run center trace immediately after a training evaluation."""
    destination = Path(run_dir) / "snapshots" / f"center_{stage}_{step:06d}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 4.5), layout="constrained")
    axis.plot(arrays["center_t"], arrays["center_exact"], color="black", linewidth=2.4, label="Exact")
    axis.plot(arrays["center_t"], arrays["center_prediction"], color="#4E79A7",
              linewidth=1.7, label=label)
    axis.set(title=f"Center trace at x=0 | {_stage_label(stage)} {step}\n{label}",
             xlabel="t", ylabel="u(0, t)")
    axis.legend(fontsize=8, loc="best")
    axis.grid(alpha=0.2)
    return _save(fig, destination)


def _periodic_trace_figures(runs: list[RunArtifacts], figures_dir: Path) -> list[Path]:
    """Compare all variants at each saved optimizer-stage evaluation point."""
    paths = []
    for seed in sorted({run.seed for run in runs}):
        seed_runs = [run for run in runs if run.seed == seed]
        points = sorted({key for run in seed_runs for key in run.snapshots})
        for stage, step in points:
            fig, axis = plt.subplots(figsize=(10, 4.8), layout="constrained")
            exact_drawn = False
            for run in seed_runs:
                arrays = run.snapshots.get((stage, step))
                required = {"center_t", "center_exact", "center_prediction"}
                if arrays is None or not required.issubset(arrays):
                    axis.plot([], [], label=f"{_run_label(run, runs)}: unavailable",
                              **_style((run.model, run.gamma)))
                    continue
                if not exact_drawn:
                    axis.plot(arrays["center_t"], arrays["center_exact"], color="black",
                              linewidth=2.4, label="Exact")
                    exact_drawn = True
                axis.plot(arrays["center_t"], arrays["center_prediction"], label=_run_label(run, runs),
                          linewidth=1.7, **_style((run.model, run.gamma)))
            axis.set(title=f"Center trace at x=0 | Seed {seed} | {_stage_label(stage)} {step}\n"
                           + _protocol_caption(seed_runs), xlabel="t", ylabel="u(0, t)")
            axis.legend(fontsize=8, loc="best")
            axis.grid(alpha=0.2)
            paths.append(_save(fig, figures_dir / f"center_trace_seed_{seed}_{stage}_{step:06d}.png"))
    return paths


def _heatmap_figure(run: RunArtifacts, limits: tuple[float, float, float], figures_dir: Path) -> Path:
    """Save exact, predicted, and absolute-error fields on experiment-wide scales."""
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.1), sharex=True, sharey=True, layout="constrained")
    required = {"x", "t", "exact", "prediction", "abs_error"}
    if run.predictions is None or not required.issubset(run.predictions):
        for axis in axes:
            axis.set_axis_off()
        axes[1].text(0.5, 0.5, f"Predictions unavailable\nStatus: {run.status}",
                     ha="center", va="center", transform=axes[1].transAxes)
    else:
        arrays = run.predictions
        for axis, key, title in zip(axes, ("exact", "prediction", "abs_error"),
                                     ("Exact solution", "PINN prediction", "Absolute error")):
            is_error = key == "abs_error"
            field = np.ma.masked_invalid(arrays[key])
            mesh = axis.pcolormesh(arrays["x"], arrays["t"], field, shading="auto",
                                   cmap="magma" if is_error else "RdBu_r",
                                   vmin=0.0 if is_error else limits[0],
                                   vmax=limits[2] if is_error else limits[1], rasterized=True)
            axis.set(title=title, xlabel="x", xlim=(arrays["x"][0], arrays["x"][-1]),
                     ylim=(arrays["t"][0], arrays["t"][-1]))
            fig.colorbar(mesh, ax=axis, label="|u predicted − u exact|" if is_error else "u(x, t)")
        axes[0].set_ylabel("t")
    fig.suptitle(f"{run.label} | Seed {run.seed} | Status: {run.status} | {_protocol_label(run.optimizer)}\n"
                 "Field and error color limits are shared across all runs", fontsize=12)
    return _save(fig, figures_dir / f"heatmap_{run.run_id}.png")


def _trace_figures(runs: list[RunArtifacts], figures_dir: Path) -> list[Path]:
    """Overlay all activation variants on center traces and spatial snapshots."""
    paths = []
    for seed in sorted({run.seed for run in runs}):
        seed_runs = [run for run in runs if run.seed == seed]
        available = [run for run in seed_runs if run.predictions is not None]
        fig, axis = plt.subplots(figsize=(10, 4.8), layout="constrained")
        exact_drawn = False
        for run in seed_runs:
            arrays = run.predictions
            if arrays is not None and {"center_t", "center_exact", "center_prediction"}.issubset(arrays):
                if not exact_drawn:
                    axis.plot(arrays["center_t"], arrays["center_exact"], color="black", linewidth=2.4, label="Exact")
                    exact_drawn = True
                axis.plot(arrays["center_t"], arrays["center_prediction"], label=_run_label(run, runs),
                          linewidth=1.7, **_style((run.model, run.gamma)))
            else:
                axis.plot([], [], label=f"{run.label}: no prediction", **_style((run.model, run.gamma)))
        axis.set(title=f"Final center trace at x=0 | Seed {seed}\n{_protocol_caption(seed_runs)}",
                 xlabel="t", ylabel="u(0, t)")
        axis.legend(fontsize=8, loc="best")
        axis.grid(alpha=0.2)
        paths.append(_save(fig, figures_dir / f"center_trace_seed_{seed}.png"))

        reference = next((run.predictions for run in available if "profile_times" in run.predictions), None)
        times = reference["profile_times"] if reference is not None else np.array([])
        count = max(len(times), 1)
        columns = min(count, 3)
        rows = (count + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(5.1 * columns, 3.7 * rows + 0.6),
                                 sharex=True, sharey=True, squeeze=False, layout="constrained")
        if not len(times):
            axes.flat[0].text(0.5, 0.5, "Spatial profiles unavailable", ha="center", transform=axes.flat[0].transAxes)
        for index, time in enumerate(times):
            axis = axes.flat[index]
            axis.plot(reference["profile_x"], reference["profile_exact"][index],
                      color="black", linewidth=2.4, label="Exact")
            for run in seed_runs:
                arrays = run.predictions
                required = {"profile_times", "profile_x", "profile_prediction"}
                if arrays is not None and required.issubset(arrays):
                    matching = np.flatnonzero(np.isclose(arrays["profile_times"], time))
                    if matching.size:
                        axis.plot(arrays["profile_x"], arrays["profile_prediction"][matching[0]],
                                  label=_run_label(run, runs), linewidth=1.7, **_style((run.model, run.gamma)))
                        continue
                axis.plot([], [], label=f"{run.label}: unavailable", **_style((run.model, run.gamma)))
            axis.set(title=f"t={time:.4g}", xlabel="x", ylabel="u(x, t)")
            axis.grid(alpha=0.2)
        for index in range(count, rows * columns):
            axes.flat[index].set_axis_off()
        if len(times):
            axes.flat[0].legend(fontsize=7, loc="best")
        fig.suptitle(f"Final spatial profile snapshots | Seed {seed}\n{_protocol_caption(seed_runs)}", fontsize=12)
        paths.append(_save(fig, figures_dir / f"spatial_profiles_seed_{seed}.png"))
    return paths


def _summary_symlog(axis: Any, means: list[float]) -> None:
    """Set a data-scaled linear region and restrained decade tick density."""
    positive = [mean for mean in means if np.isfinite(mean) and mean > 0]
    threshold = max(min(positive) * 0.1, np.finfo(float).tiny) if positive else 1.0
    axis.set_yscale("symlog", linthresh=threshold)
    # The locator uses a soft target separately across signed decade ranges.
    axis.yaxis.get_major_locator().set_params(numticks=6)
    # Interior signed-decade ticks can overlap zero inside the linear region.
    # These are static saved figures, so retain only the readable major ticks.
    ticks = [tick for tick in axis.get_yticks() if tick == 0 or abs(tick) >= threshold]
    axis.yaxis.set_major_locator(FixedLocator(ticks))


def _summary_figure(runs: list[RunArtifacts], figures_dir: Path) -> Path:
    """Aggregate completed-run metrics, retaining all runs in failure counts."""
    keys = sorted({(run.model, run.gamma, run.optimizer) for run in runs},
                  key=lambda key: (_variant_order(key[:2]), key[2]))
    variants = [key[:2] for key in keys]
    groups = [[run for run in runs if (run.model, run.gamma, run.optimizer) == key] for key in keys]
    completed_groups = [[run for run in group if run.status == "completed"] for group in groups]
    labels = ["tanh" if model == "tanh" else "SIREN" if model == "siren" else f"γ={gamma:g}"
              for model, gamma in variants]
    if len({run.optimizer for run in runs}) > 1:
        labels = [f"{label}\n{_protocol_label(key[2])}" for label, key in zip(labels, keys)]
    x = np.arange(len(variants))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
    metrics = [("relative_l2", "Relative L2 error"), ("total_loss", "Final total loss"),
               ("train_seconds", "Training time (s)"), ("loss_spike_count", "Loss spike count")]
    for axis, (metric, title) in zip([axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 2]], metrics):
        positive_means = []
        for position, variant, group in zip(x, variants, completed_groups):
            mean, std, count = sample_statistics([_number(run.metrics.get(metric)) for run in group])
            if count:
                if mean > 0:
                    positive_means.append(mean)
                axis.errorbar(position, mean, yerr=std, fmt="D" if variant[0] == "siren" else "o",
                              color=_style(variant)["color"], capsize=4, markersize=6)
                axis.annotate(f"n={count}", (position, mean + std), xytext=(0, 5),
                              textcoords="offset points", ha="center", fontsize=7)
            else:
                axis.text(position, 0.05, "n=0", transform=axis.get_xaxis_transform(), ha="center", fontsize=7)
        if (metric in {"relative_l2", "total_loss"} and positive_means
                and max(positive_means) / min(positive_means) > 10):
            _summary_symlog(axis, positive_means)
        axis.set_title(title)
        axis.set_ylabel(title + (" (symlog)" if axis.get_yscale() == "symlog" else ""))
        axis.margins(y=0.2)

    axis = axes[1, 0]
    gradient_means = []
    for shift, metric, label, marker, color in [(-0.19, "grad_norm_min", "Minimum", "v", "#4E79A7"),
                                               (0, "grad_norm_final", "Final", "o", "#59A14F"),
                                               (0.19, "grad_norm_max", "Maximum", "^", "#E15759")]:
        means, deviations = [], []
        for group in completed_groups:
            mean, std, _ = sample_statistics([_number(run.metrics.get(metric)) for run in group])
            means.append(mean)
            deviations.append(std)
        gradient_means.extend(means)
        axis.errorbar(x + shift, means, yerr=deviations, fmt=marker, color=color, capsize=3, label=label)
    _summary_symlog(axis, gradient_means)
    axis.set(title="Gradient norms across training", ylabel="Gradient norm (symlog)")
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    nonfinite = [sum(bool(run.metrics.get("has_nonfinite")) or run.status == "nonfinite" for run in group) for group in groups]
    diverged = [sum(bool(run.metrics.get("divergence_detected")) or run.status == "diverged" for run in group) for group in groups]
    failures = [sum(run.status not in {"completed", "ok", "success"} for run in group) for group in groups]
    axis.bar(x - 0.23, failures, width=0.23, label="Failed status", color="#777777")
    axis.bar(x, nonfinite, width=0.23, label="Nonfinite", color="#E15759")
    axis.bar(x + 0.23, diverged, width=0.23, label="Divergence", color="#B07AA1")
    axis.set(title="Stability: failure counts (flags may overlap)", ylabel="Number of runs")
    axis.set_ylim(0, max(max(failures + nonfinite + diverged, default=0), 1) + 1)
    axis.yaxis.get_major_locator().set_params(integer=True)
    axis.legend(fontsize=8)
    for position, group in zip(x, groups):
        axis.text(position, axis.get_ylim()[1] - 0.25, f"N={len(group)}", ha="center", fontsize=7)
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_xlabel("Activation / Gaussian damping γ (categorical spacing)")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("Across-seed summary: mean ± sample SD (ddof=1); finite completed-run values only\n"
                 + _protocol_caption(runs) + "; protocols summarized separately\n"
                 "n=1 gives SD=0 and provides no evidence about seed uncertainty. Failure counts include all runs.\n"
                 "SIREN baseline uses diamond markers in metric panels. Inspect failure counts alongside accuracy.\n"
                 "Symmetric SD bars may extend below zero; this does not indicate negative observed losses.", fontsize=11)
    return _save(fig, figures_dir / "gamma_summary.png")


def plot_all(output_dir: Path, *, activation_omega: float = 1.0,
             figures_dir: Path | None = None) -> list[Path]:
    """Plot disk data, optionally into a new directory preserving the input.

    Without figures_dir the historical OUTPUT_DIR/figures behavior is retained.
    An explicit destination must be empty to avoid overwriting archived figures.
    """
    output_dir = Path(output_dir)
    runs = load_runs(output_dir)
    if not runs:
        raise ValueError(f"No run artifacts found under {output_dir / 'runs'}")
    if figures_dir is None:
        figures_dir = output_dir / "figures"
    else:
        figures_dir = Path(figures_dir)
        if figures_dir.exists() and (not figures_dir.is_dir() or any(figures_dir.iterdir())):
            raise FileExistsError(f"Figure output directory must be empty: {figures_dir}")
    figures_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 10, "figure.titlesize": 12, "savefig.facecolor": "white"}):
        run_gammas = {run.gamma for run in runs if run.model == "gaussian" and run.gamma is not None}
        paths = [_activation_figure(figures_dir, activation_omega, run_gammas)]
        losses = [("total_loss", "Total loss"), ("pde_loss", "PDE loss"),
                  ("ic_loss", "IC loss"), ("bc_loss", "BC loss")]
        errors = [("relative_l2", "Relative L2 error"), ("residual_rms", "PDE residual RMS")]
        gradients = [("grad_norm", "Global gradient norm"),
                     ("grad_norm_first", "First-layer gradient norm"),
                     ("grad_norm_last", "Last-layer gradient norm")]
        velocity = [("ic_value_loss", "IC displacement loss (training pool)"),
                    ("ic_velocity_loss", "IC velocity loss (training pool)"),
                    ("ic_velocity_relative_l2", "IC velocity relative L2 (held-out)"),
                    ("center_amplitude_ratio", "Center amplitude ratio")]
        observed_stages = set()
        for run in runs:
            for series in (run.history, run.evaluations):
                if 'stage' in series:
                    observed_stages.update(series['stage'].tolist())
                elif len(series.get('epoch', [])):
                    observed_stages.add('adam')
        stages = [stage for stage in ('adam', 'lbfgs') if stage in observed_stages] or ['adam']
        for stage in stages:
            prefix = "" if stage == "adam" else "lbfgs_"
            stage_name = "Adam" if stage == "adam" else "L-BFGS"
            paths.append(_history_figure(runs, figures_dir, losses, prefix + "learning_curves.png",
                                          f"{stage_name}: training-loss convergence", source="evaluations", stage=stage))
            paths.append(_history_figure(runs, figures_dir, gradients, prefix + "gradient_norms.png",
                                          f"{stage_name}: gradient norms during training", stage=stage))
            paths.append(_history_figure(runs, figures_dir, errors, prefix + "error_convergence.png",
                                          f"{stage_name}: evaluation-grid error and residual convergence",
                                          source="evaluations", stage=stage))
            if any(run.evaluations for run in runs):
                paths.append(_history_figure(runs, figures_dir, velocity, prefix + "ic_velocity_convergence.png",
                                              f"{stage_name}: initial-condition and oscillation diagnostics",
                                              source="evaluations", stage=stage))
        if any(run.evaluations for run in runs):
            paths.append(_history_figure(runs, figures_dir, [losses[0], *errors, velocity[2]],
                                          "training_time_convergence.png", "Convergence versus actual training time",
                                          source="evaluations", stage=None, elapsed_time=True))
        limits = heatmap_limits(runs)
        paths.extend(_heatmap_figure(run, limits, figures_dir) for run in runs)
        paths.extend(_trace_figures(runs, figures_dir))
        paths.extend(_periodic_trace_figures(runs, figures_dir))
        paths.append(_summary_figure(runs, figures_dir))
    return paths


def main(argv: list[str] | None = None) -> None:
    """Expose plot regeneration as a headless command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="Directory containing runs/ and experiment.json")
    parser.add_argument("--output-dir", type=Path,
                        help="New empty figure directory; preserves saved results (default: RESULTS/figures)")
    parser.add_argument("--activation-omega", type=float, default=1.0,
                        help="Explicit frequency for the activation illustration only (default: 1)")
    args = parser.parse_args(argv)
    if not np.isfinite(args.activation_omega) or args.activation_omega <= 0:
        parser.error("--activation-omega must be positive and finite")
    for path in plot_all(args.results, activation_omega=args.activation_omega, figures_dir=args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
