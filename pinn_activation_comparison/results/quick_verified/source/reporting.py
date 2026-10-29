"""Strict artifact serialization and completed-run summary statistics."""

from __future__ import annotations

from collections.abc import Mapping
import csv
import json
import math
from numbers import Integral, Real
from pathlib import Path, PurePath
import statistics


AGGREGATE_METRICS = (
    "relative_l2", "residual_rms", "total_loss", "pde_loss", "ic_loss",
    "ic_value_loss", "ic_velocity_loss", "bc_loss", "train_seconds",
    "grad_norm_initial", "grad_norm_final", "grad_norm_min", "grad_norm_max",
    "loss_spike_count", "epochs_completed",
)


def _json_safe(value):
    """Normalize nested numbers and paths without emitting nonstandard NaN."""
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, PurePath):
        return str(value)
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value) if math.isfinite(value) else None
    return value


def write_json(path, obj):
    """Write portable UTF-8 JSON, representing nonfinite numbers as null."""
    path = Path(path)
    serialized = json.dumps(_json_safe(obj), ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")


def write_csv(path, rows):
    """Write every observed column in first-seen order, including failure data."""
    path = Path(path)
    rows = list(rows)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_json_safe(row) for row in rows)


def aggregate_metrics(records):
    """Aggregate finite completed results, retaining all run/failure counts.

    Each metric gets its own count because failed evaluation fields can be
    unavailable independently. A singleton's SD is descriptively zero; an empty
    successful sample has null mean/SD rather than a misleading zero.
    """
    groups = {}
    for record in records:
        key = (record["model"], record.get("gamma"), record.get("comparison", "controlled"))
        groups.setdefault(key, []).append(record)
    output = []
    for (model, gamma, comparison), group in groups.items():
        completed = [record for record in group if record.get("status") == "completed"]
        row = {
            "model": model, "gamma": gamma, "comparison": comparison,
            "n_total": len(group), "n_completed": len(completed),
            "n_failed": len(group) - len(completed),
            "n_nonfinite": sum(bool(record.get("has_nonfinite")) or record.get("status") == "nonfinite"
                               for record in group),
            "n_diverged": sum(bool(record.get("divergence_detected")) or record.get("status") == "diverged"
                              for record in group),
        }
        for metric in AGGREGATE_METRICS:
            values = [float(record[metric]) for record in completed
                      if isinstance(record.get(metric), Real)
                      and not isinstance(record[metric], bool)
                      and math.isfinite(record[metric])]
            row[metric + "_count"] = len(values)
            row[metric + "_mean"] = statistics.mean(values) if values else None
            row[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
        output.append(row)
    return output
