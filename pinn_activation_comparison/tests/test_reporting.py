"""Trustworthy serialization and statistics for completed and failed runs."""

import csv
import json
import math

import numpy as np
import pytest


def test_write_json_replaces_nonfinite_nested_values_with_null(tmp_path):
    from pinn_activation_comparison.reporting import write_json

    destination = tmp_path / "nested" / "metrics.json"
    write_json(destination, {
        "metric": float("nan"), "positive": float("inf"),
        "nested": [1.25, {"negative": -float("inf")}, (2, float("nan"))],
        "path": tmp_path / "checkpoint.pt", "numpy": np.float32(0.5),
    })
    raw = destination.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    loaded = json.loads(raw)
    assert loaded["metric"] is None
    assert loaded["positive"] is None
    assert loaded["nested"] == [1.25, {"negative": None}, [2, None]]
    assert loaded["path"] == str(tmp_path / "checkpoint.pt")
    assert loaded["numpy"] == 0.5


def test_write_csv_preserves_union_of_heterogeneous_record_keys(tmp_path):
    from pinn_activation_comparison.reporting import write_csv

    destination = tmp_path / "nested" / "summary.csv"
    write_csv(destination, [
        {"model": "tanh", "relative_l2": 0.5},
        {"model": "siren", "status": "nonfinite", "reason": "comma, newline\nvalue"},
    ])
    with destination.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert set(reader.fieldnames) == {"model", "relative_l2", "status", "reason"}
    assert rows[0]["relative_l2"] == "0.5"
    assert rows[0]["status"] == ""
    assert rows[1]["relative_l2"] == ""
    assert rows[1]["reason"] == "comma, newline\nvalue"


def test_aggregate_uses_sample_std_and_excludes_failed_and_nonfinite_values():
    from pinn_activation_comparison.reporting import aggregate_metrics

    common = {"model": "siren", "gamma": None, "comparison": "controlled"}
    records = [
        dict(common, seed=0, status="completed", relative_l2=1.0, total_loss=2.0),
        dict(common, seed=1, status="completed", relative_l2=3.0, total_loss=float("nan")),
        dict(common, seed=2, status="diverged", relative_l2=100.0, total_loss=1e20,
             divergence_detected=True),
        dict(common, seed=3, status="nonfinite", relative_l2=float("inf"), has_nonfinite=True),
    ]
    row, = aggregate_metrics(records)
    assert row["n_total"] == 4
    assert row["n_completed"] == 2
    assert row["n_failed"] == 2
    assert row["n_nonfinite"] == 1
    assert row["n_diverged"] == 1
    assert row["relative_l2_mean"] == 2.0
    assert row["relative_l2_std"] == pytest.approx(math.sqrt(2))
    assert row["relative_l2_count"] == 2
    assert row["total_loss_mean"] == 2.0
    assert row["total_loss_std"] == 0.0
    assert row["total_loss_count"] == 1


def test_aggregate_separates_gamma_and_comparison_and_keeps_failed_groups():
    from pinn_activation_comparison.reporting import aggregate_metrics

    records = [
        {"model": "gaussian", "gamma": 0.0, "comparison": "controlled", "status": "completed", "relative_l2": 0.5},
        {"model": "gaussian", "gamma": 0.01, "comparison": "controlled", "status": "completed", "relative_l2": 0.25},
        {"model": "gaussian", "gamma": 0.01, "comparison": "tuned", "status": "diverged", "relative_l2": 1e8},
    ]
    groups = {(row["gamma"], row["comparison"]): row for row in aggregate_metrics(records)}
    assert len(groups) == 3
    assert groups[(0.0, "controlled")]["relative_l2_mean"] == 0.5
    assert groups[(0.01, "controlled")]["relative_l2_mean"] == 0.25
    failed = groups[(0.01, "tuned")]
    assert failed["n_failed"] == 1
    assert failed["relative_l2_count"] == 0
    assert failed["relative_l2_mean"] is None
    assert failed["relative_l2_std"] is None
    assert aggregate_metrics([]) == []
