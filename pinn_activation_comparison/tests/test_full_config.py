"""Full training budgets, resumable configuration, and protocol summaries."""

import json

import pytest
import torch

from pinn_activation_comparison.config import ExperimentConfig, parse_args
from pinn_activation_comparison.reporting import aggregate_metrics


def test_full_defaults_keep_shared_architecture_and_use_three_seeds():
    quick = parse_args([])
    full = parse_args(["--mode", "full"])
    assert (quick.epochs, quick.eval_every, quick.checkpoint_every, quick.seeds) == (100, 20, 20, (0,))
    assert (full.epochs, full.eval_every, full.checkpoint_every, full.seeds) == (5000, 500, 500, (0, 1, 2))
    assert (full.n_f, full.n_ic, full.n_bc, full.batch_size) == (4096, 256, 256, 512)
    assert (quick.hidden_layers, quick.width) == (full.hidden_layers, full.width) == (4, 64)
    assert quick.optimizer == full.optimizer == "adam"


@pytest.mark.parametrize("target", [10000, 20000])
def test_explicit_full_epoch_target_and_seed_override(target):
    config = parse_args(["--mode", "full", "--epochs", str(target), "--seed", "9"])
    assert config.epochs == target
    assert config.seeds == (9,)


def test_lbfgs_protocol_has_explicit_defaults_and_cli_overrides():
    config = parse_args(["--optimizer", "adam_lbfgs"])
    assert config.optimizer == "adam_lbfgs"
    assert (config.lbfgs_steps, config.lbfgs_max_iter, config.lbfgs_history_size) == (100, 20, 50)
    assert (config.lbfgs_lr, config.lbfgs_eval_every) == (1.0, 10)
    overridden = parse_args([
        "--optimizer", "adam_lbfgs", "--lbfgs-steps", "5", "--lbfgs-max-iter", "3",
        "--lbfgs-history-size", "7", "--lbfgs-lr", "0.3", "--lbfgs-eval-every", "2",
        "--checkpoint-every", "4",
    ])
    assert (overridden.lbfgs_steps, overridden.lbfgs_max_iter, overridden.lbfgs_history_size) == (5, 3, 7)
    assert (overridden.lbfgs_lr, overridden.lbfgs_eval_every, overridden.checkpoint_every) == (0.3, 2, 4)


@pytest.mark.parametrize("field,value", [
    ("checkpoint_every", 0), ("checkpoint_every", True), ("checkpoint_every", 1.5),
    ("lbfgs_steps", -1), ("lbfgs_steps", 0), ("lbfgs_steps", True),
    ("lbfgs_max_iter", 0), ("lbfgs_max_iter", 2.5),
    ("lbfgs_history_size", 0), ("lbfgs_history_size", False),
    ("lbfgs_eval_every", 0), ("lbfgs_eval_every", "10"),
    ("lbfgs_lr", 0), ("lbfgs_lr", -0.1), ("lbfgs_lr", float("inf")),
    ("lbfgs_lr", float("nan")), ("lbfgs_lr", True), ("optimizer", "unsupported"),
])
def test_invalid_checkpoint_and_lbfgs_settings_fail_validation(field, value):
    with pytest.raises(ValueError, match=field):
        ExperimentConfig(**{field: value})


def _saved_experiment(tmp_path, **overrides):
    source = tmp_path / "original"
    source.mkdir()
    saved = ExperimentConfig(
        mode="full", models=("siren",), gammas=(0.01,), seeds=(4, 5),
        epochs=50, width=12, n_f=31, output_dir=str(source), optimizer="adam_lbfgs",
        checkpoint_every=7,
    ).to_dict()
    saved.update(overrides)
    (source / "experiment.json").write_text(json.dumps({"config": saved}), encoding="utf-8")
    return source, saved


def test_resume_directory_inherits_saved_config_and_preserves_original_output(tmp_path):
    source, _ = _saved_experiment(tmp_path)
    config = parse_args(["--resume", str(source)])
    assert config.resume == str(source)
    assert (config.epochs, config.width, config.n_f, config.checkpoint_every) == (50, 12, 31, 7)
    assert (config.models, config.seeds, config.optimizer) == (("siren",), (4, 5), "adam_lbfgs")
    assert config.output_dir == str(source) + "_resumed"


def test_resume_saved_then_explicit_json_then_cli_priority(tmp_path):
    source, _ = _saved_experiment(tmp_path)
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"epochs": 10000, "width": 18, "seeds": [3], "lbfgs_steps": 8}))
    config = parse_args([
        "--resume", str(source), "--config", str(path), "--epochs", "20000", "--seed", "8",
    ])
    assert (config.epochs, config.width, config.n_f, config.seeds, config.lbfgs_steps) == (20000, 18, 31, (8,), 8)
    assert config.output_dir == str(source) + "_resumed"


def test_resume_can_be_selected_by_json_and_explicit_output_is_respected(tmp_path):
    source, _ = _saved_experiment(tmp_path)
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"resume": str(source), "output_dir": str(tmp_path / "json-output")}))
    config = parse_args(["--config", str(path)])
    assert config.resume == str(source)
    assert config.output_dir == str(tmp_path / "json-output")
    config = parse_args(["--config", str(path), "--output-dir", str(tmp_path / "cli-output")])
    assert config.output_dir == str(tmp_path / "cli-output")


@pytest.mark.parametrize("model,gamma", [("gaussian", 0.01), ("siren", None)])
def test_checkpoint_resume_selects_its_condition(tmp_path, model, gamma):
    source, saved = _saved_experiment(tmp_path)
    checkpoint = source / "checkpoint.pt"
    torch.save({"config": saved, "condition": {"model": model, "gamma": gamma, "seed": 5}}, checkpoint)
    config = parse_args(["--resume", str(checkpoint)])
    assert config.resume == str(checkpoint)
    assert (config.models, config.seeds) == ((model,), (5,))
    if gamma is not None:
        assert config.gammas == (gamma,)
    assert config.output_dir == str(source) + "_resumed"


def test_checkpoint_condition_can_be_overridden_explicitly(tmp_path):
    source, saved = _saved_experiment(tmp_path)
    checkpoint = source / "checkpoint.pt"
    torch.save({"config": saved, "condition": {"model": "gaussian", "gamma": 0.01, "seed": 5}}, checkpoint)
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"models": ["tanh"], "seeds": [7]}))
    config = parse_args(["--resume", str(checkpoint), "--config", str(path)])
    assert (config.models, config.seeds) == (("tanh",), (7,))
    config = parse_args(["--resume", str(checkpoint), "--config", str(path), "--model", "siren", "--seed", "8"])
    assert (config.models, config.seeds) == (("siren",), (8,))


def test_old_saved_config_uses_new_optional_defaults(tmp_path):
    source, saved = _saved_experiment(tmp_path)
    for field in ("resume", "checkpoint_every", "optimizer", "lbfgs_steps", "lbfgs_max_iter",
                  "lbfgs_history_size", "lbfgs_lr", "lbfgs_eval_every"):
        saved.pop(field, None)
    (source / "experiment.json").write_text(json.dumps({"config": saved}))
    config = parse_args(["--resume", str(source)])
    assert (config.optimizer, config.lbfgs_steps) == ("adam", 100)
    assert config.resume == str(source)


def test_resume_path_is_serialized_as_string(tmp_path):
    config = ExperimentConfig(resume=tmp_path / "checkpoint.pt")
    assert config.resume == str(tmp_path / "checkpoint.pt")
    assert json.loads(json.dumps(config.to_dict()))["resume"] == str(tmp_path / "checkpoint.pt")


@pytest.mark.parametrize("manifest", [{}, {"config": []}, {"config": {"typo": 1}}])
def test_invalid_resume_metadata_fails_cleanly(tmp_path, manifest):
    source = tmp_path / "original"
    source.mkdir()
    (source / "experiment.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit):
        parse_args(["--resume", str(source)])


def test_aggregate_separates_optimizer_protocol_and_defaults_old_records_to_adam():
    common = {"model": "siren", "gamma": None, "comparison": "controlled", "status": "completed"}
    records = [
        dict(common, seed=0, relative_l2=1.0),
        dict(common, seed=1, optimizer="adam", relative_l2=3.0),
        dict(common, seed=0, optimizer="adam_lbfgs", relative_l2=0.2),
    ]
    rows = {row["optimizer"]: row for row in aggregate_metrics(records)}
    assert set(rows) == {"adam", "adam_lbfgs"}
    assert (rows["adam"]["n_completed"], rows["adam"]["relative_l2_mean"]) == (2, 2.0)
    assert (rows["adam_lbfgs"]["n_completed"], rows["adam_lbfgs"]["relative_l2_mean"]) == (1, 0.2)


@pytest.mark.parametrize("metric", [
    "ic_velocity_relative_l2", "center_amplitude_ratio", "lbfgs_steps_completed",
    "lbfgs_closure_evaluations", "adam_train_seconds", "lbfgs_train_seconds",
])
def test_new_aggregate_metrics_have_individual_finite_completed_counts(metric):
    common = {"model": "siren", "optimizer": "adam_lbfgs"}
    records = [
        dict(common, status="completed", **{metric: 1.0}),
        dict(common, status="completed", **{metric: 3.0}),
        dict(common, status="completed", **{metric: float("nan")}),
        dict(common, status="completed"),
        dict(common, status="diverged", **{metric: 100.0}),
    ]
    row, = aggregate_metrics(records)
    assert (row["n_total"], row["n_completed"], row["n_failed"]) == (5, 4, 1)
    assert row[metric + "_count"] == 2
    assert row[metric + "_mean"] == 2.0
    assert row[metric + "_std"] == pytest.approx(2 ** 0.5)
