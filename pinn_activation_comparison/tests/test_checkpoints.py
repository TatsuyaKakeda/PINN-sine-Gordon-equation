"""Resumption preserves optimizer state, experiment settings, and prior artifacts."""

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from pinn_activation_comparison.config import ExperimentConfig


def _payload(config=None, *, kind="siren", gamma=None, seed=0, adam_epoch=2, lbfgs_step=0):
    config = config or ExperimentConfig(models=("siren",), epochs=4)
    return {
        "format_version": 2, "condition": {"model": kind, "gamma": gamma, "seed": seed},
        "status": "running", "config": config.to_dict(), "model_kwargs": {"model_type": kind},
        "model_state_dict": {"weight": torch.tensor([1.0])}, "adam_state_dict": {},
        "lbfgs_state_dict": {} if lbfgs_step else None,
        "batch_rng_state": torch.Generator().get_state(), "torch_rng_state": torch.get_rng_state(),
        "history": [{"epoch": 1}], "evaluations": [{"epoch": 0}],
        "run_state": {"adam_epoch": adam_epoch, "lbfgs_step": lbfgs_step},
        "points": {"f": torch.zeros(3, 2), "ic": torch.zeros(2, 2), "bc": torch.zeros(4, 2)},
    }


@pytest.mark.parametrize("kind,gamma,seed,label", [
    ("tanh", None, 0, "tanh_seed_0"), ("siren", None, 2, "siren_seed_2"),
    ("gaussian", 0.0, 1, "gaussian_gamma_0.0_seed_1"),
    ("gaussian", 0.001, 9, "gaussian_gamma_0.001_seed_9"),
])
def test_run_label_preserves_existing_names(kind, gamma, seed, label):
    from pinn_activation_comparison.checkpoints import run_label

    assert run_label(kind, gamma, seed) == label


def test_no_resume_and_missing_directory_run_start_fresh(tmp_path):
    from pinn_activation_comparison.checkpoints import load_resume

    assert load_resume(ExperimentConfig(), "siren", None, 0) == (None, None)
    assert load_resume(ExperimentConfig(resume=str(tmp_path)), "siren", None, 0) == (None, None)


@pytest.mark.parametrize("relative", ["checkpoint.pt", "checkpoints/latest.pt"])
def test_load_resume_finds_final_or_interrupted_directory_checkpoint(tmp_path, relative):
    from pinn_activation_comparison.checkpoints import load_resume

    source_config = ExperimentConfig(models=("siren",), epochs=4)
    path = tmp_path / "runs" / "siren_seed_0" / relative
    path.parent.mkdir(parents=True)
    torch.save(_payload(source_config), path)
    loaded, selected = load_resume(replace(source_config, resume=str(tmp_path), epochs=8), "siren", None, 0)
    assert selected == path
    assert loaded["run_state"]["adam_epoch"] == 2
    torch.testing.assert_close(loaded["model_state_dict"]["weight"], torch.tensor([1.0]))


def test_directory_resume_prefers_final_checkpoint_over_latest(tmp_path):
    from pinn_activation_comparison.checkpoints import load_resume

    config = ExperimentConfig(models=("siren",), epochs=4)
    run = tmp_path / "runs" / "siren_seed_0"
    (run / "checkpoints").mkdir(parents=True)
    torch.save(_payload(config, adam_epoch=1), run / "checkpoints" / "latest.pt")
    torch.save(_payload(config, adam_epoch=4), run / "checkpoint.pt")
    loaded, selected = load_resume(replace(config, resume=str(tmp_path)), "siren", None, 0)
    assert selected == run / "checkpoint.pt"
    assert loaded["run_state"]["adam_epoch"] == 4


def test_explicit_checkpoint_requires_matching_condition(tmp_path):
    from pinn_activation_comparison.checkpoints import load_resume

    config = ExperimentConfig(models=("siren",), epochs=4)
    path = tmp_path / "checkpoint.pt"
    torch.save(_payload(config), path)
    resumed = replace(config, resume=str(path))
    loaded, selected = load_resume(resumed, "siren", None, 0)
    assert loaded["condition"]["model"] == "siren"
    assert selected == path
    with pytest.raises(ValueError, match="condition"):
        load_resume(resumed, "tanh", None, 0)


def test_missing_explicit_checkpoint_does_not_silently_start_over(tmp_path):
    from pinn_activation_comparison.checkpoints import load_resume

    with pytest.raises((FileNotFoundError, ValueError)):
        load_resume(ExperimentConfig(resume=str(tmp_path / "missing.pt")), "siren", None, 0)


def test_legacy_weight_only_checkpoint_has_clear_rejection(tmp_path):
    from pinn_activation_comparison.checkpoints import load_resume

    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": {}, "config": ExperimentConfig().to_dict()}, path)
    with pytest.raises(ValueError, match="(?i)(format|version|weight.only)"):
        load_resume(ExperimentConfig(resume=str(path)), "siren", None, 0)


@pytest.mark.parametrize("key", [
    "model_state_dict", "adam_state_dict", "batch_rng_state", "torch_rng_state", "history",
    "evaluations", "run_state", "points", "config", "model_kwargs",
])
def test_resume_rejects_incomplete_payload(key):
    from pinn_activation_comparison.checkpoints import validate_resume

    payload = _payload()
    del payload[key]
    with pytest.raises(ValueError, match=key):
        validate_resume(payload, ExperimentConfig(models=("siren",), epochs=4))


@pytest.mark.parametrize("status", ["nonfinite", "diverged"])
def test_failed_checkpoints_are_inspectable_but_cannot_resume(status):
    from pinn_activation_comparison.checkpoints import validate_resume

    payload = _payload()
    payload["status"] = status
    with pytest.raises(ValueError, match=status):
        validate_resume(payload, ExperimentConfig(models=("siren",), epochs=4))


@pytest.mark.parametrize("changes", [
    {"lr": 0.002}, {"width": 8}, {"n_f": 33}, {"dtype": "float64"},
    {"batch_size": 0}, {"eval_every": 7}, {"checkpoint_every": 3}, {"lbfgs_lr": 0.2},
])
def test_resume_rejects_changed_training_settings(changes):
    from pinn_activation_comparison.checkpoints import validate_resume

    original = ExperimentConfig(models=("siren",), epochs=4)
    with pytest.raises(ValueError, match=next(iter(changes))):
        validate_resume(_payload(original), replace(original, **changes))


def test_resume_allows_extended_targets_and_runtime_changes():
    from pinn_activation_comparison.checkpoints import validate_resume

    original = ExperimentConfig(models=("siren",), epochs=4)
    resumed = replace(original, epochs=8, lbfgs_steps=200, output_dir="new", resume="old",
                      plots=False, device="auto", threads=2)
    validate_resume(_payload(original), resumed)


def test_resume_allows_narrowed_conditions_but_rejects_added_conditions():
    from pinn_activation_comparison.checkpoints import validate_resume

    original = ExperimentConfig(epochs=4, seeds=(0, 1, 2))
    validate_resume(_payload(original), replace(original, models=("siren",), gammas=(0.0,), seeds=(0,)))
    with pytest.raises(ValueError, match="seeds"):
        validate_resume(_payload(original), replace(original, seeds=(0, 9)))


def test_adam_can_resume_into_hybrid_before_lbfgs_has_started():
    from pinn_activation_comparison.checkpoints import validate_resume

    original = ExperimentConfig(models=("siren",), epochs=4)
    validate_resume(_payload(original), replace(original, optimizer="adam_lbfgs"))
    hybrid = replace(original, optimizer="adam_lbfgs")
    with pytest.raises(ValueError, match="optimizer"):
        validate_resume(_payload(hybrid), original)


def test_resume_rejects_target_before_completed_adam_updates():
    from pinn_activation_comparison.checkpoints import validate_resume

    original = ExperimentConfig(models=("siren",), epochs=4)
    with pytest.raises(ValueError, match="epochs"):
        validate_resume(_payload(original, adam_epoch=3), replace(original, epochs=2))


def test_lbfgs_resume_requires_state_and_preserves_adam_target():
    from pinn_activation_comparison.checkpoints import validate_resume

    original = ExperimentConfig(models=("siren",), epochs=4, optimizer="adam_lbfgs")
    payload = _payload(original, adam_epoch=4, lbfgs_step=3)
    validate_resume(payload, replace(original, lbfgs_steps=200))
    with pytest.raises(ValueError, match="epochs"):
        validate_resume(payload, replace(original, epochs=8))
    with pytest.raises(ValueError, match="lbfgs_steps"):
        validate_resume(payload, replace(original, lbfgs_steps=2))
    del payload["lbfgs_state_dict"]
    with pytest.raises(ValueError, match="lbfgs_state_dict"):
        validate_resume(payload, original)


def test_checkpoint_save_is_loadable_and_creates_parent(tmp_path):
    from pinn_activation_comparison.checkpoints import save_checkpoint

    path = tmp_path / "checkpoints" / "latest.pt"
    save_checkpoint(path, _payload())
    saved = torch.load(path, weights_only=True, map_location="cpu")
    assert saved["format_version"] == 2
    assert [item.name for item in path.parent.iterdir()] == ["latest.pt"]


def test_failed_checkpoint_write_keeps_previous_checkpoint(tmp_path, monkeypatch):
    from pinn_activation_comparison.checkpoints import save_checkpoint

    path = tmp_path / "latest.pt"
    save_checkpoint(path, _payload(adam_epoch=1))
    before = path.read_bytes()

    def failed_save(payload, destination):
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(torch, "save", failed_save)
    with pytest.raises(OSError, match="disk write"):
        save_checkpoint(path, _payload(adam_epoch=2))
    assert path.read_bytes() == before
    assert [item.name for item in tmp_path.iterdir()] == ["latest.pt"]


@pytest.mark.parametrize("nested", [False, True])
def test_copy_snapshots_preserves_only_completed_stage_steps(tmp_path, nested):
    from pinn_activation_comparison.checkpoints import copy_prior_snapshots

    source = tmp_path / "original"
    snapshot_dir = source / "snapshots"
    snapshot_dir.mkdir(parents=True)
    keep = {"adam_000000.npz", "adam_000002.npz", "center_adam_000002.png",
            "lbfgs_000001.npz", "center_lbfgs_000001.png"}
    excluded = {"adam_000003.npz", "center_adam_000003.png", "lbfgs_000002.npz", "other.npz"}
    for name in keep | excluded:
        (snapshot_dir / name).write_bytes(name.encode())
    for name in ("adam_checkpoint.pt", "adam_metrics.json", "adam_predictions.npz"):
        (source / name).write_bytes(name.encode())
    checkpoint = source / "checkpoints" / "latest.pt" if nested else source / "checkpoint.pt"
    destination = tmp_path / "resumed"
    copy_prior_snapshots(checkpoint, destination, _payload(adam_epoch=2, lbfgs_step=1))
    assert {path.name for path in (destination / "snapshots").iterdir()} == keep
    assert all((destination / name).read_bytes() == name.encode() for name in
               ("adam_checkpoint.pt", "adam_metrics.json", "adam_predictions.npz"))


def test_adam_resume_does_not_copy_future_lbfgs_or_stale_endpoint(tmp_path):
    from pinn_activation_comparison.checkpoints import copy_prior_snapshots

    source = tmp_path / "original"
    (source / "snapshots").mkdir(parents=True)
    (source / "snapshots" / "lbfgs_000000.npz").write_bytes(b"future")
    (source / "adam_metrics.json").write_bytes(b"stale endpoint")
    destination = tmp_path / "resumed"
    copy_prior_snapshots(source / "checkpoint.pt", destination, _payload())
    assert not (destination / "adam_metrics.json").exists()
    assert not (destination / "snapshots" / "lbfgs_000000.npz").exists()


def test_resume_source_rejects_empty_directory(tmp_path):
    from pinn_activation_comparison.checkpoints import validate_resume_source

    with pytest.raises(ValueError, match="experiment.json"):
        validate_resume_source(ExperimentConfig(resume=str(tmp_path)))


@pytest.mark.parametrize("manifest", ["broken JSON", "[]", '{"config": []}', '{}'])
def test_resume_source_requires_readable_manifest_config(tmp_path, manifest):
    from pinn_activation_comparison.checkpoints import validate_resume_source

    (tmp_path / "experiment.json").write_text(manifest)
    with pytest.raises(ValueError, match="(?i)(config|manifest|experiment.json)"):
        validate_resume_source(ExperimentConfig(resume=str(tmp_path)))


def test_resume_source_rejects_changed_learning_rate_without_any_runs(tmp_path):
    from pinn_activation_comparison.checkpoints import validate_resume_source

    original = ExperimentConfig()
    (tmp_path / "experiment.json").write_text(json.dumps({"config": original.to_dict()}))
    with pytest.raises(ValueError, match="lr"):
        validate_resume_source(replace(original, resume=str(tmp_path), lr=0.002))


def test_valid_unstarted_resume_manifest_accepts_narrowed_conditions(tmp_path):
    from pinn_activation_comparison.checkpoints import validate_resume_source

    original = ExperimentConfig(seeds=(0, 1, 2))
    (tmp_path / "experiment.json").write_text(json.dumps({"config": original.to_dict()}))
    resumed = replace(original, resume=str(tmp_path), epochs=200, models=("siren",),
                      gammas=(0.0,), seeds=(1,), output_dir="new")
    validate_resume_source(resumed)
    with pytest.raises(ValueError, match="seeds"):
        validate_resume_source(replace(resumed, seeds=(9,)))


def test_resume_source_defers_checkpoint_files_to_run_loader(tmp_path):
    from pinn_activation_comparison.checkpoints import validate_resume_source

    validate_resume_source(ExperimentConfig())
    validate_resume_source(ExperimentConfig(resume=str(tmp_path / "checkpoint.pt")))


@pytest.mark.parametrize("saved_gamma,resume_gamma", [(0, 0.0), (0.0, 0)])
def test_directory_resume_matches_numeric_gamma_regardless_of_label_spelling(tmp_path, saved_gamma, resume_gamma):
    from pinn_activation_comparison.checkpoints import load_resume, run_label

    original = ExperimentConfig(models=("gaussian",), gammas=(saved_gamma,), epochs=4)
    source = tmp_path / "runs" / run_label("gaussian", saved_gamma, 0) / "checkpoint.pt"
    source.parent.mkdir(parents=True)
    torch.save(_payload(original, kind="gaussian", gamma=saved_gamma), source)
    resumed = replace(original, resume=str(tmp_path), gammas=(resume_gamma,))
    loaded, selected = load_resume(resumed, "gaussian", resume_gamma, 0)
    assert selected == source
    assert loaded["run_state"]["adam_epoch"] == 2


def test_empty_canonical_gamma_directory_does_not_hide_equivalent_saved_run(tmp_path):
    from pinn_activation_comparison.checkpoints import load_resume

    original = ExperimentConfig(models=("gaussian",), gammas=(0,), epochs=4)
    source = tmp_path / "runs" / "gaussian_gamma_0_seed_0" / "checkpoint.pt"
    source.parent.mkdir(parents=True)
    torch.save(_payload(original, kind="gaussian", gamma=0), source)
    (tmp_path / "runs" / "gaussian_gamma_0.0_seed_0").mkdir()
    loaded, selected = load_resume(replace(original, resume=str(tmp_path), gammas=(0.0,)), "gaussian", 0.0, 0)
    assert selected == source
    assert loaded is not None


def test_canonical_gamma_checkpoint_is_preferred_without_loading_alternate_files(tmp_path):
    from pinn_activation_comparison.checkpoints import load_resume

    original = ExperimentConfig(models=("gaussian",), gammas=(0.0,), epochs=4)
    source = tmp_path / "runs" / "gaussian_gamma_0.0_seed_0" / "checkpoint.pt"
    source.parent.mkdir(parents=True)
    torch.save(_payload(original, kind="gaussian", gamma=0.0), source)
    alternate = tmp_path / "runs" / "gaussian_gamma_0_seed_0" / "checkpoint.pt"
    alternate.parent.mkdir()
    alternate.write_bytes(b"not a checkpoint")
    loaded, selected = load_resume(replace(original, resume=str(tmp_path)), "gaussian", 0.0, 0)
    assert selected == source
    assert loaded is not None


def test_ambiguous_numeric_gamma_alternatives_are_rejected(tmp_path):
    from pinn_activation_comparison.checkpoints import load_resume

    original = ExperimentConfig(models=("gaussian",), gammas=(0.0,), epochs=4)
    for gamma_label in ("0", "0.00"):
        source = tmp_path / "runs" / f"gaussian_gamma_{gamma_label}_seed_0" / "checkpoint.pt"
        source.parent.mkdir(parents=True)
        torch.save(_payload(original, kind="gaussian", gamma=0.0), source)
    with pytest.raises(ValueError, match="(?i)(ambiguous|multiple)"):
        load_resume(replace(original, resume=str(tmp_path)), "gaussian", 0.0, 0)


def test_interrupted_latest_save_recovers_most_recent_named_checkpoint(tmp_path):
    from pinn_activation_comparison.checkpoints import load_resume

    original = ExperimentConfig(models=("siren",), epochs=4, optimizer="adam_lbfgs")
    checkpoints = tmp_path / "runs" / "siren_seed_0" / "checkpoints"
    checkpoints.mkdir(parents=True)
    for name, epoch, step in (("adam_000004.pt", 4, 0), ("lbfgs_000001.pt", 4, 1), ("lbfgs_000002.pt", 4, 2)):
        torch.save(_payload(original, adam_epoch=epoch, lbfgs_step=step), checkpoints / name)
    loaded, selected = load_resume(replace(original, resume=str(tmp_path)), "siren", None, 0)
    assert selected == checkpoints / "lbfgs_000002.pt"
    assert loaded["run_state"]["lbfgs_step"] == 2


@pytest.mark.parametrize("name,contents", [
    ("metrics.json", '{"epochs_completed": 2, "lbfgs_steps_completed": 0}'),
    ("history.csv", "epoch,stage,stage_step,step_status\n1,adam,1,completed\n"),
    ("evaluations.csv", "epoch,stage,stage_step\n2,adam,2\n"),
])
def test_recorded_accepted_progress_without_checkpoint_is_not_silently_restarted(tmp_path, name, contents):
    from pinn_activation_comparison.checkpoints import load_resume

    run = tmp_path / "runs" / "gaussian_gamma_0_seed_0"
    run.mkdir(parents=True)
    (run / name).write_text(contents)
    config = ExperimentConfig(models=("gaussian",), gammas=(0.0,), resume=str(tmp_path))
    with pytest.raises(ValueError, match="(?i)checkpoint"):
        load_resume(config, "gaussian", 0.0, 0)


def test_config_and_initial_evaluation_without_accepted_updates_can_start_fresh(tmp_path):
    from pinn_activation_comparison.checkpoints import load_resume

    run = tmp_path / "runs" / "gaussian_gamma_0_seed_0"
    run.mkdir(parents=True)
    (run / "config.json").write_text('{"model": "gaussian", "gamma": 0, "seed": 0}')
    (run / "evaluations.csv").write_text("epoch,stage,stage_step\n0,adam,0\n")
    config = ExperimentConfig(models=("gaussian",), gammas=(0.0,), resume=str(tmp_path))
    assert load_resume(config, "gaussian", 0.0, 0) == (None, None)
