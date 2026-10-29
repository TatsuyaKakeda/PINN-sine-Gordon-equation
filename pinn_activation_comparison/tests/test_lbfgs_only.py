"""The pure L-BFGS experiment must start untrained and never use Adam."""

from dataclasses import replace
import json

import pytest
import torch

from pinn_activation_comparison.config import ExperimentConfig, parse_args
from pinn_activation_comparison.train import run_experiment


def tiny(path, **changes):
    """Build an inexpensive deterministic pure-L-BFGS experiment."""
    cfg = dict(models=('tanh',), seeds=(0,), epochs=0, optimizer='lbfgs',
               hidden_layers=1, width=4, n_f=9, n_ic=5, n_bc=3,
               eval_nx=5, eval_nt=7, eval_batch_size=4, plots=False,
               lbfgs_steps=4, lbfgs_max_iter=2, lbfgs_history_size=4,
               lbfgs_eval_every=1, output_dir=str(path))
    cfg.update(changes)
    return ExperimentConfig(**cfg)


def load(root):
    """Load the one saved run without arbitrary pickle classes."""
    return torch.load(root / 'runs/tanh_seed_0/checkpoint.pt', weights_only=True, map_location='cpu')


def identical(left, right):
    """Compare nested model/optimizer state exactly."""
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            identical(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            identical(a, b)
    else:
        assert left == right


def test_cli_pure_lbfgs_has_no_implicit_adam_budget():
    cfg = parse_args(['--mode', 'full', '--model', 'tanh', '--optimizer', 'lbfgs', '--lbfgs-steps', '1000'])
    assert cfg.epochs == 0 and cfg.lbfgs_steps == 1000
    with pytest.raises((SystemExit, ValueError)):
        parse_args(['--optimizer', 'lbfgs', '--epochs', '10'])


def test_pure_lbfgs_never_constructs_adam_and_saves_only_its_stage(tmp_path, monkeypatch):
    def forbidden_adam(*args, **kwargs):
        pytest.fail('Pure L-BFGS must never construct Adam')
    monkeypatch.setattr(torch.optim, 'Adam', forbidden_adam)
    output = run_experiment(tiny(tmp_path / 'pure'))
    saved = load(output)
    assert saved['adam_state_dict'] is None
    assert saved['run_state']['adam_epoch'] == 0
    assert saved['run_state']['adam_train_seconds'] == 0
    assert saved['run_state']['lbfgs_step'] == 4
    assert {row['stage'] for row in saved['history']} == {'lbfgs'}
    assert {row['stage'] for row in saved['evaluations']} == {'lbfgs'}
    assert [row['epoch'] for row in saved['history']] == [1, 2, 3, 4]
    run = output / 'runs/tanh_seed_0'
    assert not (run / 'adam_checkpoint.pt').exists()
    assert not (output / 'summary_adam.csv').exists()
    metrics = json.loads((run / 'metrics.json').read_text())
    assert metrics['optimizer'] == 'lbfgs' and metrics['epochs_completed'] == 0
    assert metrics['lbfgs_steps_requested'] == 4


def test_pure_resume_reproduces_uninterrupted_optimizer_and_weights(tmp_path):
    cfg = tiny(tmp_path / 'full')
    full = run_experiment(cfg)
    first = run_experiment(replace(cfg, lbfgs_steps=2, output_dir=str(tmp_path / 'first')))
    continued = run_experiment(replace(cfg, resume=str(first), output_dir=str(tmp_path / 'continued')))
    for key in ('model_state_dict', 'lbfgs_state_dict', 'batch_rng_state', 'torch_rng_state', 'points'):
        identical(load(full)[key], load(continued)[key])


def test_pure_initialization_and_point_pool_match_adam_control(tmp_path):
    pure_cfg = tiny(tmp_path / 'pure', lbfgs_steps=1)
    pure = run_experiment(pure_cfg)
    adam_cfg = replace(pure_cfg, optimizer='adam', epochs=1, output_dir=str(tmp_path / 'adam'))
    adam = run_experiment(adam_cfg)
    a = torch.load(adam / 'runs/tanh_seed_0/checkpoints/adam_000000.pt', weights_only=True)
    b = torch.load(pure / 'runs/tanh_seed_0/checkpoints/lbfgs_000000.pt', weights_only=True)
    identical(a['model_state_dict'], b['model_state_dict'])
    identical(a['points'], b['points'])


def test_pure_plots_do_not_create_empty_adam_curves(tmp_path):
    output = run_experiment(tiny(tmp_path / 'plotted', lbfgs_steps=1, plots=True))
    assert (output / 'figures/lbfgs_learning_curves.png').is_file()
    assert not (output / 'figures/learning_curves.png').exists()
