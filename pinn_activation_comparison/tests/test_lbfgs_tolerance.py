"""An explicit stopping-tolerance refinement must survive optimizer restore."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from pinn_activation_comparison.config import ExperimentConfig, parse_args
from pinn_activation_comparison.models import PINN
from pinn_activation_comparison.optimization import make_lbfgs


def test_lbfgs_stopping_tolerance_is_configurable_from_cli():
    cfg = parse_args(['--lbfgs-tolerance-change', '1e-12'])
    assert cfg.lbfgs_tolerance_change == 1e-12


@pytest.mark.parametrize('value', [0, -1e-9, float('nan'), float('inf'), True])
def test_invalid_stopping_tolerance_is_rejected(value):
    with pytest.raises(ValueError, match='lbfgs_tolerance_change'):
        ExperimentConfig(lbfgs_tolerance_change=value)


def test_tolerance_change_preserves_history_and_unblocks_small_updates():
    torch.manual_seed(3)
    model = PINN(hidden_layers=1, width=2).double()
    cfg = ExperimentConfig(lbfgs_tolerance_change=1e-9)
    optimizer = make_lbfgs(model, cfg)
    # A scaled objective has a nonzero gradient but g*d below the old threshold.
    parameter = next(model.parameters())
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = 1e-6 * sum(p.square().sum() for p in model.parameters())
        loss.backward()
        return loss
    before = parameter.detach().clone()
    optimizer.step(closure)
    assert torch.equal(parameter, before)
    saved = deepcopy(optimizer.state_dict())
    resumed = make_lbfgs(model, replace(cfg, lbfgs_tolerance_change=1e-15), saved)
    assert resumed.param_groups[0]['tolerance_change'] == 1e-15
    assert resumed.state_dict()['state'].keys() == saved['state'].keys()
    assert resumed.state_dict()['state'][0]['n_iter'] == saved['state'][0]['n_iter']
    optimizer = resumed
    optimizer.step(closure)
    assert not torch.equal(parameter, before)


def test_runner_records_tolerance_refinement_without_resetting_weights(tmp_path):
    import json
    from pinn_activation_comparison.train import run_experiment
    cfg = ExperimentConfig(models=('tanh',), epochs=1, hidden_layers=1, width=4,
                           n_f=7, n_ic=5, n_bc=3, batch_size=4, eval_nx=5, eval_nt=5,
                           plots=False, optimizer='adam_lbfgs', lbfgs_steps=1,
                           lbfgs_max_iter=2, output_dir=str(tmp_path / 'first'))
    source = run_experiment(cfg)
    first = torch.load(source / 'runs/tanh_seed_0/checkpoint.pt', weights_only=True)
    destination = run_experiment(replace(cfg, resume=str(source), lbfgs_tolerance_change=1e-12,
                                         output_dir=str(tmp_path / 'refined')))
    second = torch.load(destination / 'runs/tanh_seed_0/checkpoint.pt', weights_only=True)
    for key in first['model_state_dict']:
        assert torch.equal(first['model_state_dict'][key], second['model_state_dict'][key])
    assert first['lbfgs_state_dict']['state'][0]['n_iter'] == second['lbfgs_state_dict']['state'][0]['n_iter']
    assert second['lbfgs_state_dict']['param_groups'][0]['tolerance_change'] == 1e-12
    event = json.loads((destination / 'runs/tanh_seed_0/optimizer_adjustments.json').read_text())
    assert event['lbfgs_tolerance_change'] == {'old': 1e-9, 'new': 1e-12}
    assert event['history_preserved']
