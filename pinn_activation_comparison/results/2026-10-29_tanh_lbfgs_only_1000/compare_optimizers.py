"""Compare the saved pure L-BFGS experiment against the fixed tanh reference.

Evaluation and reporting only. This script never changes reference artifacts or
trains a model. Run after the final experiment and its 401x401 diagnostics finish.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import torch

PROJECT = next(p for p in Path(__file__).resolve().parents if (p / 'pinn_activation_comparison/train.py').is_file())
sys.path.insert(0, str(PROJECT))
from pinn_activation_comparison.models import PINN
from pinn_activation_comparison.physics import breather, pde_residual
from pinn_activation_comparison.reporting import write_json, write_csv
from pinn_activation_comparison.plot_results import plt


def read_json(path):
    """Read one saved experiment object."""
    return json.loads(path.read_text())


def main():
    """Evaluate unseen points, prove matching setup and plot both trajectories."""
    output = Path(__file__).resolve().parent
    results = output.parent
    run = output / 'runs/tanh_seed_0'
    reference = results / 'tanh_reference'
    prefix = '2026-10-29_tanh_lbfgs_only_500'
    first = results / prefix / 'runs/tanh_seed_0'
    checkpoint = torch.load(run / 'checkpoint.pt', weights_only=True, map_location='cpu')
    initial = torch.load(first / 'checkpoints/lbfgs_000000.pt', weights_only=True, map_location='cpu')
    control_initial = torch.load(results / 'full_5000/runs/tanh_seed_0/checkpoints/adam_000000.pt', weights_only=True, map_location='cpu')
    before = torch.load(first / 'checkpoint.pt', weights_only=True, map_location='cpu')
    resumed = torch.load(run / 'checkpoints/lbfgs_000500.pt', weights_only=True, map_location='cpu')
    for key, value in control_initial['model_state_dict'].items():
        assert torch.equal(value, initial['model_state_dict'][key])
    for key in ('f', 'ic', 'bc'):
        assert torch.equal(control_initial['points'][key], checkpoint['points'][key])
    for state in (initial, before, resumed, checkpoint):
        assert state['adam_state_dict'] is None and state['run_state']['adam_epoch'] == 0
        assert state['run_state']['adam_train_seconds'] == 0
        assert all(row['stage'] == 'lbfgs' for row in state['history'])
    def same(a, b):
        """Check all restored curvature tensors and scalar history exactly."""
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                same(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                same(x, y)
        else:
            assert a == b
    same(before['model_state_dict'], resumed['model_state_dict'])
    same(before['lbfgs_state_dict']['state'], resumed['lbfgs_state_dict']['state'])
    assert resumed['lbfgs_state_dict']['param_groups'][0]['tolerance_change'] == 1e-12
    model = PINN(**checkpoint['model_kwargs']).to(dtype=getattr(torch, checkpoint['config']['dtype']))
    model.load_state_dict(checkpoint['model_state_dict'])
    torch.set_num_threads(1)
    model.eval()
    coordinates = np.random.default_rng(20261028).uniform([-10., 0.], [10., 20.], size=(10000, 2))
    error = denominator = residual_sum = 0.
    for start in range(0, len(coordinates), 512):
        points = torch.as_tensor(coordinates[start:start + 512], dtype=next(model.parameters()).dtype)
        with torch.no_grad():
            predicted = model(points).numpy().astype(float)
            exact = breather(points).numpy().astype(float)
        error += np.square(predicted - exact).sum()
        denominator += np.square(exact).sum()
        residual_sum += np.square(pde_residual(model, points).detach().numpy().astype(float)).sum()
    random_metrics = dict(seed=20261028, count=10000, relative_l2=float(np.sqrt(error / denominator)),
                          residual_rms=float(np.sqrt(residual_sum / 10000)))
    write_json(run / 'validation_random.json', random_metrics)
    variants = [('L-BFGS only', run), ('Adam + L-BFGS', reference)]
    rows = []
    for label, folder in variants:
        metrics = read_json(folder / 'metrics.json')
        physical = read_json(folder / 'physical_diagnostics.json')
        random = random_metrics if folder == run else read_json(folder / 'validation_random.json')
        rows.append(dict(protocol=label, **{key:metrics.get(key) for key in
                    ('status','epochs_completed','lbfgs_steps_completed','lbfgs_closure_evaluations',
                     'total_loss','pde_loss','ic_loss','ic_velocity_loss','bc_loss','train_seconds',
                     'adam_train_seconds','lbfgs_train_seconds','peak_rss_mb')},
                    relative_l2_201=metrics['relative_l2'], relative_l2_401=physical['relative_l2'],
                    relative_l2_random=random['relative_l2'], residual_rms_401=physical['residual_rms'],
                    ic_velocity_relative_l2=physical['ic_velocity_relative_l2'],
                    center_relative_l2=physical['center_relative_l2'], period=physical['period_prediction'],
                    amplitude_ratio=physical['center_amplitude_ratio'], energy_ratio=physical['energy_final_ratio']))
    write_csv(output / 'comparison.csv', rows)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout='constrained')
    for label, folder in variants:
        with (folder / 'evaluations.csv').open() as stream:
            data = [row for row in csv.DictReader(stream) if row['stage'] == 'lbfgs']
        x = [int(row['stage_step']) for row in data]
        axes[0].semilogy(x, [float(row['relative_l2']) for row in data], label=label)
        axes[1].semilogy(x, [float(row['total_loss']) for row in data], label=label)
    axes[0].axhline(.01, color='black', linestyle=':', label='1% error target')
    for ax in axes:
        ax.axvline(500, color='grey', linestyle='--', label='tolerance 1e-9 → 1e-12')
        ax.set(xlabel='L-BFGS outer step')
        ax.legend()
        ax.grid(alpha=.2)
    axes[0].set(ylabel='Relative L2 on 201x201 grid')
    axes[1].set(ylabel='Post-update full-pool total loss')
    fig.suptitle('Same initial draw, point pools and L-BFGS schedule; reference includes 10000 preceding Adam updates')
    fig.savefig(output / 'figures/optimizer_comparison.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4.5), layout='constrained')
    for index, (label, folder) in enumerate(variants):
        with np.load(folder / 'physical_diagnostics.npz') as arrays:
            if index == 0:
                ax.plot(arrays['center_fine_t'], arrays['center_fine_exact'], 'k', linewidth=2.5, label='Exact')
            ax.plot(arrays['center_fine_t'], arrays['center_fine_prediction'], label=label)
    ax.set(xlabel='t', ylabel='phi(0,t)', title='Final center oscillation')
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(output / 'figures/center_comparison.png', dpi=160)
    plt.close(fig)
    audit = output / 'previous_results_manifest.json'
    old_files = json.loads(audit.read_text()) if audit.exists() else None
    unchanged = None
    if old_files is not None:
        for item in old_files:
            assert hashlib.sha256((results / item['path']).read_bytes()).hexdigest() == item['sha256'],item['path']
        unchanged = len(old_files)
    write_json(output / 'verification.json', dict(initial_weights_match_reference=True, fixed_pools_match_reference=True,
        no_adam_state_or_updates=True, resume_curvature_preserved=True, previous_result_files_unchanged=unchanged,
        actual_lbfgs_internal_iteration_counter=next(iter(checkpoint['lbfgs_state_dict']['state'].values()))['n_iter']))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
