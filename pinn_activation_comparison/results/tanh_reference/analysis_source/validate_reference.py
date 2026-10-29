"""Validate and package the selected tanh trajectory; never trains a model.

Run from the project root. Existing reference bundles are preserved; choose a
new --bundle directory to repeat this validation and packaging.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import sys

import numpy as np
import torch

PROJECT = next(parent for parent in Path(__file__).resolve().parents
               if (parent / 'pinn_activation_comparison/train.py').is_file())
sys.path.insert(0, str(PROJECT))
from pinn_activation_comparison.models import PINN
from pinn_activation_comparison.physics import breather, pde_residual
from pinn_activation_comparison.reporting import write_json, write_csv
from pinn_activation_comparison.plot_results import plt


def sha256(path):
    """Hash immutable weights or source artifacts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Check unseen random points and create a small reusable reference bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=PROJECT / 'pinn_activation_comparison/results/tanh_refined')
    parser.add_argument('--bundle', type=Path, default=PROJECT / 'pinn_activation_comparison/results/tanh_reference')
    args = parser.parse_args()
    root, bundle = args.source.resolve(), args.bundle.resolve()
    if bundle.exists() and any(bundle.iterdir()):
        raise FileExistsError(f'Choose an empty bundle directory: {bundle}')
    run = root / 'runs/tanh_seed_0'
    checkpoint_path = run / 'checkpoint.pt'
    original_hash = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    assert checkpoint['status'] == 'completed'
    assert checkpoint['condition'] == dict(model='tanh', gamma=None, seed=0)
    torch.set_num_threads(1)
    model = PINN(**checkpoint['model_kwargs']).to(dtype=getattr(torch, checkpoint['config']['dtype']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    rng = np.random.default_rng(20261028)
    coordinates = rng.uniform([-10., 0.], [10., 20.], size=(10000, 2))
    squared_error = squared_exact = squared_residual = 0.
    for start in range(0, len(coordinates), 512):
        points = torch.as_tensor(coordinates[start:start + 512], dtype=next(model.parameters()).dtype)
        with torch.no_grad():
            prediction = model(points).numpy().astype(float)
            exact = breather(points).numpy().astype(float)
        squared_error += np.square(prediction - exact).sum()
        squared_exact += np.square(exact).sum()
        residual = pde_residual(model, points).detach().numpy().astype(float)
        squared_residual += np.square(residual).sum()
    random_metrics = dict(seed=20261028, count=len(coordinates),
                          relative_l2=float(np.sqrt(squared_error / squared_exact)),
                          residual_rms=float(np.sqrt(squared_residual / len(coordinates))))
    diagnostics = json.loads((run / 'physical_diagnostics.json').read_text())
    assert diagnostics['eval_nx'] == diagnostics['eval_nt'] == 401
    assert diagnostics['relative_l2'] < .01 and random_metrics['relative_l2'] < .01
    assert not diagnostics['near_zero_center']
    assert diagnostics['positive_peak_count'] == 3
    assert sha256(checkpoint_path) == original_hash
    assert not any(parameter.grad is not None for parameter in model.parameters())
    source_meta = json.loads((root / 'experiment.json').read_text())
    for name, expected in source_meta['source_sha256'].items():
        assert sha256(root / 'source' / name) == expected
    with np.load(run / 'physical_diagnostics.npz') as arrays:
        absolute = arrays['abs_error']
        index = np.unravel_index(np.argmax(absolute), absolute.shape)
        maximum_error = dict(value=float(absolute[index]), x=float(arrays['x'][index[1]]),
                             t=float(arrays['t'][index[0]]))
    bundle.mkdir(parents=True, exist_ok=True)
    for name in ('checkpoint.pt', 'metrics.json', 'config.json', 'history.csv', 'evaluations.csv',
                 'predictions.npz', 'physical_diagnostics.json', 'physical_diagnostics.npz',
                 'physical_diagnostics.png', 'optimizer_adjustments.json',
                 'adam_checkpoint.pt', 'adam_metrics.json', 'adam_predictions.npz'):
        shutil.copy2(run / name, bundle / name)
    shutil.copy2(root / 'test_results.txt', bundle / 'test_results.txt')
    shutil.copytree(root / 'source', bundle / 'source')
    (bundle / 'analysis_source').mkdir()
    shutil.copy2(__file__, bundle / 'analysis_source/validate_reference.py')
    shutil.copy2(PROJECT / 'pinn_activation_comparison/tanh_diagnostics.py', bundle / 'analysis_source/tanh_diagnostics.py')
    figures = bundle / 'figures'
    figures.mkdir()
    for name in ('learning_curves.png', 'error_convergence.png', 'lbfgs_learning_curves.png',
                 'lbfgs_error_convergence.png', 'gradient_norms.png', 'lbfgs_gradient_norms.png',
                 'center_trace_seed_0.png', 'spatial_profiles_seed_0.png',
                 'heatmap_tanh_seed_0.png', 'training_time_convergence.png'):
        shutil.copy2(root / 'figures' / name, figures / name)
    for stage, step in (('adam', 5000), ('adam', 10000), ('lbfgs', 100), ('lbfgs', 500), ('lbfgs', 570), ('lbfgs', 1000)):
        shutil.copy2(run / 'snapshots' / f'center_{stage}_{step:06d}.png', figures / f'center_{stage}_{step:06d}.png')
    evaluation = checkpoint['evaluations']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for col, stage in enumerate(('adam', 'lbfgs')):
        rows = [row for row in evaluation if row['stage'] == stage]
        x = [row['stage_step'] for row in rows]
        axes[0, col].semilogy(x, [row['relative_l2'] for row in rows], marker='.', label='Relative L2')
        axes[0, col].axhline(.01, color='black', linestyle=':', label='1% target')
        axes[0, col].set(ylabel='Relative L2 error', title='Adam' if stage == 'adam' else 'L-BFGS')
        for key in ('total_loss', 'pde_loss', 'ic_loss', 'bc_loss'):
            axes[1, col].semilogy(x, [row[key] for row in rows], label=key)
        axes[1, col].set(xlabel='Adam epoch' if stage == 'adam' else 'L-BFGS outer step', ylabel='Post-update full-pool loss')
        for ax in axes[:, col]:
            if stage == 'lbfgs':
                ax.axvline(500, color='grey', linestyle='--', label='tolerance 1e-9 → 1e-12')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle('Tanh reference: fixed architecture, point pools and loss weights; no interior solution supervision')
    fig.savefig(figures / 'optimization_progress.png', dpi=170)
    plt.close(fig)
    stages = []
    for stage, directory in [('Adam 5000', 'full_5000'), ('Adam 10000', 'tanh_10000'),
                             ('L-BFGS 100', 'tanh_10000_lbfgs'), ('L-BFGS 500', 'tanh_10000_lbfgs500'),
                             ('L-BFGS 1000, tighter tolerance', 'tanh_refined')]:
        row = json.loads((root.parent / directory / 'runs/tanh_seed_0/metrics.json').read_text())
        stages.append(dict(stage=stage, **{key: row[key] for key in ('epochs_completed', 'lbfgs_steps_completed',
                           'relative_l2', 'total_loss', 'pde_loss', 'ic_loss', 'bc_loss', 'train_seconds')}))
    write_csv(bundle / 'stages.csv', stages)
    write_json(bundle / 'validation_random.json', random_metrics)
    write_json(run / 'validation_random.json', random_metrics)
    write_json(bundle / 'reference.json', dict(source_run=str(run), checkpoint_sha256=original_hash,
               criterion=dict(relative_l2_below=.01, dense_grid='401x401', unseen_random_points=10000),
               seed_selection='Seed 0 declared before L-BFGS refinements; other seeds not used to select this reference.',
               max_absolute_error=maximum_error, dense_validation=diagnostics, random_validation=random_metrics,
               runtime_source_sha256=source_meta['source_sha256'],
               training_environment=source_meta['environment'],
               analysis_source_sha256={p.name: sha256(p) for p in (bundle/'analysis_source').glob('*.py')},
               optimizer_state_preserved=True, fixed_training_points_verified=True,
               memory_note='peak_rss_mb in metrics is process peak at training end, before final comparison plots.'))
    print(json.dumps(dict(bundle=str(bundle), dense_relative_l2=diagnostics['relative_l2'],
                          random_validation=random_metrics, max_absolute_error=maximum_error), indent=2))


if __name__ == '__main__':
    main()
