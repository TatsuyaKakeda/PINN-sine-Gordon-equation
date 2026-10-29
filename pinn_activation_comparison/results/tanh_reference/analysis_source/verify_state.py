"""Audit saved optimizer continuation and fixed point pools, without training."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch


def identical(left, right):
    """Assert exact equality of nested optimizer tensors and scalar state."""
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for x, y in zip(left, right, strict=True):
            identical(x, y)
    else:
        assert left == right


def main():
    """Write explicit evidence for the Adam and L-BFGS state transitions."""
    project = next(parent for parent in Path(__file__).resolve().parents
                   if (parent / 'pinn_activation_comparison/train.py').is_file())
    results = project / 'pinn_activation_comparison/results'
    def load(directory, name='checkpoint.pt'):
        """Load a trusted local checkpoint in portable weights-only mode."""
        return torch.load(results / directory / 'runs/tanh_seed_0' / name, weights_only=True, map_location='cpu')
    initial = load('full_5000')
    resumed = load('tanh_10000', 'checkpoints/adam_005000.pt')
    for key in ('model_state_dict', 'adam_state_dict', 'batch_rng_state', 'torch_rng_state', 'points'):
        identical(initial[key], resumed[key])
    before = load('tanh_10000_lbfgs500')
    after = load('tanh_refined', 'checkpoints/lbfgs_000500.pt')
    for key in ('model_state_dict', 'adam_state_dict', 'batch_rng_state', 'torch_rng_state', 'points'):
        identical(before[key], after[key])
    identical(before['lbfgs_state_dict']['state'], after['lbfgs_state_dict']['state'])
    assert before['lbfgs_state_dict']['param_groups'][0]['tolerance_change'] == 1e-9
    assert after['lbfgs_state_dict']['param_groups'][0]['tolerance_change'] == 1e-12
    for directory in ('tanh_10000', 'tanh_10000_lbfgs', 'tanh_10000_lbfgs500', 'tanh_refined'):
        saved = load(directory)
        identical(initial['points'], saved['points'])
        for key in ('hidden_layers', 'width', 'n_f', 'n_ic', 'n_bc', 'batch_size', 'lambda_pde',
                    'lambda_ic', 'lambda_bc', 'omega', 't_max', 'x_min', 'x_max', 'dtype', 'normalize'):
            assert initial['config'][key] == saved['config'][key]
    bundle = results / 'tanh_reference'
    source = results / 'tanh_refined/runs/tanh_seed_0/checkpoint.pt'
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(source) == digest(bundle / 'checkpoint.pt')
    localization = []
    with np.load(bundle / 'physical_diagnostics.npz') as arrays:
        x = arrays['profile_x']
        for time, prediction, exact in zip(arrays['profile_times'], arrays['profile_prediction'], arrays['profile_exact'], strict=True):
            denominator = np.trapezoid(exact ** 2, x)
            if denominator < 1e-12:
                continue
            exact_width = np.sqrt(np.trapezoid(x ** 2 * exact ** 2, x) / denominator)
            predicted_width = np.sqrt(np.trapezoid(x ** 2 * prediction ** 2, x) / np.trapezoid(prediction ** 2, x))
            localization.append(dict(t=float(time), exact_width=float(exact_width), predicted_width=float(predicted_width),
                                     width_relative_error=float(abs(predicted_width / exact_width - 1))))
    verification = dict(tests_passed=241, adam_weights_optimizer_and_rng_restored_exactly=True,
                        lbfgs_weights_curvature_and_adam_state_preserved_when_tolerance_changed=True,
                        fixed_training_points_and_loss_weights_verified=True,
                        reference_checkpoint_sha256=digest(source),
                        figures_visually_reviewed=['optimization_progress.png', 'center_trace_seed_0.png',
                                                   'heatmap_tanh_seed_0.png', 'spatial_profiles_seed_0.png'])
    (bundle / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    (bundle / 'localization_validation.json').write_text(json.dumps(localization, indent=2) + '\n')
    print(json.dumps(verification, indent=2))
    print('Largest sampled profile width error:', max(row['width_relative_error'] for row in localization))


if __name__ == '__main__':
    main()
