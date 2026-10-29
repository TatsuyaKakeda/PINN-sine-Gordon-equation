# Tanh breather reference implementation and experiment plan

**Goal:** Improve the existing tanh PINN toward relative L2 < 0.01 on the
original sine-Gordon domain, and report any remaining error honestly.

**Architecture:** Reuse PINN, physical derivatives, losses, diagnostics, plotting,
and atomic optimizer checkpoints. First continue the saved Adam trajectories
without changing any training condition. Then isolate one optimization or
sampling intervention per new experiment. Exact interior field values remain
evaluation-only. Use sequential CPU runs, one thread, bounded derivative chunks.

**Tech stack:** Existing project Python environment, PyTorch and matplotlib.

- [x] Inspect all tanh 5000-epoch histories, predictions and optimizer states.
  All three format-2 checkpoints contain Adam state and fixed point/RNG state.
- [x] Continue all three saved tanh seeds to total 10000 Adam epochs using
  `python -m pinn_activation_comparison.train --resume
  pinn_activation_comparison/results/full_5000 --model tanh --epochs 10000
  --output-dir pinn_activation_comparison/results/tanh_10000`.
- [x] Diagnose plateau using full-pool PDE/IC/BC losses, independent residual,
  initial velocity, time windows and center oscillations. Keep seed 0 as the
  declared reference for subsequent interventions, avoiding seed selection.
- [x] From the 10000 seed-0 checkpoint, change only the optimizer to full-pool
  L-BFGS with the existing default implementation. Inspect intermediate curves.
- [x] A separate refinement runner was unnecessary. Reuse the existing runner
  with configurable L-BFGS tolerance (default unchanged), state-preserving
  restore and adjustment logs. The default g*d stopping test caused the 4.13%
  plateau; tightening only tolerance_change from 1e-9 to 1e-12 resumed updates.
  Regression tests verify that small updates resume and curvature is retained.
- [x] Causal/time-window changes were unnecessary: the unchanged global tanh
  architecture reached the requested accuracy with the optimizer refinement.
- [x] Save 500-update checkpoints, all requested figures, source/environment
  provenance, stage-specific counters and timing. Verify final error on a finer
  independent grid and audit initial conditions, residual, amplitude and period.
- [x] Run affected and regression tests; write a final experiment report and
  reproducible continuation commands, including unsuccessful interventions.

The user has authorized necessary fixes and sequential investigation. Preserve
the original controlled experiments; no SIREN/Gaussian training is performed.

## Completion evidence

- Final: Adam 10000 epochs + L-BFGS 1000 outer calls, seed 0, original 4x64 tanh.
- Relative L2: 0.00265763 on a 401x401 grid; 0.00270146 on 10000 unseen random points.
- Amplitude/period, localization, IC velocity, residual and energy checked.
- Optimizer/RNG restoration and unchanged training pools verified from files.
- 241 tests passed with warnings treated as errors; source checkpoints preserved.
- Reusable bundle/report: `pinn_activation_comparison/results/tanh_reference/`.
