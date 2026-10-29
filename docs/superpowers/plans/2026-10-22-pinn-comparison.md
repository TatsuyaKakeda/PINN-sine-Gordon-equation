# sine-Gordon activation comparison implementation plan

**Goal:** Implement the user's specified controlled six-condition PINN comparison, verify equations and derivatives, and run a CPU Quick experiment. Full training is explicitly excluded from automatic execution.

**Architecture:** Physical-coordinate PyTorch networks with internal normalization; independent physics, sampling, configuration, training, evaluation, and matplotlib plotting modules. Exact interior values are evaluation-only. Each seed shares collocation points and minibatch sequences; SIREN and Gaussian variants share initial parameter draws. Final metrics use the final checkpoint and full training pool; test-grid errors are diagnostic only.

**Tech stack:** Python >=3.10, PyTorch, NumPy, matplotlib, pytest. An isolated local virtual environment supplies dependencies.

- [x] Physics/models: test exact residual and initial velocity in float64, gamma-zero identity through two derivatives, physical normalization chain rule, and parameter backward propagation; implement SIREN reference initialization.
- [x] Runner: deterministic separate RNG streams for samples, batches and model initialization; JSON/CLI config with quick/full presets; Adam and loss terms, per-layer/global gradient logging, finite/divergence diagnostics, device probe and CPU restart fallback.
- [x] Evaluation: chunked held-out grid residuals, relative L2, center and spatial snapshots, final training losses, checkpoints, strict JSON/CSV/NPZ, sample mean/std across seeds and explicit failed-run accounting.
- [x] Figures: activation and two derivatives, losses, gradients, errors, shared-scale heatmaps, center/profile plots, gamma comparisons. Verify generated figures visually.
- [x] Documentation: Japanese setup/CLI/reference/method notes, controlled vs tuned distinction, limits of Quick and of integrable breathers for oscillon research.
- [x] Integration: run all pytest tests and default Quick comparison for three seeds; report observed accuracy without claiming convergence or superiority. Preserve actual result files and environment versions.

Verification commands: `.venv/bin/python -m pytest pinn_activation_comparison/tests -q`; `.venv/bin/python -m pinn_activation_comparison.train --mode quick --seed 0 1 2 --output-dir pinn_activation_comparison/results/quick_verified`.

The detailed user specification authorizes implementation and fixes. No additional design approval or git commit is needed; this workspace is not a Git repository.
