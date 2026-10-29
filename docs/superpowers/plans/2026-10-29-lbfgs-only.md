# Tanh L-BFGS-only comparison

**Goal:** Train the same tanh network from random initialization for 1000
L-BFGS outer calls, with no Adam updates or pretrained weights.

**Protocol:** Seed 0; original 4x64 architecture and normalization; identical
4096/256/256-per-side fixed coordinate pools; all loss weights 1; float32 CPU,
one thread. Match the reference L-BFGS schedule: tolerance_change=1e-9 for the
first 500 outer calls, then 1e-12 for calls 501--1000, retaining curvature state.
Use max_iter=20, history_size=50, lr=1 and strong-Wolfe. Distinguish outer calls,
internal iterations and closure counts. Exact interior values are evaluation-only.

- [x] Test/configure optimizer='lbfgs' with zero Adam budget; CLI defaults the
  Adam budget to zero for this protocol and rejects ambiguous positive budgets.
- [x] Reuse the existing training loop and checkpoint format. Never instantiate
  Adam in the pure CPU route; store adam_state_dict=None and no Adam endpoint.
  Verify exact continuation and preserve existing Adam/hybrid behavior.
- [x] Plot only stages actually present; label the new protocol explicitly.
- [x] Run 500 calls in a new directory, then resume to cumulative 1000 in another
  new directory with only the tolerance changed. Preserve all existing results.
- [x] Evaluate on 201x201, independent 401x401 and unseen random points; inspect
  center oscillation, spatial profiles, residual and IC error. Compare to the
  reference with the same stop schedule and actual training time.
- [x] Save report/figures, update RESULTS and configuration documentation,
  verify initialization/pool equality and run relevant regression tests.

## Completion evidence

- Adam state None and zero Adam updates/time; initial weights/pools match exactly.
- 1000 L-BFGS outer calls; 11581 internal iterations; 14650 closure evaluations.
- Relative L2 = 0.00821649 on 401x401, 0.00831099 on 10000 held-out random points.
- Total update time 386.11s vs reference314.73s; one seed, matched stop schedule.
- 255 tests passed; all 2102 pre-existing result files unchanged.
- Entry: `pinn_activation_comparison/results/2026-10-29_tanh_lbfgs_only_1000/README.md`.
