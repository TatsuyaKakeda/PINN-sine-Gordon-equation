# Full training, monitoring, and continuation

The existing 100 epoch results remain a Quick smoke experiment. Preserve the
physics, architectures, common samples, and Adam update rule. No Full training
is automatically launched as part of the original implementation request.
The follow-up request on 2026-10-23 explicitly authorizes the 5,000 epoch
comparison: after integration verification, run all six activation/gamma
conditions with seeds 0, 1, 2 and report convergence and stability.

1. Set Full defaults to 5,000 Adam updates, three seeds, evaluation and checkpoint
   intervals of 500; expose optional L-BFGS and continuation settings.
2. Add held-out initial velocity diagnostics and periodic full-pool evaluations,
   center traces, and optimizer-stage-aware plots.
3. Save optimizer states, random states, authoritative samples, counters and
   histories atomically. Resume to a new directory with immutable experimental
   conditions and an increased total update budget.
4. Integrate chunked full-batch L-BFGS after Adam, preserving the Adam endpoint
   and reporting outer steps, closure counts and timing separately.
5. Test uninterrupted versus resumed trajectories, failure handling, saved
   artifacts and backward compatibility. Run short three-seed smoke comparisons,
   inspect figures, and document commands and scientific limitations.

Ownership: configuration/checkpoint helpers, evaluation/L-BFGS, and plotting/docs
are independent delegated tasks; training integration and end-to-end verification
are performed by the primary agent.

## Completion evidence (2026-10-23)

- All five implementation steps completed; 230 tests passed with warnings as errors.
- Full CPU experiment attempted 18 conditions: 12 reached 5,000 Adam updates,
  six stopped at the configured gradient threshold; no NaN/Inf stops.
- Saved predictions reproduce reported errors; common point pools and exact
  SIREN / gamma-zero trajectories verified for all three seeds.
- Periodic artifacts, 69 comparison figures and 156 center snapshot figures saved.
- Results and interpretation: `pinn_activation_comparison/results/full_5000/FULL_RESULTS.md`.
- Original Quick artifacts and runtime source snapshots retained. The later
  numerical-gamma resume lookup fix is separately archived in `post_run_source/`.
