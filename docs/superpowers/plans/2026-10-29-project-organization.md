# Project navigation and experiment catalog plan

**Goal:** Make source code and existing experiments easy to locate without
moving or changing saved results, checkpoints, configurations or source archives.

**Design:** Retain the importable `pinn_activation_comparison/` package as the
implementation location. Add a short root README and a RESULTS index, three
configuration examples, and a read-only catalog command. Extend plotting with
an optional empty output directory while preserving its existing default.

- [x] Inspect the project, experiment metadata, snapshots, logs and metrics.
  Preserve a SHA-256 inventory of all existing research files outside caches.
- [x] Test a standard-library catalog against missing metadata, partial runs,
  reference bundles, source mismatches, exact duplicates and overwrite refusal.
  Implement `scripts/catalog_results.py` with read-only listing and explicit
  exports to a new directory; never import training code.
- [x] Test and add `plot_results --output-dir NEW_DIRECTORY`; default behavior
  stays compatible and custom output refuses nonempty directories.
- [x] Add valid Quick/Full/continuation JSON examples under `configs/`.
- [x] Write root README, RESULTS.md and docs/PROJECT_MAP.md with actual code,
  data, graph, log, checkpoint and source mappings; unknown values stay unknown.
- [x] Export run-level catalog and SHA-confirmed duplicates to docs/organization/.
- [x] Run read-only CLI checks and affected tests. Verify that every pre-existing
  research artifact is byte-identical and at its original path.

The initial snapshot contains 2,137 non-cache files, 1,149.93 MiB. Exact SHA-256
duplicates form 340 groups (1,024 files); extra copies occupy 100.10 MiB. Preserve
all copies, including latest/named checkpoints, reference bundles and code archives.
No long training or regeneration into an existing result directory is authorized.
