# Current configuration refactor status

This is the current overview after Stages 1–6. Dated handoffs and benchmark
reports remain historical snapshots; their original “remaining work” lists do
not describe the current implementation.

## Maintained workflow

Select inputs and simulation settings in one TOML file:

```bash
dsh check configs/my_simulation.toml
dsh run configs/my_simulation.toml
```

The parser is host-only. Backend selection precedes numerical imports. Shared
input adapters construct numerical plans; existing transport/scoring APIs
execute them; shared staging publishes products and provenance. The completion
manifest is written last. Individual files are atomic; an interrupted group
may leave files without a valid completion marker.

- [Configuration reference](CONFIGURED_SIMULATIONS.md): scene, source, material,
  execution, observer and output sections. Seven root examples cover the six
  configured source formats and native FITS input; Stage 3 adds uniform discs,
  selected image energy/cropping and representative flux tables.
- [Runner compatibility](RUNNER_MIGRATION_STAGE3.md): root Python settings,
  `dsh-v1` and its script shim use the shared plan/execution/publication path.
- [Slurm](SLURM_STAGE4.md): optional `[slurm]` settings, `dsh slurm-plan` and
  `dsh slurm-submit`. Portable planning passed; actual allocation/GPU execution
  is still pending cluster testing.
- [Physical responses](RESPONSE_STAGE5.md): `response_collect` and
  `response_fold` TOML workflows preserve existing archive schemas and
  photon-grouped Monte Carlo uncertainties. These are dust halo responses;
  instrumental response modeling remains separate.
- [Cleanup/profiling audit](CLEANUP_PROFILING_STAGE6.md): shared publication,
  retained APIs, profile methodology, measured integration workloads and limits.

The numerical kernels, material producers, observer estimators, source
normalization and frozen benchmark evidence remain unchanged by this cleanup.
Historical standalone tools remain available where compatibility or scientific
reproducibility requires them. No broad deletion or performance rewrite was
made on the basis of a static unused-name scan.

## Profiling

```bash
python -m dsh.profiling configs/my_simulation.toml --output-directory outputs/profile-new --repeats 3
```

The profile directory must be new and outside the configured production output
directory. The tool executes a validated plan but does not export production
products. It writes `profile.json`, `calls.txt` and `calls.pstats`, reports first
and repeated execution separately, and verifies exact seeded scientific output
on every repetition. Source/material/input paths still come from the TOML file;
`--repeats` controls the measurement, not the simulation settings.

Use a configuration with unused output destinations (the normal parser's
checks apply). For meaningful production tuning, profile your actual scene,
packet/chunk budget, energy support and backend. The small acceptance workloads
establish reproducibility and integration, not convergence or GPU scaling.

## Remaining work

Actual Slurm allocation and GPU validation are deferred at the user's request.
Real-source production configurations still require the selected cloud cube,
source history, geometry and a convergence study. Workload-specific performance
changes, peak-memory measurements and export/plot profiling need separate
measurements and validation. The standalone variable-power-law numerical API
and historical scripts have been retained; future removal requires an explicit
compatibility/deprecation decision. New material prescriptions and additional
scientific validation are separate from this configuration refactor.
