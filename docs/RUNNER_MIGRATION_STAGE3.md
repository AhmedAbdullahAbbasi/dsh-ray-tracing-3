# Shared runners: Stage 3

`dsh check simulation.toml` and `dsh run simulation.toml` remain the preferred
interface. `run_simulation.py simulation.toml` delegates to those commands.

## Architecture

The historical `dsh-v1` CLI and `run_simulation.py` Python-settings workflow
translate settings through `dsh.compat`, then use the strict parser, input
adapters, `build_run_plan`, `execute_plan` and `publish_run`. Neither entry
point initializes JAX or executes Python settings when merely imported.
Backend selection happens before numerical imports.

`python scripts/run_dsh_v1.py ...` retains the CLI compatibility shim.
`python run_simulation.py` still loads `sim_config.py`;
`python run_simulation.py --legacy-config settings.py --check` validates an
explicit Python settings file without transport or output creation. Python
settings execute Python code, as in the historical workflow.

## Compatibility and deliberate changes

- Preserve source sampling, material selections, launch proposals, seed/chunk
  accounting, geometry and observer bins. Historical arrival bins keep their
  equal-width linspace convention even for a noninteger duration/bin ratio.
- Preserve root custom/disc/FITS, representative and continuous flux histories,
  exponential/constant sources and the explicit RefleX table hook. Material
  identity follows the selected files; WD01 bundles are not claimed identical
  to RefleX.
- Historical flux tables explicitly record their first MJD as epoch and the
  last cadence as final interval; unknown time scale is recorded as unspecified.
  Normal TOML flux tables retain explicit timing requirements.
- Preserve CLI NPZ/FITS destinations and root image/event-map/log destinations.
  A sibling `<output-stem>.run` directory contains physical products where
  applicable, translated configuration, report, provenance and completion hashes.
- All runners reject capped or invalid histories with exit code 3 and withhold
  fresh production products. Exit 2 indicates configuration/input errors.
  Existing outputs from an earlier run may remain: only a current verified
  completion marker establishes success. Historical outputs permit replacement.
- Publication checks prevent overwriting inputs or aliasing products, support
  separate volumes, and write the completion marker last.
- Shared plotting supports selected energy bands, whole-bin epochs, centered
  crops, fluence/intensity and scored-event maps. Pixel areas and selected
  duration determine intensity normalization. Plots may differ cosmetically
  from historical figures; the physical arrays are the parity target.

## Validation

Run from the checkout with its environment installed:

```powershell
python scripts/validate_runner_migration.py
```

This runs focused tests and six end-to-end cases: configured disc and discrete
flux, legacy CLI and root, and capped controls for both historical runners.
Checks include physical NPZ/FITS consistency, accounting, mapped destinations,
hashes and plots. It writes `stage3_report.zip` for review. Small packet counts
validate integration, not scientific convergence.

During implementation, seven independent seeded comparisons against the
Stage 2 runner implementations matched sampled packets, numerical inputs,
products and diagnostics exactly (flare, decay, power law, spectral history,
root custom, continuous flux and RefleX-hook cases; 37 packets, chunks of 16).
The main regression suite passed 289 tests before the final inspector-label
adjustment. Protected numerical kernels and benchmark evidence are unchanged.

Slurm submission and response collection/folding remain subsequent stages.
`submit_simulation.sh` continues to invoke the root compatibility entry point;
its resource/configuration interface has not yet migrated to TOML.
