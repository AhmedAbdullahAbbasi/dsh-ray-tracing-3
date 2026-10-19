# TOML-configured simulations: first implementation milestone

The installed `dsh` command accepts one simulation file. Material selection,
cloud geometry, source inputs, sampling, observer bins, products and basic plots
are selected there. Existing `dsh-v1`, root scripts and numerical APIs remain
available. This implementation adds host orchestration; it does not replace the
transport, scattering, absorption, source samplers or observer normalization.

```powershell
python -m pip install -e ".[dev,plots]"
dsh check configs/smoke_legacy.toml
dsh run configs/smoke_legacy.toml
dsh check configs/smoke_wd01.toml
dsh run configs/smoke_wd01.toml
```

These examples use 64 CPU packets for integration checks, not scientific
convergence. Outputs go under `outputs/`. `overwrite = false` rejects existing
destinations. Select another output directory or deliberately set `overwrite =
true` before repeating a run. Editing or selecting the TOML file is the normal
way to change a simulation; the configured command has no physics overrides.

## Runtime architecture

| Owner | Contract |
| --- | --- |
| `dsh.config.schema`, `dsh.config.load` | Immutable host settings, strict modes/types, defaults, relative paths, selected sidecars and output protection; no NumPy/JAX/Astropy/Matplotlib imports |
| `dsh.command` | Parse configuration, select CPU/GPU before numerical imports, then check or run |
| `dsh.inputs` | Existing scene/source/material builders behind explicit file adapters |
| `dsh.build` | Validated arrays plus separate host metadata in a `RunPlan`; no transport or output writes |
| `dsh.run.execute_plan` | Existing chunked pipeline; returns numerical products without file I/O |
| `dsh.quality` | Terminal-state/accounting/finiteness acceptance, separate from window losses and convergence |
| `dsh.run.publish_run`, `dsh.io.provenance` | Original/resolved settings, input/runtime hashes, staged products and final completion marker |
| `dsh.plots` | Optional derived views using actual bin edges and each pixel's own area |

The old input/config branch was inspected as an architectural reference. Its
older numerical code and source-FITS format were not merged into the current
checkpoint. Current JSON histories and declared edge-aware materials retain
their existing loaders and validation.

## Settings supported now

- `config_version = 1`; `[run]` requires a safe `name`; `workflow = "simulate"`.
- `[scene]`: `kind = "fits"` with `cloud_file`, or `"example_four_cloud"`;
  positive `source_distance_kpc`; `coordinates = "source_centered_offsets"`.
  FITS axes must already be source-centered local offsets, not celestial WCS.
  The source stays on the line-of-sight axis; no source-pixel recentering is implied.
- `[materials]`: `kind = "table_pair"` with `scattering_file`, `absorption_file`
  and a required `grid_file` when the tables declare one; or
  `kind = "declared_bundle"` with `directory`. All component and stored-total
  consistency checks remain active. The WD01-shaped Wilms-budget bundle is
  experimental and does not reproduce identical RefleX physics.
- `[execution]`: `packets`, `chunk_size`, `max_interactions`, uint32 `seed`,
  `device = "cpu" | "gpu"`, `launch_sampler = "radial" | "rectangular"`.
  GPU means the CUDA backend in this milestone; unavailable devices fail.
  No precision switch is advertised; actual dtypes are recorded.
- `[observer]`: `sky_grid = "cloud"`, independent `energy_edges_kev`;
  `[observer.arrival]` selects `kind = "explicit"` and `edges_days`, or
  `kind = "uniform"` and `start_days`, `stop_days`, `step_days`,
  `last_bin = "shorten"`. Bins begin at zero, cover source energy support,
  and extend beyond source emission. Scattered flux can still leave the final
  arrival window; it is reported rather than renormalized.
- `[quality]`: `allow_incomplete = false`, `save_failure_diagnostics = true`.
  Incomplete transport returns exit code 3. Explicit `allow_incomplete = true`
  permits diagnostically labeled products, still returns 3 for incomplete
  transport, and never writes a production completion marker.
- `[output]`: `directory`, nonempty unique `formats = ["npz", "fits"]`,
  `overwrite = false`. NPZ-only and FITS-only are supported.
- `[plots]`: `enabled = false`, `scales = ["log", "linear"]`,
  `energy_selection = "all"`, `crop = "full"`, optional two-value `epoch_days`
  aligned to observer bins, `dpi = 200`, `save_scored_event_map = false`.
  Intensity is averaged over the selected duration and output energy width,
  in ph cm^-2 s^-1 keV^-1 arcsec^-2. Event maps are scored Monte Carlo events,
  not detector counts. Zero images are supported.
- `[logging]`: positive `progress_every_chunks`, simple `.log` `file` basename.

Unknown keys, conflicting modes, unsupported versions, booleans as counts,
nonfinite values and missing files/sidecars fail with field-specific errors.
Relative paths use the configuration's directory, not the current shell
directory. Home expansion is supported; environment expansion is rejected.
Use forward slashes or TOML literal strings for Windows paths.

## Runnable configuration examples (Stage 2)

All seven files below work directly from the checkout. They use synthetic
inputs and 64 CPU packets. Each selects a separate output directory; no Python
settings need editing. The five `example_*.toml` files also save log/linear
intensity plots for days 0–20 and the corresponding scored-event map.

| Configuration in `configs/` | Source / scene | Input contract demonstrated |
| --- | --- | --- |
| `smoke_legacy.toml` | Representative energies / four-cloud scene | Three material nodes, explicitly integrated band fluxes |
| `smoke_wd01.toml` | Continuous power law / four-cloud scene | One broad-band flux split by the existing source builder |
| `example_band_powerlaw.toml` | Per-band power law / four-cloud scene | Independent band fluxes in two time intervals |
| `example_exponential.toml` | Post-peak exponential / four-cloud scene | Peak, baseline, decay time and exact time-cell averages |
| `example_history.toml` | Spectral-history JSON / four-cloud scene | File-owned epoch, units, band fluxes and per-cell slopes |
| `example_flux_table.toml` | MJD text flux table / four-cloud scene | Explicit full-table columns and gap/final-interval policies |
| `example_cloud_fits.toml` | Representative energies / native FITS | Source-centered sky offsets and column-per-voxel geometry |

For example, from `C:\Users\suuser\Desktop\dsh-simulation-config`:

```powershell
.\.venv\Scripts\python.exe -m dsh.command check configs/example_flux_table.toml
.\.venv\Scripts\python.exe -m dsh.command run configs/example_flux_table.toml
```

`configs/data/README.md` describes the small committed input fixtures. Replace
file selections and physical settings in a copied TOML to define another run.
Do not reinterpret the synthetic fixture epoch or columns as real MAXI data.
For a real FITS cube, all radial cell edges must be within the selected source
distance, sky axes must already be centered on the source sightline, and
voxel values must be hydrogen column increments in cm^-2.

Stage 2 rejects zero-flux inline sources, peaks below their baselines,
inconsistent source/observer support, excessive uniform-bin counts, invalid
plot epochs and final flux intervals longer than the gap limit before loading
the numerical runtime. File content, material coverage and runtime precision
checks still belong to `check`. Adapter errors identify `source.file`,
`scene.cloud_file` or `materials`; likely key typos include a spelling hint.

Run the portable Stage 2 acceptance check with:

```powershell
.\.venv\Scripts\python.exe scripts/validate_configuration_examples.py
```

This runs the focused tests, copies all seven examples to fresh session
directories, executes `check` and `run` from a different directory containing
spaces, and independently checks physical outputs, hashes, moments, units,
FITS checksums and plots. It prints a compact `stage2_report.zip` under
`outputs/configuration-stage2/<session>/`. Full NPZ/FITS/event arrays stay in
the corresponding local `products` directories. The report records the actual
platform; Linux evidence is not Windows evidence.

### Source alternatives

`kind = "representative"` uses `time_edges_days`, `effective_energy_kev` and
2D integrated `band_flux`. `kind = "exponential_decay"` uses
`time_edges_days`, `effective_energy_kev`, `peak_band_flux`, optional
`baseline_band_flux`, and `decay_time_days`; the established exponential
builder retains exact time-cell averages.

`kind = "powerlaw"` uses `time_edges_days`, `energy_edges_kev`, one integrated
broad-band `photon_flux` per time interval and optional scalar `photon_index`.
`kind = "band_powerlaw"` instead accepts independently integrated 2D
`band_flux`, which is never split or normalized a second time.

An existing spectral-history JSON supplies its own epoch, units and slopes:

```toml
[source]
kind = "spectral_history"
file = "../inputs/source_history.json"
```

A whitespace text flux table requires an explicit data contract:

```toml
[source]
kind = "flux_table"
file = "../fluxes/1630_allflux.txt"
time_column = 1
flux_columns = [3]
column_numbering = "one_based_full_table"
time_format = "mjd"
time_scale = "TT" # Example only: verify against YOUR data provenance.
epoch_mjd = 57625.0
interval_policy = "hold_until_next"
last_interval_days = 1.0
max_interval_days = 1.0
missing_data = "error"
flux_convention = "unabsorbed observer-equivalent"
flux_unit = "ph cm^-2 s^-1"
flux_scale = 1.0
spectral_mode = "band_powerlaw"
energy_edges_kev = [2.25, 3.15]
photon_index = 2.0
```

Columns count from one in the full table. Old `FLUX_COLUMNS=(2,)`, which
removed the MJD column first, corresponds to new `flux_columns=[3]`.
MJD epoch subtraction happens in host float64 before relative seconds enter
JAX. Rows hold their band flux until the next row. One row is supported with
the explicit final duration; duplicates, decreasing times, missing values and
intervals exceeding the declared maximum fail. Time scales are preserved,
not converted between UTC/TT/TDB. Flux values must already be intrinsic
band-integrated photon flux; declaring a unit does not convert instrument
rates, absorbed flux or energy flux.

## Saved products and limits

Successful production runs save original TOML bytes, expanded `resolved.json`,
`report.json`, a log, physical products and optional plots. NPZ/FITS gain
optional configuration identity/completion fields; their existing data schema
numbers and numerical array meanings are preserved. A `complete.json` marker
is published last and hashes the entire output inventory. Each file is
published atomically; a multi-file run is not claimed to be one transaction.
Inputs/sidecars are protected through resolved paths, symlinks and hardlinks.

Input SHA256 hashes are checked before and after loading, and before execution
and publication. This detects ordinary concurrent edits; it is not an
adversarial filesystem snapshot. Runtime identity includes source-file hashes,
Git state, Python/NumPy/JAX, devices, dtypes, seed and chunk settings. `check`
loads inputs and may initialize the backend, but does not invoke transport,
create outputs or submit jobs. Its memory figure is a product-array lower
bound, not a peak GPU-memory prediction.

Still staged for later work: root-runner migration, disc/FRED/variable-source
configuration, RefleX-export adapter convergence, Slurm resources/submission,
response collection/folding, partial observer energy windows, arbitrary image
band/crop choices and source-pixel recentering. Unsupported fields fail rather
than silently doing nothing. The full handoff remains the plan for these slices.

## Verification

```powershell
python -m unittest tests.test_run_config tests.test_configured_run -v
python -m unittest discover -s tests -t . -v
python -m unittest reflex_adapter.test_reflex_materials -v
python -m unittest discover -s benchmarks/newdust -p "test_*.py" -v
```

Seeded parity is checked against the existing explicit pipeline on the same
backend, seed, chunking and software environment, including a short final
batch. Cross-platform bitwise identity is not promised. Production completion
does not establish Monte Carlo convergence or detector-response validation.
