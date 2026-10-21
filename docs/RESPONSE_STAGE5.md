# Configured physical response workflows (Stage 5)

The TOML interface supports `run.workflow = "response_collect"` and
`"response_fold"`. Use the same `dsh check`, `dsh run`, `dsh slurm-plan` and
`dsh slurm-submit` commands as simulations. Cluster execution remains pending
site validation. No transport, scattering, absorption or observer algorithms
were changed by this stage.

This response is the physical dust halo transfer for one static scene and
material. It contains scattered halo samples with extinction already applied.
It does not include the direct beam, telescope effective area, detector
redistribution, PSF, background or counting noise.

## Collect once

Start from `configs/response/collect.toml`:

```bash
dsh check configs/response/collect.toml
dsh run configs/response/collect.toml
```

Select the scene and material using the same sections as a simulation, including
FITS clouds, uniform discs or the four-cloud example; declared material bundles
or explicit table pairs; and radial or rectangular launch proposals. The
configuration file remains the only place to select simulation inputs.

`[source]` defines the energy-sampling proposal. It must be continuous, have
exactly one time interval, and have positive flux in every energy band. Power
law, band power law, spectral-history JSON and continuous flux-table adapters
are available. The numerical builder rejects incompatible loaded histories
before launching transport. Proposal energy support must fit the material.
Its absolute fluence cancels in the response estimator; exact sampled energy
and geometric excess delay are retained for later target histories.

Execution requires at least two total histories and chunk size at least two.
The established collector retains its seed splitting, normalization, history
indices and treatment of a final one-packet remainder. Zero-score histories
remain in the denominator. Capped or invalid histories reject the collection:
exit code 3, an incomplete report, no new response archive or completion marker.
There is no option to publish an incomplete response as a production archive.

Output: `<output.directory>/<run.name>.npz`, using the existing schema-1 response
format. `load_spectral_response` and the historical folding CLI can read it.
Exact sample arrays and units, internal checksums, material/scene/launch/runtime
fingerprints, configuration identity and input hashes are retained. Source,
material and geometry belong to the recorded collection; changing the material
or scene requires another collection.

## Fold many target histories

After collection, use `configs/response/fold.toml`:

```bash
dsh check configs/response/fold.toml
dsh run configs/response/fold.toml
```

The source section selects the target continuous spectrum and light curve.
`[response].file` selects the reusable archive. This workflow has no scene or
material section, and accepts only `device` under execution: it launches no
photons. Folding uses CPU NumPy and requires `execution.device="cpu"`.
Source support must lie within the archived proposal. `check` loads
and validates the archive checksum, source and support without folding or
creating output directories.

`[observer]` requires explicit `arrival_time_edges_days`, `energy_edges_kev`
and `radial_edges_arcsec`. Times are relative to the source epoch declared by
the source adapter. Stored sample delays are integrated against the target
history exactly as in the existing folding API. Finite time, radius, energy
and optional scattering-order windows can exclude fluence; no renormalization
is applied. All orders are included unless `response.scattering_orders` is set.

`response.covariance = true` saves the dense covariance matrix, with the existing
2048-bin limit; set false to retain diagonal standard errors without allocating
the dense matrix. `history_block_size` controls folding workspace. Preflight
reports its principal grouped-workspace estimate. Large diagonal-only grids
still require memory for one history block; reduce this setting as needed.

Fold output retains the existing NPZ fields: `fluence`, `standard_error`,
optional `covariance`, the three edge arrays, and `metadata_json`. Fluence is
in ph cm^-2, axes are (arrival time, energy, radius), and covariance uses C-order
flattening. All events and scattering orders from each launched photon remain
grouped, including zero-score photons. These uncertainties describe finite
Monte Carlo response statistics, excluding source-fit uncertainties and
instrumental counting noise. Report/provenance sidecars also describe the
selected archive and target source.

## Publication and compatibility

Both workflows use shared input loaders, path/overwrite protection, SHA256
verification and atomic per-file publication. Each output directory receives
`original.toml`, `resolved.json`, `run.log`, `report.json`, and a completion
manifest written last. Existing outputs require explicit `output.overwrite=true`.
An unsuccessful overwrite can leave earlier products; only the current verified
completion marker establishes success. Missing, malformed or unsupported inputs
return exit code 2. Configured workflows preserve the legacy response archive
schema and numerical APIs; the historical response scripts remain available.

`python scripts/validate_response_stage5.py` runs focused tests, actual
collection, archive roundtrip, target folding and a capped rejection control,
then writes `stage5_report.zip`. It independently recomputes the folded arrays
through the existing numerical API. Small examples establish integration rather
than scientific convergence; increase histories and inspect Monte Carlo errors
before interpreting real halo predictions.
