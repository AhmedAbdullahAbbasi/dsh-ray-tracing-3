# External source and material inputs (schema 1)

The file-input runner uses a source FITS and two independently selected
intrinsic material components. These are input formats, not detector products.
The existing `dsh-v1` command remains available for reproducing the validated
pre-refactor checkpoint. A configured run produces ideal-observer fluence and
does not inherit Stage 9F scientific acceptance merely by using pinned tables.

## Source FITS

`DSHSRC=1` identifies the canonical format. `FLUXDEF=INTRINSIC` means the
unabsorbed observer-equivalent photon flux; transport applies absorption and
scattering later. `TFRAME=DIRECT` means direct-light arrival time relative to
the source reference. `TIMEUNIT=s`. When an absolute epoch exists, `MJDREF`
and `TIMESYS` are both required. The source adapter, not the simulator, must
perform any ISS/geocentric/barycentric conversion and record assumptions.

| HDU | Contents | Units |
| --- | --- | --- |
| `PRIMARY` | Format, flux/time convention, `FLU_CONT`, `FLU_LINE`, optional epoch | Fluences: `ph cm-2` |
| `TGRID` | Contiguous `T_START`, `T_STOP` source intervals | s |
| `EGRID` | Contiguous `E_LO`, `E_HI`, `SHAPE` (`FLAT` or `POWERLAW`) | keV |
| `CFLUX` | Continuum matrix `(n_time, n_energy_bin)` | Photon flux **integrated over each bin**, `ph cm-2 s-1` |
| `CGAMMA` | Photon index matrix `(n_time, n_energy_bin)` if any bin is `POWERLAW` | dimensionless |
| `LINES` | Sorted unique `E_LINE`, `LABEL` | keV |
| `LFLUX` | Line matrix `(n_time, n_line)` | Integrated photon flux, `ph cm-2 s-1` |

The continuum pair, line pair, or both must be present. `FLAT` is constant
`dN/dE` within a bin; `POWERLAW` is proportional to `E**(-CGAMMA[t,b])`.
The flux inside each time interval is constant. Zero flux is allowed in
individual cells, but selected source fluence must be positive. Photon
fluence is the sum of flux times interval duration. The writer stores FITS
checksums, and the reader verifies them, shapes, units and both part-fluence
closures. Photon-energy support must lie within the chosen material tables.

This format accepts a time series of spectra **after** a converter has given
each spectrum an interval, a photon-flux normalization and a common energy
grid. XSPEC and observational adapters are future work. It does not interpret
detector counts, correct absorption, or silently interpolate temporal gaps.

## Material component NPZ and JSON

Each component consists of `name.npz` plus adjacent `name.json` schema-1
metadata, including its payload SHA-256. The scattering NPZ contains
`energy_kev`, `scattering_angle_rad`, intrinsic
`differential_cross_section_cm2_per_sr_per_h`, its integrated
`scattering_cross_section_cm2_per_h`, and `scattering_angle_cdf`. The physical
angle spans 0 to pi; there is no observer `(1-x)^-2` factor. The absorption
NPZ contains `energy_kev` and `absorption_cross_section_cm2_per_h`.

The existing loaders check intrinsic quantity/unit labels, metadata and
payload digests, integrated scattering/CDF closure, positivity and exact
matching energy grids. The optional `materials.grid` file checks the shared
generation-grid digest. Cross sections are per hydrogen atom and are point
values at energy nodes, unlike the bin-integrated source fluxes. To replace
TBabs with TBnew, produce a new absorption NPZ and JSON on the selected
scattering energy grid and change `materials.absorption` in the run TOML.
Keep the pinned table bytes and their SHA-256 identifiers unchanged.

## Run configuration and products

Run from the repository root with Astropy and JAX installed:

```bash
python -m dsh.command check configs/file_input_smoke.toml
python -m dsh.command run configs/file_input_smoke.toml
```

An installed package also provides `dsh check` and `dsh run`. The TOML has
`format_version = 1` and sections `[run]`, `[scene]`, `[source]`,
`[materials]`, `[observer]`, `[output]`; paths resolve relative to the TOML.
`scene.kind` is `example_four_cloud` or `fits`. The observer uses explicit
arrival edges in days and energy edges in keV; spatial bin edges match the
cloud's native angular edges. `run.max_interactions` is mandatory and a cap
is always recorded as a numerical failure. `dsh check` validates inputs
without launching photons.

File-input outputs use schema 7: they retain the schema-6 observer cube,
history moments, transport statuses and material arrays, and add numerical
`source_cell_*` fields for each positive-fluence cell. `SOURCE` in FITS
records cell kind (`0` line, `1` flat continuum, `2` power law), time/energy
bounds, photon index, flux, fluence and sampling CDF. NPZ stores source-file
and component hashes, selected components, source epoch and the original and
resolved configurations. FITS records the source/configuration and material
hashes; a JSON run report records numerical terminal status. The separate
`resolved_config.json` uses absolute paths. Schema-6 output from `dsh-v1`
is unchanged.

The bundled example FITS and TOML are synthetic input/scene smoke data. The
resulting halo images are not validated observational or science products.
