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
python -m dsh.command audit configs/file_input_smoke.toml
```

An installed package also provides `dsh check` and `dsh run`. The TOML has
`format_version = 1` and sections `[run]`, `[scene]`, `[source]`,
`[materials]`, `[observer]`, `[output]`; paths resolve relative to the TOML.
For absolute Windows paths, use forward slashes (`C:/data/source.fits`) or
single-quoted TOML literal strings; backslashes in double-quoted strings are
interpreted as escapes.
`scene.kind` is `example_four_cloud` or `fits`. The observer uses explicit
arrival edges in days and energy edges in keV; spatial bin edges match the
cloud's native angular edges. `run.max_interactions` is mandatory and a cap
is always recorded as a numerical failure. `dsh check` validates inputs
without launching photons. `dsh audit` checks saved schema-7 files, numerical
accounting, FITS agreement, current input file digests and the run report,
writing an `audit_report.json`. It does not establish Monte Carlo convergence
or scientific agreement with an independent reference.

File-input outputs use schema 7: they retain the schema-6 observer cube,
history moments, transport statuses and material arrays, and add numerical
`source_cell_*` fields for each positive-fluence cell. `SOURCE` in FITS
records cell kind (`0` line, `1` flat continuum, `2` power law), time/energy
bounds, photon index, flux, fluence and sampling CDF. NPZ stores source-file
and component hashes, selected components, source epoch and the original and
resolved configurations. FITS records the source/configuration and material
hashes; a JSON run report records numerical terminal status. The separate
`resolved_config.json` uses absolute paths. External cloud FITS bytes are
hashed before and after loading, and the digest is recorded in NPZ/FITS and
the run report. Schema-6 output from `dsh-v1`
is unchanged.

The bundled example FITS and TOML are synthetic input/scene smoke data. The
resulting halo images are not validated observational or science products.

## Controlled heterogeneous line check

`scripts/validate_file_input_heterogeneous.py` constructs a reproducible
5.35 keV, one-second line-only FITS source and a native FITS version of the
asymmetric Stage 9F six-shell cloud. It loads the source, cloud and independent
material components through `build_run`, then runs the configured automatic
full-cloud pipeline and audits its schema-7 products. The script separately
uses those **same loaded arrays** with the finite launch cone defined by the
Stage 9F reference. That comparison checks four absolute first-order sky
quadrants against double-precision quadrature, repeated scattering against an
independent explicit-absorption path scorer, escape columns against a host
voxel integral, and numerical terminal states. The quadrature includes the
finite source interval in the arrival-window acceptance.

```bash
python -m scripts.validate_file_input_heterogeneous \
  --output-dir outputs/controlled_line --packets 30000 \
  --seeds 912 319 141 --configured-packets 8192
```

The JSON report labels the two launch cones separately. The configured
full-cloud run is checked for numerical and product integrity; the finite-cone
Stage 9F comparison supplies scientific acceptance **only for that controlled
proposal**. A full-cloud, time-resolved realistic scene requires its own
precision and convergence assessment. The script returns a nonzero status if
any of its recorded gates fail.

On 2026-09-29, the command above passed all seven Stage 9F gates with three
30,000-packet seeds. The four independent first-order quadrants had relative
photon standard errors of 5.44–6.88% and a maximum quadrature discrepancy of
2.80 combined standard errors (acceptance: at most 10% and 5 respectively).
The largest independent escape-column relative error was 9.1e-7 across 120
sampled events, including 79 repeated-scattering events. The 8,192-packet
configured full-cloud run passed its numerical status and schema-7 archive
audit. This evidence is for the synthetic controlled scene and one broad
arrival window, not for time-resolved production predictions.
