# Intrinsic spectral histories and physical halo response

This milestone adds changing intrinsic spectra to the existing elastic transport.
The user reported that the external XSPEC gas checks pass on 2026-10-16. Their
raw report and XSPEC version were not supplied here. This does not alter the
earlier prepared-only XSPEC records or claim that XSPEC ran in this environment.

## Source contract

`build_spectral_history_source(time_edges_s, energy_edges_kev, band_flux,
photon_index)` accepts an integrated photon flux and a local power-law slope
in each time–energy cell. Flux is **unabsorbed observer-equivalent** in
ph cm^-2 s^-1. Within a cell, `F_E = band_flux * E**(-Gamma) / integral(E**(-Gamma))`.
Time is piecewise constant, and energy is sampled continuously by analytic
inverse CDF. Slopes can differ between time intervals and energy bands. A
spectral break belongs on a band edge; the amplitudes on either side are
explicit and need not be continuous. This permits broken spectra or tabulated
approximations to fitted continua without hiding a spectral interpolation rule.

`build_powerlaw_band_source` also accepts a vector of photon indices, one per
time interval, with `photon_flux` normalized over the entire energy range. Its
scalar-index path and existing representative-energy simulations are preserved.
Spectral-history packets have the same layout and use the same launch, voxel
transport, outgoing extinction, peel-off scoring, and observer binning.

Time intervals may have zero flux. Missing values, negative flux, overlapping
or collapsed time/energy edges, lost positive CDF cells, excessive power-law
dynamic range at JAX sampling precision, and zero total fluence
are errors. Gaps must be resolved by an explicit caller policy; a declared
zero-flux gap is a physical assumption, not automatic interpolation. Material
and response energy support must cover the full source range. Absorption is
applied only by transport; an absorbed input spectrum would double-count it.

### JSON file

The adapter requires exactly these fields, including an explicit epoch and time
scale. Times are **seconds relative to that epoch**, not large absolute MJD or
Unix timestamps stored in float32. The code retains these labels but does not
perform barycentric corrections or convert between UTC, TT, and TDB. Arrival
times must use the same relative origin and scale.
The existing retarded-time convention is `arrival = emission + excess_delay`;
the common direct-flight D/c offset is removed. A history inferred from direct
source spectra should use that direct-light-curve origin. Flux is zero outside
the supplied history, so include the earlier emission needed for the delayed
observer intervals being modeled.

```json
{
  "schema_version": 1,
  "flux_convention": "unabsorbed observer-equivalent",
  "units": {"time": "s", "energy": "keV", "band_flux": "ph cm^-2 s^-1"},
  "epoch": "t=0 of the simulation",
  "time_scale": "simulation-relative",
  "time_edges_s": [0, 1800, 5400, 21600],
  "energy_edges_kev": [2, 4, 10],
  "band_flux": [[0.02, 0.02], [0, 0], [0.006, 0.002]],
  "photon_index": [[1, 1], [1, 1], [2.2, 2.2]]
}
```

The example's fluxes are independently declared per band, not a claim of a
continuous normalization at 4 keV. A programmatic single-power-law constructor
provides that normalization when wanted. The controlled validation supplies a
complete machine-readable history under `spectral_histories_2026-10-16/history.json`.

```powershell
python -m scripts.run_dsh_v1 --materials declared `
  --material-directory dsh/data/materials/wd01_wilm_c50_si90_v1 `
  --source-history benchmarks/materials/spectral_histories_2026-10-16/history.json `
  --packets 8192 --chunk-size 1024 --output outputs/spectral_history.npz
```

`--source-history` and `--source-spectrum` are mutually exclusive. The history
supplies its own time model; it cannot be combined with `exponential-decay`.
NPZ schema 8 stores all cell slopes and fluxes, source energy/time edges,
epoch/scale labels, and the original input-file SHA256. FITS `SOURCE` rows now
include cell `PHOTON_INDEX`, `ENERGY_LOW/HIGH`, time bounds, and integrated flux.
The primary header keeps a scalar `PHINDEX` only for scalar-index sources;
histories use `SPECINT=CELLPOWERLAW`, plus source-file hash and time labels.

## Response estimator

The response is a **physical halo response**, including all elastic scattering
orders and outgoing extinction for one static material/cloud/source geometry.
It does not include the direct unscattered beam or instrument area, PSF,
energy redistribution, exposure, background, or detector counting statistics.

For N independent histories sampled from the normalized proposal q(E), the
scorer's angular/escape contribution is G. Each response event stores its exact
energy, delay, sky position, order, history index, and `G / (N*q(E))` in keV.
To fold a source, multiply this weight by the exact time integral of F_E at the
sample energy over each arrival interval shifted by the delay, then sum.
Proposal/source packet fluence is divided out once. No second absorption factor,
delay rounding, energy-bin midpoint evaluation, or jump interpolation is used.

A proposal has one time interval, continuous energies, and positive flux in
every band. Its absolute flux cancels. A broad log-uniform proposal is obtained
with Gamma=1; independently chosen positive band fluxes can oversample edge
neighborhoods. The sampling distribution affects precision, not normalization.
Very different later spectra may need more response histories, especially if
they emphasize an undersampled narrow feature. Uncertainty is conditional on
the declared material, geometry, and source; no fit errors are propagated.

The response's NPZ archive uses no pickle, has an array checksum and metadata
checksum, and records exact array hashes for material, cloud, and launch inputs,
runtime source-file hashes, proposal, seed, history count, and terminal statuses.
The collector rejects numerical failures and capped histories rather than
silently treating them as complete. The archive represents its recorded model;
loading it does not claim that it matches some other current cloud/material.

`fold_spectral_response` returns `(time, energy, radius)` fluence and Monte Carlo
standard errors. Its optional full covariance uses C-order flattening. Every
event from one launched photon is grouped before forming moments, including
correlations between scattering orders/bins. Zero-score histories count toward
N. For per-history bin contributions Y_i (which already include 1/N),

`Cov = N/(N-1) * [sum_i Y_i Y_i^T - outer(sum_i Y_i, sum_i Y_i)/N]`.

Finite arrival/radial windows intentionally lose flux. Bins include their low
edge and exclude their high edge, except for the final upper edge. Full dense
covariance is limited to 2048 bins; `--no-covariance` retains diagonal standard
errors for larger products. The proposal must cover all source energy bands.

### Collect and reuse

The included `spectral_response_proposal.json` uses one time interval, positive
band fluxes, Gamma=1, and extra sampling around Fe K. Collection supports an audited material directory and
either a physical cloud FITS cube or the built-in four-cloud scene:

```powershell
python -m scripts.generate_spectral_response --proposal-history benchmarks/materials/spectral_response_proposal.json `
  --material-directory dsh/data/materials/wd01_wilm_c50_si90_v1 `
  --cloud-fits cloud.fits --source-distance-kpc 10 `
  --packets 262144 --chunk-size 8192 --output outputs/halo_response.npz

python -m scripts.fold_spectral_response --response outputs/halo_response.npz `
  --source-history history.json --arrival-time-edges-s 0 3600 21600 86400 1036800 `
  --energy-edges-kev 2 4 6.9 7.124 7.4 10 --radial-edges-arcsec 0 90 600 `
  --output outputs/folded_history.npz
```

The fold CLI output keeps axes, fluence, standard errors/covariance, source-file
hash and epoch/scale, and full response provenance. Change material, geometry,
or non-elastic physics by collecting a new response. Changing only the intrinsic
history can reuse the existing response within its energy support.

## Validation

The standalone statistical validation uses SciPy (available in the `benchmark`
or `materials` installation extras). Source construction and response folding
need only the core NumPy/JAX dependencies.

```bash
python -m scripts.validate_spectral_histories \
  --output benchmarks/materials/spectral_histories_2026-10-16
python -m unittest discover -s tests -t .
```

The frozen experiment uses a 10 kpc point source, a uniform 30 pc screen halfway
to it, column 1e23 cm^-2, the experimental C50/Si90 Wilms-budget WD01/D03 material,
4194304 response histories and 4194304 independent direct histories. The source
changes from Gamma=1 to Gamma=2.2, with a zero-flux interval. Energy bins split
at 6.9, native Fe K threshold 7.124, and 7.4 keV. Arrival bins and two radial
annuli resolve delayed evolution. The response proposal oversamples Fe K.

The comparison tests absolute time/energy/radial fluence, separately for total,
first, and multiple orders, using independent per-history covariance on each
side. Fixed gates require at least 12 well-sampled bins, maximum |z|<5,
window |z|<5, global p>0.001, at least 100 multiple-order events, complete
transport, source joint-cell sampling within 5 sigma, production/host binning
agreement within 2e-5, and exact order closure. This is a reuse/sampling check
against the same production physics, not a new external scattering calibration.
Existing independent material, absorption-edge, and RefleX evidence remains
the physics baseline.

The numerical safety limit is 16 interactions; capped histories fail the test.
The million-history pilot reached at most scattering order eight in its
response sample. Raising the history count does not change the physical model.

The initial 262144-history experiment passed the total and first-order checks
but had only eight well-sampled multiple-order bins, below the preset minimum
of twelve. Its report, arrays and response are retained in `pilot_262144/`.
The 1048576-history pilot still had only ten well-sampled multiple-order bins.
It is retained under `pilot_1048576/`. The final run increases histories to
4194304 per side and preserves every acceptance gate.

The final independent comparisons pass:

| Component | Well-sampled bins | Window z | Maximum bin \|z\| | Global p |
| --- | ---: | ---: | ---: | ---: |
| Total | 30 | -0.210 | 2.019 | 0.851 |
| First order | 30 | -0.372 | 2.176 | 0.657 |
| Multiple orders | 17 | -0.054 | 1.229 | 0.987 |

The passing in-memory comparison was followed by a truncated response export.
NPZ exports now write a temporary archive and atomically replace the destination
after ZIP finalization. Replaying the frozen response seeds reproduces the
saved fluence and covariance bit for bit; the recovered archive passes CRC,
content checksums, and runtime-file verification. The original
comparison report and replay verification are retained in `archive_recovery/`;
the incomplete archive is excluded. Fault-injection tests ensure that a failed
replacement preserves an existing completed archive.
The final suite passes all 220 tests with the optional Mie dependencies enabled.

Unit tests also cover analytic cell normalization, Gamma=1, hard/soft joint
distributions, missing-data rejection, explicit time/units input, fractional
delay/window losses, exact samples on either side of a jump, grouped covariance
with repeated events and zero-score histories, absolute proposal cancellation,
incomplete transport rejection, checksum tampering, and FITS/NPZ preservation.
Four legacy source cases (representative energies, Gamma=1, Gamma=1.7, and
variable brightness with fixed slope) additionally produce exactly the same
source arrays and seeded packets as `54b1be8`; each comparison uses 10000
packets. The frozen RefleX runtime/input fingerprint and all 63 protected
adapter evidence files were checked without changing that benchmark.
