# Validated pre-refactor code handoff (24 September 2026)

## Checkpoint and meaning of “validated”

This document describes the code snapshot on the `dsh` branch marked by the
`stage9f-validated-pre-refactor` tag. Read it together with `README.md`,
`docs/validation.md`, `docs/stage9f.md`, and
`docs/stage9f_heterogeneous.md`. The tag is the code reference; the JSON
reports listed below identify the numerical experiments and their environment.
The controlled Stage 9F **ideal-observer fluence** validation is accepted at
the specified precision and geometry. The existing four-cloud flare images
are **not** validated science products. No detector simulation, measured gas
cube, empirical source fit, single-grain test, or external literature
comparison has been accepted. Restructuring must retain these distinctions.

The validation work in this checkpoint adds validation-only runners,
independent reference calculations, documentation, and tests on top of the
Stage 9F implementation already at local commit `1bb4a763af51bfadb077619c8dcf669ff0eb459b`.
This last set of changes does **not** modify the production `dsh/geometry`,
`dsh/physics`, `dsh/sources`, `dsh/transport`, `dsh/observer`, `dsh/io`,
`dsh/pipeline.py`, or `dsh/cli.py` modules. The earlier stages of validation
*did* change the main code: examples include photon-history moments in
schema 6, source-energy endpoint and near-unit-index sampling, host precision,
and FITS axis/unit and product-integrity checks. Consult Git history to
distinguish the earlier core fixes from this final validation-only addition.

## Repository map and entry points

| Path | Responsibility |
| --- | --- |
| `dsh/geometry/{coordinates,clouds,rays}.py` | Coordinates; native angular–distance column cube; exact straight-ray voxel segments and columns. |
| `dsh/physics/{dust,newdust,rg_drude,absorption,materials,material_grid}.py` | Frozen cross sections, material validation/loaders, offline RG/Drude calculator, energy/angle interpolation. |
| `dsh/data/{newdust,scattering,absorption,materials}/` | Versioned NPZ table payloads and JSON metadata/digests, including the 2–10 keV pair. |
| `dsh/sources/{models,launch}.py` | Time/band source probabilities, continuous power-law energies, launch proposal and importance weights. |
| `dsh/transport/{kernel,directions}.py` | JAX photon tracing with repeated competing-risk interactions; stable direction geometry. |
| `dsh/observer/{scoring,binning}.py` | Next-event observer estimator, canonical 4D binning and per-history sums/moments. |
| `dsh/io/{cloud_fits,npz_output,fits_output}.py` | FITS cloud input and reproducible NPZ/FITS products. |
| `dsh/{pipeline,cli,examples}.py` | Chunked orchestration, user CLI, synthetic four-cloud example. |
| `dsh/validation/` | Analytic tests, independent host references and Stage 9A–F controlled experiments. These are not called by production transport. |
| `scripts/` | Module runners, material-generation utilities, flare/snapshot audits and the PowerShell 2.5-million-packet launcher. |
| `tests/`, `docs/` | Regression tests and stage-specific scope, commands, and interpretation. |

Python requirement is `>=3.10`; package requirements are `jax>=0.4.34` and
`numpy>=2.0`. FITS I/O requires the `astropy>=6` optional extra. Ruff is a
development dependency. The declared minimums are **not** an assertion that
all combinations were independently exercised. For the normal smoke run use
`python -m scripts.run_dsh_v1` (or installed `dsh-v1`); the 2–10 keV material
pair and continuous spectrum require explicit CLI selection. Production
images are ideal observer fluence, not telescope counts.

## Physical contract

### Geometry and gas

Cartesian positions and photon spatial momenta are ordered
`(line_of_sight, sky_x, sky_y)`; the observer is at the origin and the
centered point source at `(D,0,0)`. Position is in parsecs, energy in keV,
and four-momentum `(E,p_los,p_sky_x,p_sky_y)` is null (`|p|=E`, `c=1`).
The fractional dust distance is `x=r/D` **from the observer**. Sky offsets
are `atan2(sky_x,los)` and `atan2(sky_y,los)` in arcseconds. Observer time
relative to the direct photon is emission time plus the actual broken-path
excess `(L_actual+r-D)/c`; the small-angle first-scatter reference is
`D*x*theta^2/[2*c*(1-x)]`. Rationalized excess-distance formulas retain
arcsecond delays under JAX float32 rounding.

Cloud arrays have order `(distance, sky_y, sky_x)`, and each cell stores its
*radial column increment* `delta_NH` in `cm^-2`. Within a radial cell the
hydrogen density is constant, `n_H=delta_NH/(delta_r in cm)`. Radial faces
are spheres about the observer and angular faces are planes
`sky_x=los*tan(theta_x)` or `sky_y=los*tan(theta_y)`. `dsh/geometry/rays.py`
finds analytic line/surface intersections, sorts them into segments and sums
`n_H * length`; no Cartesian remap, random stepping, or substep integration
is applied. The world ends at the observer plane or the source-radius outer
sphere; the cloud's outer edge must be strictly before the source.
Cloud FITS ingest reads linear `CRPIX/CRVAL/CDELT` axes and converts supported
distance/angle units with `BUNIT=cm-2`; celestial or rotated WCS is rejected.
An external FITS file supplies gas directly; the code does not reconstruct
missing material outside that input cube.

### Materials

The scattering input is intrinsic per-hydrogen
`d sigma_sca / d Omega (E,theta)` in `cm^2 sr^-1 H^-1`, with physical
scattering angle (not the observed sky angle). Integrating it over
`2*pi*sin(theta) d theta` gives `sigma_sca(E)` and the normalized angular
probability density `p=(d sigma_sca/d Omega)/sigma_sca`; its CDF drives
elastic-angle sampling. The pinned model is an MRN `a^-3.5` size distribution
with 100 radii from 0.005 to 0.3 microns, grain density 3 g/cm³ and dust
mass per H `2.32475e-26 g`, evaluated with the Gaussian Rayleigh–Gans/Drude
approximation. It is not a Mie or selectable multi-grain calculation. Old
thin-screen tables with an observer geometry factor are provenance comparisons,
not physical cross sections consumed by transport.

Photoelectric absorption is an intrinsic per-H TBabs version 2 opacity on
the same energy grid, generated with XSPEC 12.14.0, Wilms abundances and a
Verner atomic cross-section baseline. Default `v1` material tables contain
3.3, 4.9 and 6.9 keV. The explicit `--materials 2-10` bundle spans 2–10 keV
at 130 adaptively chosen energies and 8,193 scattering angles from 0 to pi;
the two table payloads and JSON metadata are checked for grid/digest
consistency. Frozen 2–10 keV SHA-256 identifiers in the reports:

| Table | SHA-256 |
| --- | --- |
| Scattering `rg_drude_2_10_v2.npz` | `72253fce338bcbba2f2636afc56b2e86e710c92d7b7cfb23d413c440e52dc51c` |
| Absorption `tbabs_wilm_vern_2_10_v2.npz` | `cd146427be2a2fb363e3b365e2b322d8d16e9918ef4ed8cdff8f32aaa5cf1943` |

Scattering/absorption cross sections interpolate logarithmically in energy
when positive; exact zeros have linear fallback. The scattering-angle CDF is
interpolated between energy rows and the scorer interpolates the normalized
differential phase within angle/energy rows. Outside-grid photon energies are
invalid, not extrapolated. Four extremely short unresolved absorption-edge
intervals near 2.477, 3.203, 4.043, and 7.124 keV are recorded in
`dsh/data/materials/material_grid_2_10_v2.json`; each is under `1e-4` keV.
Interpolation inside them is approximate (the 7.124-keV jump is about 30%).
The finite search does not certify absence of still narrower features.

### Source, transport and observer

Source fluxes mean **unabsorbed, observer-equivalent** photon flux in
`ph cm^-2 s^-1`. A piecewise-constant source chooses time/band cells in
proportion to integrated cell fluence, emits uniformly within each selected
time cell, and assigns packet weight `total_source_fluence/N` over the **entire
run**, including when run in chunks. The default representative source has
one hour of emission at 3.3/4.9/6.9 keV, with configurable band fluxes. An
exact cell-average post-peak exponential builder and a continuous fixed-index
power-law library API exist. CLI hard-state mode uses a one-hour constant
flare and `dN/dE proportional to E^-Gamma` across 2–10 keV; energy is
sampled continuously (stable inverse CDF, including near `Gamma=1` and the
upper endpoint) and is binned into 2–4, 4–6 and 6–10 keV for output. CLI
currently rejects combining hard-state mode with exponential-decay mode.
The usual `Gamma=1.7` and integrated `0.038 ph cm^-2 s^-1` are illustrative
inputs, not fitted observations.

Source directions are drawn uniformly in rectangular slopes `(u,v)` with
unit direction `(-1,u,v)/sqrt(1+u^2+v^2)`. The solid-angle proposal is
`q(Omega)=(1+u^2+v^2)^(3/2)/[(u_max-u_min)(v_max-v_min)]`; the isotropic
importance weight is `1/(4*pi*q)`. `build_cloud_launch_geometry` constructs
a finite bounding cone using cloud radial/angular edges. Scientific
normalization requires that proposal to have support over every relevant
photon path; the hand-specified finite cones in Stage 9F validate conditional
fluence only and do not prove full support of the production proposal for a
real cloud.

In a voxel, extinction is `n_H*(sigma_sca+sigma_abs)` in `cm^-1`. Each flight
draws an exponential optical depth `-ln(U)` and carries it across exact
segments until a collision or world boundary. At collision, the two opacities
select scattering or terminal absorption. A scatter samples the material CDF
and a uniform azimuth, changes direction and draws a fresh flight depth;
absorption sets outgoing momentum to zero and records deposited energy.
Each packet has a fixed-length, masked JAX interaction history for JIT and
`vmap`. Status codes are active 0, observer plane 1, outer boundary 2,
absorbed 3, **numerical cap 4**, invalid energy 5, invalid state 6. A cap hit
is censored numerical transport, never a physical termination. The CLI
default `max_interactions=8` is insufficient for the optically thicker Stage
9F shell; those accepted controlled runs use 32. The production launcher
currently specifies 16 and still requires a scene-specific cap check.

Every physical scatter creates a virtual ray toward the observer. With
`r=|position|`, source distance `D`, sampled-launch density `q`, incoming
phase density `p(E,theta)`, and exact escape column `NH_escape`, its ideal
observer fluence contribution is

```text
packet_weight * (D/r)^2 * p(E,theta)/q(Omega_launch)
              * exp[-NH_escape*(sigma_sca(E)+sigma_abs(E))].
```

The true incoming trajectory's survival is already represented by the
analog history; multiplying that leg by absorption again would bias the
score. `4*pi` in the scorer cancels the `1/(4*pi*q)` launch factor. The
virtual-ray escape includes both absorption and scattering extinction.
Events retain energy, sky position, arrival time, scattering order, escape
column, transmission and weight. There is no direct-source contribution or
detector response (PSF, effective area, exposure map, redistribution,
background or Poisson noise).

### Numerical outputs and uncertainty

The canonical observer cube order is `(arrival_time, energy, sky_y, sky_x)`.
Products include first-scatter, multiple-scatter, total fluence, event
counts, bin edges and solid-angle/surface-brightness information. A photon
with multiple observer scores remains **one Monte Carlo history**. For a bin
let `W_i` sum all its scores from launched packet `i`, including zeros, and
let `S=sum_i W_i`, `Q=sum_i W_i^2`; the estimated variance of total fluence is
`N/(N-1)*(Q-S^2/N)`. Schema-6 NPZ retains these per-history raw moments for
4D bins and energy-integrated time images, plus time/order cross products for
order groups 1, 2 and 3+. Accumulation across bounded chunks uses host
float64 sums/int64 counts after JAX computation (normally float32). JAX keys
are deterministically split/folded by chunk; changing chunk size need not
preserve an identical sampled realization.

Full FITS/NPZ output includes input arrays, material identifiers, source and
launch data, status counts and software provenance (`git_head`, dirty flag,
Python, NumPy, JAX). FITS contains full/first/multiple images and `HISTQ4D`,
`HISTQIMG`, `ORDSUM`, `ORDCROSS`, `EVENTIMG`, `TOTALNH` diagnostics. A
single-time-bin snapshot can export `STDIMG`, with units `ph cm^-2` per pixel;
a multi-time-bin spatial sum has no stored inter-time pixel covariance and
cannot claim a correct `STDIMG`. The flare audit checks NPZ/FITS array and
axis consistency, checksums, moments and snapshot uncertainty. Schema-5
archives lack those moments and cannot be certified retrospectively.

## Evidence and acceptance boundary

Historical fast/controlled stages are documented in
`dsh/validation/STAGE9A.md` through `STAGE9E.md` and `docs/validation.md`.
9A checks first collisions and depth laws; 9B checks foreground attenuation;
9C checks repeated scattering; 9D checks scattering with competing absorption.
The convolution check accepts two annular light curves against an exact
emission-time convolution. These guard the source/transport/scorer path but
do not replace the Stage 9F observable tests. Stage 9E's saved 2.5-million
four-cloud product is diagnostic and fails the revised readiness gate.

Stage 9F reports were produced in the user's Windows/HEASOFT environment
(reports record Python/NumPy/JAX, `git_head` and dirty status). The pinned
material digests above are the comparability invariant: the earliest shell
run records `d8836b80a0f9317042309dea69483629fb188b04`; the later spatial
and pooled spectral reports record source head
`7866e567fc724708fbf8fbf241e6ee73bb7d7278` with pending validation
files. This checkpoint commits those validation files; its new Git SHA is
expected to differ. The JSON output is kept separately from the repository,
and **is not included in this Git tag**. Each name in this table refers to
the uploaded `validation_outputs/` JSON with the same basename:

| Evidence file | Result and scope | JSON SHA-256 |
| --- | --- | --- |
| `stage9f_5p35_100k_per_seed.json` | **Passes** original 5.35-keV shell integrated/early-window first order, analog versus explicit absorption, independent-seed and terminal gates; cap 16, 300,000 packets. This earlier schema did not store the photon moments needed to certify each narrow time bin. | `a11d2f2bec8a0605ac60c37018c462fba04dbec979674cdf09ebe5b141e27810` |
| `stage9f_4p074768_100k_per_seed.json` | **Passes** the same original-shell gates at off-node 4.0747680326 keV; cap 16, 300,000 packets, with the same narrow-time-bin evidence limit. | `ae4c22791d7bd54b0fea8f540b54584245b3749c76789faa05bd5a9bf68414ee` |
| `stage9f_radial_quadrature_audit.json` | **Passes** reference-only convergence of radial quadrature for the original reports' narrow time bins; it does not recover missing simulated per-bin moments. | `116c8504bfa56c8e6cb1930b43cc1a1ed9f3acf94f4394502cc35ec9b4118c28` |
| `stage9f_5p35_cap8_100k_per_seed.json` | **Fails** numerical terminal check: 21 analog and 59 pure-scattering caps among 300,000 launched/estimator; other shell gates passed. | `84796ed733efefe4fc5d7cf53a9e8f11bd89fd92e810b29e53dce55c5b762983` |
| `stage9f_5p35_cap32_100k_per_seed.json` | **Passes** independent first-order time bins, radial/Cartesian reference agreement, order-resolved explicit-absorption comparison, precision, terminal and seed gates; zero caps/invalids among 300,000. | `a75d45e8b5338d911f6a7935496c6e06c3ac268b89f7b963d93e3e07aee60105` |
| `stage9f_spectrum_pooled.json` | **Passes** 2–4, 4–6, 6–10 keV continuous `E^-1.7` first-order time bins and orders 1, 2, 3+ after pooling disjoint seeds for outer bands; 1,620,000 / 180,000 / 1,200,000 histories, zero cap/invalid terminals. Maximum absolute comparison `z` ≈1.974. | `863e421a18b2333544bc659162bf3587192fbede7230d2dcdab262831859cf89` |
| `stage9f_absorbed_annuli_5p35.json` | **Passes** independent first-order absorbed radial-shell quadrature in 0–45, 45–90, 90–1800 arcsec over days 0–30; 300,000 packets, max `|z|=1.2964`, reference closure `8.34e-6`. | `b38b63f97c72547611a9604f1dd71b4f6b47f2102f9710f90b9ec6bcf8b06521` |
| `stage9f_heterogeneous_5p35.json` | **Passes all seven gates** for three nonuniform radial slabs and four quadrants: independent first-order quadrature, order/spatial comparison to independent host absorption scorer, 120 independently integrated escape columns (79 after higher-order scatters), zero cap/invalid among 300,000; maximum escape-column relative difference `1.215e-6`; max multiple-scatter quadrant `|z|=1.541`. | `ab0353899c64f2e0e24ef4f4959800653ce1e9453e069562e2b66dab66158e3c` |

The pooled spectral report combines the disjoint-seed baseline
`stage9f_spectrum_stratified_cap32.json` (allocation per seed
300,000/60,000/220,000) and supplement
`stage9f_spectrum_supplement.json` (240,000/2/180,000) for bands 0 and 2.
Each input may fail a precision gate in isolation; the merge verifies
matching source/material/configuration and disjoint seeds, reconstructs
per-history scalar second moments, and recomputes scalar uncertainty. It
does not reconstruct spatial pixel covariance. The shell's independent
first-order reference integrates physical incident and escape attenuation
in float64, and the repeated-order reference traces scattering without
stochastic absorption while scoring deterministic absorption along **all**
actual flights and the virtual escape. The heterogeneous host integrator
does not call the production JAX ray integrator. These references share the
frozen material arrays and imposed finite launch cones; they are not an
independent astrophysical dust model or an all-sky completeness proof.

Thresholds in the runners include five combined standard errors, at least
30 effective histories per gated component, at most 10% relative photon
error for narrow bins, and 2% reference quadrature change where relevant.
Inspect each report's `case.checks` and `all_passed`, rather than assuming
that a small `z` alone establishes precision. The original failed spectral
serialization (`numpy.bool_` under strict JSON) and the missing annulus
runner were resolved in this checkpoint; the committed runner and merge
tests prevent silently treating those failed invocations as evidence.

## Real-scene boundary and restructuring priorities

The built-in `dsh/examples.py` scene is synthetic: 160 radial by 51 by 51
angular cells, four Gaussian clouds around 2, 3.7, 5.6 and 7.8 kpc plus a
diffuse baseline, spanning 500 arcsec at 10-arcsec sampling, with a 10-kpc
source. The separate user's `200 x 500 x 500` radial/y/x, 1-arcsec,
four-cloud FITS test has cells ending at 10 kpc and a 10.5-kpc source; it
is also synthetic, **not** a measured line of sight. The PowerShell launcher
`scripts/run_four_cloud_flare_2p5m.ps1` requests 2,500,000 packets, chunk
size 512, interaction cap 16, seed 2026, a one-hour `Gamma=1.7` flare and
observer-day snapshots `[3,4)`, `[6,7)`, `[9,10)` from a 60-day archive.
These snapshots require enough events and supported 20-arcsec rebinned
cells. The old schema-5 attempt had only 369/331/317 events in the three
windows, supported-cell event fractions 22.5/13.0/7.3% against the 80%
gate, and two out-of-table energies. It did **not** pass; re-extracting its
images with the explicit diagnostic exception does not repair its moments
or sampling. No successful rerun with the new schema is claimed here.

For the restructuring chat, preserve the formulae, data-array and FITS
contracts, material digests, status meanings, source/launch weights,
per-history covariance and independent test oracles. Then run the full
regression suite and controlled comparisons on the refactored code, compare
representative outputs against this checkpoint, and verify `MAX_INTERACTIONS`
stays zero for chosen science configurations. Before a real-scene release,
audit actual FITS geometry/units, launch support, source and dust assumptions,
physical coverage, photon-history uncertainty and spatial convergence at
the desired day/pixel resolution. A continuous-spectrum spatial test,
independent external matched-material benchmark and fitted/observational
comparison remain separate scientific work.

Useful commands from the repository root:

```powershell
python -m unittest discover -v
python -m ruff check dsh scripts tests
python -m ruff format --check dsh scripts tests
python -m scripts.run_absorbed_annulus_validation --energy 5.35 --packets 100000 --chunk-size 256 --max-interactions 32 --output validation_outputs/stage9f_absorbed_annuli_5p35.json
python -m scripts.run_heterogeneous_observer_validation --energy 5.35 --packets 100000 --chunk-size 256 --max-interactions 32 --output validation_outputs/stage9f_heterogeneous_5p35.json
```

The default CLI smoke settings remain `packets=4096`, `chunk-size=256`,
`max-interactions=8`, `seed=2026`, 60 observer days in one-day bins,
`--materials v1`, representative-energy constant flare, built-in cloud and
`outputs/dsh_v1_ideal_observer.{npz,fits}`. The larger launcher is a
separate explicit workflow; do not infer production readiness from a smoke
run or from the controlled Stage 9F passes.
