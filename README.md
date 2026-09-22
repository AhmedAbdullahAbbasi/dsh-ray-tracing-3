# DSH ray tracing

JAX Monte Carlo transport for X-ray dust-scattering halos (DSHs). Version 1
models a physical point source, an angular--distance hydrogen-column cube,
energy-dependent dust scattering and photoelectric absorption, repeated
interactions, exact geometric delays, and ideal-observer halo products.

The output is physical observer fluence, not telescope counts. Effective
area, PSF, exposure maps, detector redistribution, background, and counting
noise are intentionally deferred to Version 2.

## Package layout

```text
dsh/
  geometry/   coordinates, physical cloud cubes, exact ray integrals
  physics/    NewDust scattering, TBabs absorption, JAX physics tables
  sources/    light curves, spectra, packet sampling, source launch
  transport/  direction geometry and repeated-interaction photon kernel
  observer/   peel-off scoring and weighted DSH binning
  io/         cloud FITS input and multi-extension FITS output
  data/       versioned cross-section tables and provenance
  pipeline.py end-to-end source-to-observer orchestration
  cli.py      local Version-1 runner
```

The old Cartesian toy transport, temporary one-event/isotropic transport,
legacy thin-screen adapters, Henyey--Greenstein placeholder, and notebook-only
detector/plotting modules have been removed. The production path uses only the
native angular--distance frustum and intrinsic material cross-sections.

## Installation

Create an environment and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Install the JAX build appropriate for the machine separately when GPU support
is required. The baseline dependency works on CPU.

## Run the complete V1 simulation

The compatibility command remains:

```powershell
python -m scripts.run_dsh_v1
```

After installation, the equivalent console command is:

```powershell
dsh-v1
```

The default smoke run uses 4,096 packets, a built-in four-cloud synthetic
scene, a 10 kpc point source, and the three V1 energies 3.3, 4.9, and 6.9 keV.
It writes:

- `outputs/dsh_v1_ideal_observer.npz`: complete reproducibility payload.
- `outputs/dsh_v1_ideal_observer.fits`: images, cubes, axes, inputs, and
  diagnostics.

A larger post-outburst decay run is:

```powershell
python -m scripts.run_dsh_v1 `
  --source-model exponential-decay `
  --peak-band-fluxes 0.020 0.012 0.006 `
  --baseline-band-fluxes 0 0 0 `
  --decay-time-days 25 `
  --decay-start-days 0 `
  --decay-duration-days 120 `
  --source-time-bin-days 0.25 `
  --arrival-days 240 `
  --packets 1000000 `
  --chunk-size 512 `
  --output outputs/dsh_v1_decay_1m.npz
```

The source flux convention is unabsorbed observer-equivalent photon flux in
`ph cm^-2 s^-1`. Packet weights sum to the time-integrated source fluence.

## Physical cloud input

The native cube order is `(distance, sky_y, sky_x)`. Each voxel contains the
hydrogen column increment `delta_NH` in `cm^-2`; it is not a Cartesian number
density. The cloud adapter derives

```text
n_H = delta_NH / delta_r
```

without rescaling or downsampling. Radial edges must lie between observer and
source. A physical FITS cube can replace the built-in scene:

```powershell
python -m scripts.run_dsh_v1 `
  --cloud-fits "C:\path\to\delta_nh_cube.fits" `
  --source-distance-kpc 10.5
```

The primary FITS HDU must contain the three-dimensional column-increment
array with linear `CRPIX`, `CRVAL`, and `CDELT` keywords and `BUNIT=cm-2`.

## Transport physics

For energy `E`, the local extinction coefficient is

```text
mu_total = n_H * (sigma_sca(E) + sigma_abs(E)).
```

The transport samples a target optical depth, traverses exact frustum cells,
locates the interaction by accumulated column, and selects scattering or
absorption from their opacity ratio. Dust scattering is elastic and samples
the intrinsic NewDust angular CDF. Absorption terminates the packet and
records its deposited energy. Every interaction is stored in a fixed-size
history so the kernel remains JIT-compatible.

At every scattering, the peel-off estimator aims a virtual photon at the
observer. Its weight includes the source-launch importance correction, the
normalized phase density, exact geometry, and attenuation through the
remaining event-to-observer column. Products use canonical order
`(arrival_time, energy, sky_y, sky_x)` and retain total, first-scatter,
multiple-scatter, and raw event-count cubes.

## Physics tables

The checked-in Version-1 tables share the energy grid 3.3, 4.9, and 6.9 keV:

- NewDust/xdust MRN Rayleigh--Gans/Drude intrinsic
  `d sigma_sca / d Omega`, integrated `sigma_sca`, and angular CDF.
- XSPEC TBabs version 2 `sigma_abs`, using Wilms abundances and the Verner
  atomic cross-section baseline.

Tables are validated against adjacent JSON provenance and SHA-256 metadata.
Unsupported energies are rejected rather than extrapolated. An optional
2–10 keV material pair is included on the same 130-node energy grid:

- Intrinsic RG/Drude scattering calculated within this library on 8,193
  physical scattering angles from zero to pi.
- Intrinsic XSPEC TBabs Version-2 absorption supplied from a separate
  HEASoft run (XSPEC 12.14.0, Wilms abundances).

The 3.3/4.9/6.9 keV entries agree with the frozen V1 tables. The 2–10 keV
grid records four narrow unresolved absorption-edge intervals in
`dsh/data/materials/material_grid_2_10_v2.json`. Opacity interpolation is
approximate inside each interval (under 0.0001 keV wide); the finite midpoint
scan also cannot establish that all finer features have been found.
The NewDust JSON records a comparison with historical screen files as
provenance only; no observer-space thin-screen kernel enters the simulation.

Regeneration is an offline operation:

```bash
python scripts/generate_newdust_table.py
python scripts/generate_tbabs_table.py
```

NewDust generation requires xdust, Astropy, and SciPy. TBabs generation
requires an initialized XSPEC/HEASoft environment.

To generate *separate* 2–10 keV materials on a shared adaptive axis, run from
the repository root after initializing XSPEC/HEASoft:

```bash
python -m scripts.generate_material_grid --output validation_outputs/material_grid_2_10.json
python -m scripts.generate_rg_drude_table --energy-grid validation_outputs/material_grid_2_10.json --output validation_outputs/rg_drude_2_10.npz
python -m scripts.generate_tbabs_table --energy-grid validation_outputs/material_grid_2_10.json --output validation_outputs/tbabs_2_10.npz
```

The first command checks TBabs against the runtime's log-energy,
log-cross-section interpolation. It narrows detected absorption edges to at
most 0.0001 keV and records their still approximate intervals in the grid
JSON. It checks midpoint errors between 0.1 keV initial samples; this finite
scan does not establish accuracy for unresolved fine features. XSPEC must
produce the actual absorption data; the scattering generator does not require
xdust. The new RG/Drude calculation uses exactly the V1 grain prescription,
validated against the three frozen NewDust rows and independent 2 and 10 keV
xdust evaluations. The matching NPZ and JSON files can be loaded explicitly:

```python
from dsh.physics.absorption import load_photoelectric_absorption_table
from dsh.physics.newdust import build_dust_physics_from_tables, load_newdust_scattering_table

scattering = load_newdust_scattering_table("validation_outputs/rg_drude_2_10.npz")
absorption = load_photoelectric_absorption_table("validation_outputs/tbabs_2_10.npz")
physics = build_dust_physics_from_tables(scattering, absorption)
```

The simulation defaults to the frozen three-energy material tables. To use the
bundled 130-node material pair with the existing three-band source, pass:

```bash
python -m scripts.run_dsh_v1 --materials 2-10 --packets 4096 --output outputs/dsh_2_10_materials.npz
```

This selects the new absorption, scattering opacity, and angular phase tables;
the source still emits at 3.3, 4.9, and 6.9 keV. Source sampling on a
continuous 2–10 keV spectrum is a separate change.

## Validation

Run the automated suite from the repository root:

```bash
ruff check dsh scripts tests
ruff format --check dsh scripts tests
python -m unittest discover -v
```

The tests cover source-fluence closure, coordinate conversion, native column
closure, exact ray integrals, full-kernel finite-slab interaction statistics,
table normalization and provenance, null four-momenta, elastic scattering,
absorption, repeated interactions, source-launch importance weights, peel-off
normalization, arrival delays, observer binning, chunked execution, and FITS
output.

The rigorous physics ladder is documented in
[`docs/validation.md`](docs/validation.md). Fast analytic/statistical checks
run with the normal suite. The production-path, high-statistics experiment is
run explicitly with:

```powershell
python -m scripts.run_validation_ladder --packets 1000000
```

It writes a machine-readable report under `validation_outputs/` and exits
nonzero if an acceptance criterion fails.

## Current V1 limits

- Fixed MRN Rayleigh--Gans/Drude dust model.
- Only three representative energies.
- One fixed TBabs abundance mixture per hydrogen atom.
- No spatially varying abundances, fluorescence, Compton scattering, or
  polarization.
- No telescope or detector response.
