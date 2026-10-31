# Compton extension: stage 1

Branch: `feature/compton-scattering`  
Base: `refactor/simulation-config` at `39e1790e991cc83c392962587e806210f044a267`  
Date: 31 October 2026

## Implemented scope

This checkpoint adds an independently testable Compton physics and material
foundation. **Configured simulations still use the validated elastic dust
scattering plus photoabsorption pipeline.** There is no Compton TOML switch
or production transport integration in this stage.

New numerical and material interfaces live in `dsh/physics/compton.py`:

- Unpolarized Klein–Nishina differential and total cross-sections per free
  electron; total uses a stable integrated series near the Thomson limit.
- Stationary-electron photon recoil energy and transferred kinetic energy.
  Recoil is evaluated without subtracting two nearly equal photon energies.
- Full-sphere angular tables per H nucleus, explicit composition fingerprints,
  normalized phase functions and CDFs.
- JAX evaluation and inverse-CDF sampling, usable with `jit` and `vmap`.
- Atomic NPZ archives with physical-array and metadata checksums, no pickle,
  and mandatory expected-composition checking on load.
- A free-electron producer with an explicit electron/H input and an angular
  grid accuracy check against the independently evaluated analytic total.

The existing `DustPhysicsTable` and production call signatures are unchanged.
No imports or flags are added to the production runner. Existing response
archives keep their elastic meaning.

## Scientific meaning

For energy E and physical scattering-angle cosine mu,

`E_out = E / [1 + (E/m_ec^2)(1-mu)]`.

The unpolarized free-electron differential law is

`d sigma/d Omega = (3 sigma_T / 16 pi) [k^3 + k - k^2(1-mu^2)]`,

where `k=E_out/E`. Constants use the 2022 CODATA values
`m_ec^2 = 510.99895069 keV` and `sigma_T = 6.6524587051e-25 cm^2`.

The analytical model is a **free-electron reference**, not the neutral
RefleX material. An electron/H number alone cannot specify binding,
molecular or solid-phase corrections. No RefleX-matched bound-electron
angular table is shipped or validated at this checkpoint.

The `external_incoherent` interface accepts a supplied differential law
with declared free/bound electron state, reference, producer and material
fingerprint. It currently specifies stationary-electron Compton recoil
with uniform azimuth; it cannot encode Doppler broadening, shell-dependent
outgoing-energy distributions or polarization. Accepting an external table
validates its numerical contract, not its underlying physical prescription.

Table assumptions and checks:

- The cosine grid increases from **-1 to +1** and covers the full sphere.
- Differential quantities are intrinsic `cm2 sr-1 H-1`. The total is their
  `2 pi integral dmu`, with no observer geometry factors.
- Every energy row has a positive finite angular integral. A zero channel is
  represented by not enabling that channel; zero-opacity table rows are not
  supported by this first-stage positive-density contract.
- The runtime phase is piecewise linear in cosine. Its CDF integral is
  inverted analytically within each interval, so sampling and phase
  evaluation agree even on a coarse angular grid.
- Between energy nodes, normalized angular laws are mixed linearly in
  log(E); positive totals are interpolated logarithmically. This is a declared
  table approximation, not an exact analytical KN calculation at every
  intermediate energy. Refine the energy grid for the selected tolerance.
- Out-of-support runtime requests return `supported=False` and zero values;
  invalid analytical arguments return NaNs. Callers must check support.
  Internal safe indexing does not authorize extrapolation or energy clipping.
- An optional independent total-opacity array must agree with the angular
  integral within 5e-4 relative error. Supplied external data are not rescaled.
- The free-electron producer derives its total from its angular grid and
  rejects grids whose total differs from the analytic reference by more
  than 5e-4. This gate does not establish every angular-bin error.

The tests include energies from the Thomson limit through 1000 keV. This is
validation of the free-electron foundation, not an extension of the current
production dust/absorption materials beyond their supported energy range.

## Run stage-1 acceptance

From the repository root, with the existing environment installed:

```powershell
.\.venv\Scripts\python.exe -m scripts.validate_compton_stage1
```

Or, using an activated Python environment:

```bash
python -m scripts.validate_compton_stage1
```

The runner checks the unchanged production source against the base commit,
verifies the existing frozen-evidence manifests, runs the 19 Compton tests
in separate float32 and float64 CPU processes, and runs focused dust-only
transport, observer, material, configured-run and physical-response tests.
It then writes a clearly labeled synthetic fully ionized H/He reference
material and a `compton_stage1_report.zip` beneath `outputs/compton-stage1`.
The example's electron inventory is 1.2 per H by explicit construction; it
is not an assumed inventory for an observed neutral cloud.

The new tests cover independent angular quadrature of the analytical total,
Thomson and forward/backward limits, recoil-electron mass-shell consistency,
small recoil, Monte Carlo angular CDFs, azimuthal sampling, exact inversion of
a coarse triangular phase, per-H normalization, intermediate-energy phase
normalization, invalid support, archive corruption and composition mismatch.

Local CPU acceptance on Python 3.12.14, JAX 0.4.34 and NumPy 2.3.5 passed:
19 new tests in float32, the same 19 in float64, and 52 existing regression
tests (71 distinct tests, 90 executions). All 146 frozen benchmark manifest
entries matched. The example table's maximum angular-integral difference
from the analytical free-electron total was 2.305e-7 relative. Ruff checks
passed. Windows acceptance and new external RefleX transport runs have not
been performed for this checkpoint. The saved local report was generated
before commit and records exact hashes of the tested source files.

The acceptance runner uses NumPy and JAX already declared by the project,
plus Astropy for existing regression tests (`pip install -e ".[dev]"`).
Ruff checks can be run with:

```bash
python -m ruff check dsh/physics/compton.py tests/test_compton.py scripts/validate_compton_stage1.py
```

## Next stage

1. Select and validate the neutral/bound-electron material prescription,
   including the differential law, H2 and depleted material inventory.
   Compare total opacities with the supplied RefleX export and the angular
   distribution with an independent reference. Do not attach the exported
   bound-electron total to an unrelated free-electron angular sampler.
2. Add energy to the evolving transport state and compete absorption,
   elastic dust scattering and Compton scattering at the current energy.
   Track recoil and per-channel counts while preserving the disabled path.
3. Extend peel-off scoring: channel-specific phase, observer-directed recoil
   energy and full escape opacity at that virtual energy. Event component
   counts describe the history up to that score, not later continuation.
4. Define lower-energy table support and a declared cutoff policy. Stopping
   a continuation must preserve valid contributions from the last event.
5. Add the versioned inelastic response with separate source/observer energy,
   configuration, component outputs and provenance.
6. Validate analog versus peel-off estimates, competition and energy
   bookkeeping, response folding and matched RefleX images/spectra, followed
   by the established Compton-off regressions.

The accepted 4.9-keV/NH=1e22 image comparison is not by itself validation of
the future Compton-enabled transport. Higher-column, component-separated
RefleX controls and adequate Monte Carlo covariance remain necessary.

## Primary references

- [NIST 2022 CODATA constants](https://physics.nist.gov/cuu/Constants/Table/allascii.txt).
- [Geant4 Compton kinematics and Klein–Nishina final state](https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/compton.html).
- [Geant4 low-energy incoherent scattering treatment](https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/livermore_compton.html).

The implementation uses the free-electron formula, not Geant4's empirical
per-atom total cross-section fit, whose stated low-energy domain does not
cover the few-keV cold-cloud benchmark.
