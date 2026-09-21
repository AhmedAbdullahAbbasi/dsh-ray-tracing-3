# DSH simulation validation ladder

This branch separates fast regression tests from high-statistics Monte Carlo
experiments. The fast tests run on every code change. The convergence runner
uses the production launch, voxel-transport, peel-off, and scoring path and is
run deliberately before a physics release.

The coordinate convention throughout is

```text
x = observer-to-dust distance / observer-to-source distance.
```

## Coverage

| Step | Fast deterministic/statistical test | Local convergence experiment | Current status |
|---|---|---|---|
| 1. Delay geometry | Exact Euclidean broken-path delay at `x=0.1, 0.5, 0.9`; inversion of the small-angle ring relation | Reports the maximum event-by-event delay residual from the production scorer | Implemented |
| 2. Cross section | MRN phase function is forward-peaked, normalized, and matches the intrinsic table | Reports containment angles at all table energies | Implemented for the MRN-integrated table |
| 3. Flux conservation | Low-`tau` analog scattering fraction is compared with `1-exp(-tau)` | Compares analog collisions and integrated peel-off fluence with the input fluence times `tau_sca` | Implemented |
| 4. Energy scaling | Fits the slopes of median scattering angle and `sigma_sca` against energy | Holds the screen's hydrogen column fixed and measures halo radii and scattering fluence at 3.3, 4.9, and 6.9 keV | Implemented |
| 5. Screen thickness | Checks analytic inner/outer ring bounds and the zero-thickness limit | Sweeps increasing thickness and requires monotonically increasing delay-width around the central-screen relation | Implemented |
| 6. Symmetry | Scores a deterministic ring at 64 azimuths and checks equal delays/weights | Measures weighted azimuthal Fourier modes inside a circular aperture | Implemented |
| 7. Convolution | Scores the same three-angle scattering history for an impulse and a three-bin top-hat source, then bins and compares with discrete convolution | A continuous light-curve convergence case remains for a later stage | Implemented for scorer and binning |
| 8. Literature | None | Requires a frozen external reference case with matched dust, geometry, energy, and normalization conventions | Pending |

## Geometry reference

For source distance `D`, event radius `r=xD`, and observed angle `theta`, the
exact excess path is

```text
sqrt((D-r)^2 + 4 D r sin(theta/2)^2) - (D-r).
```

The implementation evaluates a rationalized equivalent to avoid cancellation
at arcsecond angles. In the small-angle limit,

```text
Delta t = (D/c) x theta^2 / (2 (1-x)).
```

These references live in `dsh.validation.analytic`; they are never called by
the transport or scorer.

## Cross-section scope

The Version-1 NewDust table stores an MRN-size-distribution-integrated
`d sigma/d Omega`. It has no grain-radius axis. The present test therefore
validates the actual MRN physics used by the simulation, including
normalization, forward peaking, containment angles, and energy scaling.

A literal single-grain test requires a new generator product with at least
`(energy, grain_radius, scattering_angle)` axes. Until that exists, a test
claiming to validate `d sigma/d Omega(E,a,theta)` would be false precision.

## Flux-conservation interpretation

The fast transport checkpoint disables absorption and uses one low-opacity
slab. The probability of at least one scattering is

```text
P_sca = 1 - exp(-tau_sca).
```

The high-statistics experiment assigns unit source fluence across all packets.
It checks both the analog scattering fraction and the sum of peel-off observer
weights. At `tau_sca=0.01`, the distinction between `tau_sca`, at-least-one
scattering, and exactly-one-scattering probabilities is below one percent but
remains recorded explicitly in the JSON output.

The energy sweep fixes the screen's hydrogen column using the 3.3-keV target
optical depth. Each higher-energy optical depth follows its own tabulated
scattering cross section; the fluence ratios are therefore allowed to change
with energy.

The uniform screen has a finite square field. The report therefore records the
phase probability enclosed by an inscribed circular aperture; flux outside the
simulated field must not be misdiagnosed as a normalization error.

The validation runner uses a three-cone mixture for source launch directions:
80 percent in a narrow halo cone, 15 percent in a broader cone, and 5 percent
in the full screen cone. Each photon is weighted by the *complete mixture PDF*,
including all overlapping cones. This preserves the isotropic source measure
and full-field support while reducing the variance of forward-peaked peel-off
events. Production source launching and observer scoring are unchanged. The
report gives both an estimated relative Monte Carlo standard error for the
total scored fluence and the effective sample size in the symmetry aperture.

## Fast checks

Run the validation tests alone:

```powershell
python -m unittest tests.test_validation -v
```

Run the entire regression suite:

```powershell
python -m unittest discover -v
python -m ruff check dsh scripts tests
python -m ruff format --check dsh scripts tests
```

## High-statistics local run

The default experiment transports one million packets at each of the three V1
energies and 300,000 packets at each of three screen thicknesses:

```powershell
python -m scripts.run_validation_ladder `
  --packets 1000000 `
  --chunk-size 100000 `
  --target-tau 0.01 `
  --screen-fraction 0.5 `
  --screen-thickness-kpc 0.01 `
  --thickness-sweep-kpc 0.001 0.01 0.1 `
  --sweep-packets 300000 `
  --output validation_outputs/rigorous_dsh_validation.json
```

The command exits nonzero when any acceptance check fails. Its JSON report
contains configuration, raw metrics, pass/fail flags, and explicit scope
limitations. Do not obtain a pass by trying new seeds. Increase the packet
count when a result is statistically marginal; later, add an ensemble-of-seeds
test before freezing release tolerances.

Current acceptance thresholds are:

- maximum relative delay error below `1e-3`;
- analog scattering residual below five binomial standard deviations;
- integrated peel-off fluence within 10 percent of `fluence * tau_sca`;
- simulated radius slope within `0.03` of `E^-1`;
- scattering-opacity slope within `0.03` of `E^-2` for the current RG table;
- simulated fluence-versus-energy slope within `0.20` of that tabulated slope;
- estimated observer-fluence Monte Carlo error below 5 percent and at least
  400 effective weighted events in the symmetry aperture at each energy;
- azimuthal harmonics below `5/sqrt(N_eff)`;
- monotonically increasing ring/delay width with increasing screen thickness.

The 10-percent peel-off threshold is deliberately provisional until the first
high-statistics run establishes the estimator variance. Tightening it must be
based on repeated-seed convergence, not on one favorable realization.

## Literature benchmark boundary

The first external benchmark should freeze a machine-readable reference case,
not merely compare figures by eye. It must specify the source fluence and
distance, screen distance/thickness and column, dust population, differential
cross section, absorption treatment, energy definition, angular/time bins,
scattering-order convention, and whether an instrument response is applied.

Recommended sequence:

1. Reproduce NewDust/xdust directly with the same MRN RG/Drude configuration;
   this isolates table ingestion and normalization.
2. Reproduce the expanding-ring delay relation used by
   [Vianello, Tiengo & Mereghetti (2007)](https://arxiv.org/abs/0707.2343).
3. Treat [Predehl & Schmitt (1995)](https://ui.adsabs.harvard.edu/abs/1995A%26A...293..889P/abstract)
   as an observational/empirical comparison only after matching its dust and
   band conventions; it is not a drop-in unit test for the present MRN table.

No literature comparison is marked as passed until its reference inputs and
expected outputs are committed under a versioned validation-data schema.
