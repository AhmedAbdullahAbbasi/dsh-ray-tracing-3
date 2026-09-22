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
| 1. Delay geometry | Exact Euclidean broken-path delay at `x=0.1, 0.5, 0.9`; inversion of the small-angle ring relation | Reports event delay residuals and tests binned 3.3-keV ring images in 1–4-day time slices at `x=0.1, 0.5, 0.9` | Implemented for all three screen distances |
| 2. Cross section | MRN phase function is forward-peaked, normalized, and matches the intrinsic table | Reports containment angles at all table energies | Implemented for the MRN-integrated table |
| 3. Flux conservation | Low-`tau` analog scattering fraction is compared with `1-exp(-tau)` | Compares analog collisions and integrated peel-off fluence with the input fluence times `tau_sca` | Implemented |
| 4. Energy scaling | Fits the slopes of median scattering angle and `sigma_sca` against energy | Holds the screen's hydrogen column fixed and measures halo radii and scattering fluence at 3.3, 4.9, and 6.9 keV; reports the weighted-radius slope's Monte Carlo error | Implemented |
| 5. Screen thickness | Checks analytic inner/outer ring bounds and the zero-thickness limit | Sweeps increasing thickness and requires monotonically increasing delay-width around the central-screen relation | Implemented |
| 6. Symmetry | Scores a deterministic ring at 64 azimuths and checks equal delays/weights | Measures weighted azimuthal Fourier modes inside a circular aperture | Implemented |
| 7. Convolution | Scores a three-angle impulse and top-hat source; also checks exact fractional-bin convolution and its conditional Monte Carlo variance | Runs a tabulated post-peak decay through the real source sampler, voxel transport, scorer, and observer bins; compares two annular light curves with exact continuous-time impulse convolution | Accepted rerun: rigorous_flare_convolution_fixed(1).json; 12 usable time bins in each of two annuli |
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

The runner saves **simulated ideal-observer images** from actual scored
3.3-keV photons at `x=0.1, 0.5, 0.9` as time-by-y-by-x NPZ cubes. It reuses
the main energy-run image when its screen fraction matches a sweep fraction;
otherwise each extra screen uses `--image-sweep-packets` packets. Each
128-by-128 sky crop covers its predicted 4-day ring radius with 30 percent
margin where the cloud field allows it. Images use 0.25-day time bins over
0–8 days. Twelve time slices from days 1–4 are eligible for each image check.
Each selected image must have its fluence-weighted median ring radius inside
the analytic bounds set by its time-bin edges, allowing half a pixel diagonal
for sampling. At least six common slices must show strictly decreasing ring
radius with increasing screen distance. The JSON report stores the measured
radius and analytic midpoint prediction for every checked slice. Image-fluence
checks explicitly include events outside each sky/time crop; the cropped image
alone should not equal the full observer fluence. No instrument response or
PSF is applied.

The energy-width slope has a Monte Carlo standard error inferred from the
weighted median's local 40–60-percentile span and effective event count. A
three-sigma consistency check is separate from the precision requirement of
`0.03` on that slope. A consistent but imprecise run therefore remains
inconclusive instead of being misclassified as wrong physics.

## Fast checks

Run the validation tests alone:

```powershell
python -m unittest tests.test_validation tests.test_validation_launch tests.test_validation_image -v
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
  --image-screen-fractions 0.1 0.5 0.9 `
  --image-sweep-packets 300000 `
  --output validation_outputs/rigorous_dsh_validation.json
```

The command exits nonzero when any acceptance check fails. Its JSON report
contains configuration, raw metrics, pass/fail flags, and explicit scope
limitations. Do not obtain a pass by trying new seeds. Increase the packet
count when a result is statistically marginal; later, add an ensemble-of-seeds
test before freezing release tolerances.

The main screen's 3.3-keV binned image is saved beside the JSON report with
suffix `_3p3_image.npz`; other screen images use `_x0p1_3p3_image.npz` and
`_x0p9_3p3_image.npz` with the default fractions. Each contains
`fluence_time_y_x`, `event_count_time_y_x`, and the corresponding sky and
arrival-time bin edges. If Astropy is installed, the runner also writes FITS
versions with an integrated image, time cube, event-count cube, and exact
time-bin table.

Current acceptance thresholds are:

- maximum relative delay error below `1e-3`;
- ring medians within each screen's analytic time-bin bounds (plus half a pixel
  diagonal), with at least six shared bins ordered from near to far screen;
- analog scattering residual below five binomial standard deviations;
- integrated peel-off fluence within 10 percent of `fluence * tau_sca`;
- simulated radius slope within three estimated Monte Carlo standard errors
  of the table's slope, **and** slope standard error below `0.03`;
- scattering-opacity slope within `0.03` of `E^-2` for the current RG table;
- simulated fluence-versus-energy slope within `0.20` of that tabulated slope;
- estimated observer-fluence Monte Carlo error below 5 percent and at least
  400 effective weighted events in the symmetry aperture at each energy;
- azimuthal harmonics below `5/sqrt(N_eff)`;
- monotonically increasing ring/delay width with increasing screen thickness.

The 10-percent peel-off threshold is deliberately provisional until the first
high-statistics run establishes the estimator variance. Tightening it must be
based on repeated-seed convergence, not on one favorable realization.

The near/far screens can have fewer scored events in late time bins than the
midpoint screen. When only `image_screen_geometry` and
`image_screen_fraction_order` fail because a screen has fewer than six time
bins with at least 20 events, increase the image packet count. The completed
energy and thickness runs can be reused without simulating them again:

```powershell
python -m scripts.rerun_screen_images `
  --previous-report validation_outputs/rigorous_dsh_validation_screen_sweep.json `
  --packets 1000000 `
  --chunk-size 100000 `
  --output validation_outputs/rigorous_dsh_validation_screen_sweep_refined.json
```

This reuses the saved midpoint image, independently resamples the near and
far screens with a different random seed, and recalculates only the image
checks. The JSON records the original report and packet counts for each
screen. The command rejects a prior report with failures elsewhere in the
ladder. Do not treat an image with fewer than six usable common time bins as
a passed geometry check, even when its measured radii are correct.

## Flare-convolution experiment

Run the independent temporal checkpoint after the screen-distance images pass:

```powershell
python -m unittest tests.test_validation_convolution -v
python -m scripts.run_flare_convolution_validation `
  --packets 1000000 `
  --chunk-size 100000 `
  --output validation_outputs/rigorous_flare_convolution.json
```

The default case uses the same 10-kpc, `x=0.5`, 3.3-keV, `tau_sca=0.01`
uniform screen as the image experiment. It generates a 2-day post-peak
exponential with 0.8-day decay time. Its source cells are 0.125 day wide;
the flux in each cell is the exact exponential average, and the production
source sampler draws emission times uniformly within that cell. The report
records the maximum difference between this tabulated CDF and the true
continuous exponential CDF.

Every source packet takes the normal path through the importance-weighted
source launch, native voxel transport, and peel-off scorer. The real event
times are binned with the production observer binning function in two annuli.
For the reference, each *same physical scattering history* is treated as an
impulse at emission time zero. Its actual geometric delay and observer weight
are convolved with the source CDF by integrating the emission time exactly
between arrival-bin edges. A 5-sigma threshold uses the conditional variance
of the independently sampled emission time of every packet; at least six
time bins per annulus must have an expected signal five times larger than
that standard error. This paired test isolates source timing and binning from
transport sampling noise. It is not a second independent transport run.

The JSON report and adjacent `_lightcurves.npz` contain the impulse,
convolved, and sampled flare curves, plus conditional uncertainties, source
time bins, and annulus definitions. Absorption and higher scattering orders
are disabled here so that a failure can be assigned to the time-domain path.

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
