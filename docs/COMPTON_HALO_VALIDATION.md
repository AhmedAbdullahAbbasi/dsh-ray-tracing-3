# Scientific validation: fixed 4.9-keV RefleX Compton pair

This benchmark compares accepted coupled transport with the supplied 4.9-keV,
NH=1e22 cm^-2 RefleX camera images. It retains accepted geometry, aperture
treatment and absolute cone/camera normalization. No physical model parameter
or image amplitude is fitted.

## Matched experiment and material scope

The scene is a 30-pc-radius, 30-pc-thick disc at 5 kpc, with the source at
10 kpc. A 300-arcsec square image uses 5-arcsec pixels. RefleX launches 1e8
photons inside a 1-degree cone and combines 10,000 0.1-pc camera apertures at
0.2-pc spacing. The finite circular aperture is sampled at each event using
its observer distance, following the accepted comparison convention.

The reference manifest hashes raw on images, parameters and completion logs,
and the existing frozen off files. Only the Compton switch differs among
active parameter lines. `DUSTSCAT > 0` selects dust-only plus mixed histories.
The all-minus-dust central remainder is a direct-dominated proxy, not a
separately tagged direct image in the on run.

DSH uses the declared WD01/D03 C50/Si90 budget, residual gas/H2/grain absorption
and accepted cold Compton model, with explicit grain policy (default
`atomic_incoherent`). RefleX uses a different Fe dust fraction, effective
opacity scale and angular sampler. This tests observables under the declared
models, not exact material equivalence. Old benchmark files and manifests
are unchanged.

## Paired estimator

The off control uses the same six-stream kernel with positive 1e-35 cm2/H
Compton opacity (nominal tau=1e-13). This respects strictly positive table/log
interpolation and preserves the elastic distribution to a negligible declared
error. Any realized control Compton event fails completion. This does not
reproduce the legacy kernel's exact random stream.

On/off histories share launch, transport and aperture random variates. Entire
annular vectors are grouped by source photon before forming cross moments;
zero scorers count in N. Stored moments retain cross-component and paired
on/off covariance. Ratio errors include the covariance term. Weighted DSH
pixels are never treated as independent Poisson count realizations.

Three independent production seeds have a fixed final budget, with distinct
pilot seeds. Prefix estimates reuse histories and are correlated. They show
precision growth without a significance-based stopping rule. Each absolute
RefleX comparison uses model MC covariance and fixed-N multinomial reference
counting variance, with independent Gaussian finite-MC errors in predictive
calibration. No joint significance is assigned to the shared-seed RefleX on/off
difference, whose covariance is unavailable.

## Commands and checkpointing

Use the project environment with JAX, NumPy, SciPy, Astropy and Matplotlib:

```powershell
python -m unittest tests.test_compton_halo_validation -v
python -m scripts.validate_compton_halo --pilot --output-directory outputs/compton-halo-pilot
python -m scripts.validate_compton_halo --output-directory outputs/compton-halo-validation
```

The protocol is `configs/compton/halo_validation.json`. The default final budget
is 2,097,152 histories per seed: 6,291,456 paired source histories and twice
that many transport realizations. Batch numerics default to float32; accumulated
moments and inference use float64. `--precision float64` selects a separate
collection, not a way to reuse float32 checkpoints. Allow several minutes or
longer depending on CPU; progress is saved regularly.

Continue the same collection after interruption:

```powershell
python -m scripts.validate_compton_halo --resume --output-directory outputs/compton-halo-validation
```

Input/runtime/material fingerprints, embedded checkpoint seeds and history
counts must match. Sums are per history, so extending the count does not rescale
earlier photons. For a new scientific study after a pilot or insufficient
budget, declare a fixed final count and use fresh production seeds.

Advanced workers can run `--prepare-only`, then disjoint `--seed-only SEED
--resume` invocations sharing one directory. Each writes its own checkpoints.
Run ordinary `--resume` after all workers finish to aggregate and plot. Resource
scheduling is external to the numerical collector.

## Gates and outputs

Numerical completion requires valid terminal histories, no guard/cutoff/support
losses, no unsupported observer rays, no control Compton events and energy
closure below 2e-5 keV per photon. Component closure is checked separately.
Precision requires at least three seeds, aperture MC error <=0.6%, paired ratio
SE <=0.002 and annular model errors <=half the reference counting error.
Seed consistency, a near-forward dust-only survival control and condition-wise
RefleX agreement use their declared thresholds.

Pilot exit 0 certifies numerical completion only. Scientific exit 0 requires
all gates; exit 3 preserves an incomplete/inconclusive report. A failed scientific
gate can mean inadequate sampling or a material-model difference, not necessarily
a transport defect. No gate is silently loosened to publish a success.

Outputs include Markdown/JSON reports, per-seed raw moments/checkpoints, joint
annular covariance, component energy/time summaries, CSV radial profiles,
expected-count FITS maps, and PNG/PDF figures. There is no detector response or
Poisson realization. The archive lacks corresponding RefleX spectra/time products,
so those observables are not externally validated. Weak components can remain
poorly sampled even when the dust-selected halo meets its precision requirements.

The reference's roughly 1.3% marginal halo counting error limits sub-percent
equivalence claims. Higher columns, other energies, separated RefleX components
and energy/time products need new reference runs. The cold prescription retains
its accepted 2--10 keV support, stationary-electron recoil and grain approximation.

## Completed production experiment (2026-11-01)

The three fixed production seeds completed 6,291,456 paired histories. All
numerical and declared precision gates passed. Predicted dust-selected counts
were 5979.18 off and 5950.89 on, compared with 5976 and 5981 RefleX counts.
The paired modeled halo reduction was 0.4731% with 0.1458 percentage-point
MC standard error. Component closure passed; the maximum energy-closure
error was 9.54e-7 keV and no unsupported observer contributions occurred.

External equivalence remains unresolved. Predictive radial p-values were
0.0272 off and 0.0030 on. These pass the declared 0.001 gate but indicate
radial tension at the conventional 5% level. No amplitude was fitted and
no threshold was changed. The outer 150--215 arcsec bin covers only square-field
corners; it is not corrected to a complete circular annulus.

The next control should match exported RefleX opacity and angular inputs to
separate material/phase differences from implementation differences. These
residuals alone do not identify their cause. Higher columns, other energies,
and separated component/energy/time reference products follow that control.
The complete report archive retains figures, arrays and per-seed checkpoints.
