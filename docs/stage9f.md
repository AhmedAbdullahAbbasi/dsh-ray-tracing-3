# Stage 9F: absorbed observer validation

The goals are to validate the *observable* (halo fluence) with absorption and
multiple scattering, and to make photon-level Monte Carlo uncertainty available.
This is not a detector-response comparison. The 2–10 keV scattering and
absorption materials are frozen by their SHA256 digests in each report.

## Scope implemented

- The flare audit rejects inconsistent NPZ/FITS cubes, missing/mismatched energy
  and sky axes, wrong primary and component images, mismatched material arrays,
  corrupted checksums, and products lacking photon-history moments. Old
  schema-5 archives remain diagnostic and cannot pass the new production gate.
- Production bins save squared **per-photon** contributions for each 4D bin and
  energy-integrated time image. They also save a covariance matrix over all
  time bins and three scattering-order groups (1, 2, 3+). All launched photons,
  including those with zero observer scores, count toward the denominator.
  Sums of many chunks accumulate in host float64/int64. The NPZ schema is 6.
  NPZ/FITS files record the generating git head, worktree-dirty flag, and
  Python/NumPy/JAX versions separately from the checkout that audits them.
- Each one-bin, one-day extracted snapshot now has a `STDIMG` extension:
  `sqrt(N/(N-1)*(Q-S*S/N))` per sky pixel. Multiple time bins combined into
  one snapshot do **not** have a spatial covariance in this schema: the
  extractor marks uncertainty unavailable and the readiness audit rejects that
  snapshot. Integrated multi-time and multi-order fluence uncertainties can be
  computed from `time_order_fluence_cross`.
- Source energies near photon index Γ=1 use stable `log1p`/`expm1` sampling;
  cloud FITS local axes require supported CTYPE/CUNIT and reject rotations or
  celestial WCS. Compatible angular/distance units are converted.
- `scripts.run_absorbed_observer_validation` uses a controlled, uniform 4–5 kpc
  spherical shell and centered 10-kpc source with a stated finite launch cone.
  A double-precision slope/depth quadrature independently estimates absolute
  first-order observer fluence with extinction on both legs. Separately, a
  scattering-only transport and independent host geometry/phase scorer apply
  deterministic absorption through **all preceding actual flights** and the
  virtual escape ray. This is compared with production analog absorption for
  first, second, and 3+ scattering orders. The host scorer shares the frozen
  material arrays, but not the production observer-scoring or ray-traversal
  implementation. Separate PRNG streams are used for the two estimators.

## Local PowerShell commands

Run in the repository root *after applying the patch*:

```powershell
python -m unittest discover -v
python -m ruff check dsh scripts tests
python -m ruff format --check dsh scripts tests
python -m scripts.run_absorbed_observer_validation --packets 20000 --chunk-size 256 --output validation_outputs/stage9f_absorbed_observer_5p35.json
python -m scripts.run_absorbed_observer_validation --energy 4.0747680326 --packets 20000 --chunk-size 256 --output validation_outputs/stage9f_absorbed_observer_offnode.json
```

A nonzero validation-runner exit code with a written JSON file usually means
the comparison was **inconclusive** because effective photon histories or
precision fell below the predefined checks. Inspect the failed checks and
per-order uncertainty before increasing `--packets` per seed. Do not loosen
the sigma threshold, effective-count threshold, or precision goals until a
revised design and new run are agreed upon. Seeds default to `[912,319,141]`.
The default screening precision is 5% integrated and 10% for selected order
groups. For a stricter scientific result, target the 1% / 5% levels proposed
in the audit by increasing photon histories and verifying the estimator's
variance across independent seeds. The default run is a local validation
experiment, not a four-cloud 2.5-million-packet flare.

The saved 2.5-million-photon schema-5 flare predates these changes; its
uncertainty moments cannot be reconstructed from its aggregate event counts.
Do not promote to the `dsh` branch or repeat the expensive flare until the
controlled benchmark is sufficiently powered, the full product gate has been
tested with new output, and the actual cloud's FITS header/units are confirmed.

## Remaining 9F validation

The current controlled benchmark is monoenergetic at an on-node or specified
off-node energy. Extend its independent quadrature to the continuous Γ=1.7
source and band/time/annulus comparisons. Test convergence with interaction
caps 8, 16 and 32 and a small heterogeneous cloud. The quadrature currently
gates integrated first order and the broad 0–2 day window; its individual
short time-bin values need tighter angular-convergence work before they can
serve as absolute references. The standalone radial audit now addresses this
reference-only limitation in the symmetric shell: it integrates the exact
azimuthal coverage of the rectangular launch cone and splits the radial domain
at time-bin crossings of both shell surfaces. It reuses the saved reports, so
it does not repeat the photon transport:

```powershell
python -m scripts.audit_stage9f_quadrature validation_outputs/stage9f_5p35_100k_per_seed.json validation_outputs/stage9f_4p074768_100k_per_seed.json --output validation_outputs/stage9f_radial_quadrature_audit.json
```

The radial audit checks convergence of **each reference time bin**, original
material digests, integrated first-order agreement, and broad 0–2 day
agreement. Original Stage 9F reports do not store per-bin analog moments: the
new audit therefore cannot validate the simulated narrow-bin fluence, just its
independent reference. The radial simplification applies only to the centered,
rotationally symmetric shell and source used in Stage 9F, not a real cloud.

Next assess a launch proposal with demonstrably complete support. Re-run the
requested one-day four-cloud images locally and judge photon-level
uncertainties at the actual desired image resolution.
