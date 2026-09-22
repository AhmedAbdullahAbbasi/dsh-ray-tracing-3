# Stage 9A: absorption and competing first interactions

Run from the repository root in the user's current checkout after applying the
patch. Confirm the checkout contains the audited validation code, or compare
file contents if commit IDs differ after applying earlier patches.

```powershell
python -m unittest discover -q
python -m ruff check dsh scripts tests
python -m scripts.run_first_event_validation `
  --packets-per-ray 20000 `
  --chunk-size 1000 `
  --output validation_outputs/stage9a_first_events.json
```

The full default run uses the actual 3.3, 4.9, and 6.9 keV NewDust and TBabs
coefficients, with central columns fixed by target 3.3-keV scattering optical
depths 0, 0.1, 0.3, and 1.0. For nonzero columns it runs pure absorption,
pure scattering, and both processes. Zero-opacity controls include zero
column and a finite column with both cross sections deliberately disabled.
Pure-process runs are controlled code checks, not alternative science models.

Each case uses an on-axis source ray and an oblique ray at 0.2 rad. An
independent NumPy sphere-chord calculation gives the exact path length through
a uniform-density 4–5 kpc radial shell. The report compares the production
ray column to this result, then tests no event, first scattering, and first
absorption against competing exponential risks. The no-event category also
tests direct-beam transmission `exp(-NH*(sigma_sca+sigma_abs))`; it does not
insert a direct source into the halo image. Event-position checks compare the
conditional cumulative distribution at 25%, 50%, and 75% of the shell path.

Packets stop after one transport iteration. A first scatter correctly has
`MAX_INTERACTIONS`; its recorded scattering event remains valid. Do not
interpret this cap as a physical terminal fraction. The report checks event
records, terminal category accounting, deposited energy, outgoing momentum,
and conservation of initial photon energy in the analog history.

`all_passed` applies a five-standard-deviation threshold separately to each
reported binomial marginal. These categories and position thresholds are
correlated, so the fields should be inspected individually; a pass does not
give one calibrated familywise p-value. The small default test suite and any
few-hundred-packet exploratory runs are smoke checks. Run with the default
20,000 packets per ray before accepting Stage 9A, inspect rare-category
counts, and repeat at a higher count or independent seed if a marginal has
too little statistical power or fails narrowly. The output stores its exact
code version, table hashes, seed, packet configuration, and JAX environment.

This gate isolates first-event transport. Stage 9B must separately validate
observer escape attenuation on controlled scattering histories. Stage 9C/D
then need repeated-event and absorption-enabled order-resolved benchmarks;
the current first-event probabilities are not terminal high-opacity laws.
