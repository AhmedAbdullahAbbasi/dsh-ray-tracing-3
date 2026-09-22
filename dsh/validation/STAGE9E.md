# Stage 9E: production flare readiness audit

Run this **before** spending time on another 2.5-million-packet run. It reads
an existing saved NPZ and writes a diagnostic report; it never launches
transport or changes the simulation archive:

~~~powershell
python -m scripts.audit_four_cloud_flare `
  --input-npz outputs/flare_2p5m_four_cloud_2_10/flare_2p5m_full.npz `
  --output validation_outputs/stage9e_existing_flare_audit.json
~~~

A nonzero exit is expected for the older archive with two invalid-energy
packets. The report is still written and identifies the independent
deficiencies of the three requested one-day snapshots. The diagnostic
extraction option from patch 0003 does not make that archive a successful
production run. The NPZ-only command is a diagnostic screen; a successful
`all_passed` report also requires the full FITS and all three snapshot FITS.

For a fresh run with the complete FITS and extracted snapshots, give all
inputs to check that the FITS arrays, status counts, time edges, metadata,
and snapshot images agree with the NPZ:

~~~powershell
python -m scripts.audit_four_cloud_flare `
  --input-npz outputs/flare_2p5m_four_cloud_2_10/flare_2p5m_full.npz `
  --input-fits outputs/flare_2p5m_four_cloud_2_10/flare_2p5m_full.fits `
  --snapshot-dir outputs/flare_2p5m_four_cloud_2_10/snapshots `
  --output validation_outputs/stage9e_full_flare_audit.json
~~~

The PowerShell flare workflow now runs this gate after creating the three
snapshots and only prints `Success` if it passes. It requires zero invalid
energy or interaction-cap terminals, exact packet/event accounting, matching
2–10 keV material checksums, a continuous hard-state source, and no sky or
energy scores outside their bins. Out-of-range **arrival** events may occur,
but their total fluence must be less than 1% of scored fluence; in particular,
the old run's 6,755 late events carry only ~0.01% of the scored fluence.

## Diagnostic spatial screen

For each one-day image at days 3, 6 and 9, the default gate requires at least
2,000 binned events, plus at least 80% of those events in 20-native-pixel
cells with ten or more events. On the current 1-arcsecond pixels these are
20-arcsecond cells. Ten equally weighted events give a best-case Poisson
relative error of about 32% per cell. Unequal peel-off weights increase noise,
so the event-count screen is a **necessary but insufficient** condition for
credible morphology; the archive does not store per-bin sums of squared
weights. The numeric screen is configurable via `--min-snapshot-events`,
`--coarse-factor`, `--min-supported-cell-events`, and
`--min-supported-event-fraction`; do not lower the defaults just to pass.

The older 2.5-million-packet run has only 369, 331 and 317 binned events in
the requested days. Only 22.5%, 13.0% and 7.3%, respectively, reside in
20-arcsecond cells with ten or more events. At least seven times the packet
count (about 17.5 million), *assuming linear count scaling*, would be needed
to meet the weakest count-only threshold. Even that may fail spatial support
or weighted-noise requirements. The same saved run also has two invalid
energy packets; the energy-bound fix needs a new run to verify.

The FITS primary images are ideal-observer photon fluence in `ph cm-2` per
sky pixel. They are not detected Chandra counts: no effective area, exposure,
PSF, detector redistribution, or background has been applied. A native
1-arcsecond image is therefore an especially misleading visualization when
nearly all nonzero pixels have one simulated event.

Passing this screen does not establish convergence across independent seeds,
instrument response, or the astrophysical correctness of the cloud geometry.
It prevents the mere existence of three FITS images from being treated as
evidence that their spatial structure is resolved. Do not push to `dsh` based
only on the older run or a report that fails this gate.
