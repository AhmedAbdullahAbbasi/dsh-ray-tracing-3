# Stage 9F: asymmetric-cloud observer control

Three separated radial slabs carry different hydrogen columns in each of
four sky quadrants; vacuum cells separate the slabs. A double-precision
quadrature checks absolute first-order quadrant fluence using explicit source
chords and radial observer columns. A separately seeded scattering-only
transport is scored on the host with absorption on every actual flight and
the virtual observer flight. It compares first, second, third-plus, total,
and spatial multiple-scattering fluence against absorbing production transport.
The host column integrator never calls the production JAX ray integrator.

The runner spot-checks production escape columns against host intersections,
checks photon scores against each binned production image, includes histories
with zero observer score in the photon covariance, and rejects numerical
interaction-cap or invalid statuses.

In the repository root of the validated checkpoint:

```powershell
python -m unittest tests.test_validation_heterogeneous_observer -v
python -m scripts.run_heterogeneous_observer_validation --energy 5.35 --packets 100000 --chunk-size 256 --max-interactions 32 --output validation_outputs/stage9f_heterogeneous_5p35.json
```

The report is written even when a statistical gate fails. Inspect
`case.checks`, `case.first_order_quadrants`, `case.order_comparisons`, and
`case.multiple_scatter_quadrants` to identify the gate; add histories with new
seeds or increase packets per seed when uncertainty is too large. The scene
checks fluence **conditional on a finite launch cone** (slopes ±0.0012),
which does not cover every point in the full ±1800-arcsec frustum. It is a
controlled geometry and does not establish complete launch support, the
continuous-spectrum spatial image, or the actual four-cloud FITS inputs.
