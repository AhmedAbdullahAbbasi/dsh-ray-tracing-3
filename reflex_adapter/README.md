# RefleX material adapter — tested 16 October 2026

The supplied RefleX 3.0 run and the existing DSH MRN model do not use the same
material coefficients. This optional package supplies RefleX-derived scattering
and absorption tables at **3.3, 4.9 and 6.9 keV**, a table producer, a loader, and
a driver patch. It preserves the existing default material and NewDust benchmarks.
The absorbed, all-order 4.9-keV radial/aperture comparison now passes against the supplied RefleX images. Other energies and stronger multiple-scattering regimes remain outside this result.

## Quantitative diagnosis

Cross-sections below are intrinsic **cm² per hydrogen nucleus**. The DSH values
use the integral of the existing differential table, including its documented
1.044633 ratio to NewDust's separately reported total. The RefleX values reproduce
the supplied executable's runtime lookup, to the precision of its float32 DUMP.

| Energy (keV) | Existing DSH scattering | RefleX scattering | DSH / RefleX | RefleX absorption |
| --- | --- | --- | --- | --- |
| 3.3 | 1.25279e-23 | 4.45662e-24 | 2.811 | 7.60283e-24 |
| 4.9 | 5.68216e-24 | 1.97720e-24 | 2.874 | 2.48326e-24 |
| 6.9 | 2.86555e-24 | 9.62067e-25 | 2.979 | 9.30269e-25 |

Thus the reported factor of 2–3 has a direct counterpart in the scattering
opacity, before comparing image estimators. It is not a universal correction
factor for every radial bin: the angular distributions differ too.

The published RefleX dust prescription uses WD01 graphite and olivine
populations with Draine optical properties and Mie/ADT scattering calculations.
Its `DUST` parameter means the fraction of iron in grains, rather than a dust
mass fraction. The supplied run uses Wilms abundances and `DUST 0.691`.
Reference: [Ricci & Paltani (2023), Sections 2–3](https://arxiv.org/html/2301.10268v3).

## What was actually implemented

The executable's supported `DUMP` command exports gas photoelectric opacity,
total dust opacity, dust albedo and an angular scale. In this binary, the
`Dust_angular` function samples

    alpha = theta50 * sqrt(u / (1-u)),     azimuth = 2*pi*v

for independent uniform variates. Matching this implementation is more precise
for this comparison than substituting a generic WD01/Mie calculation with
unspecified size parameters or normalization. The angular scales in the exported
table are 111.000, 74.7624 and 53.0953 arcsec. They describe physical deflection;
for a midpoint thin screen the corresponding halo scale is approximately half.

The producer computes:

    sigma_sca = DUST * ALBEDO * 1e-18
    sigma_abs = (PHOTO + DUST * (1-ALBEDO)) * 1e-18

The conversion is from megabarns to cm²/H. DUMP has already scaled its DUST
column by the selected composition/depletion. Multiplying by 0.691 again would
be an error. The absorption includes depleted gas plus grain absorption;
**do not add the existing full TBabs opacity to it**.

The analytic angle density is converted to a per-steradian differential
cross-section using `2*pi*sin(alpha)`, so its normalization matches the DSH
transport and observer contract. The producer integrates and normalizes the
differential law, then builds its CDF from the same quadrature. No column,
source distance, or `(1-x)^-2` observer factor is stored in the material.

This provides ensemble grain physics through effective material tables. It
does not add explicit per-grain trajectories or a configurable Mie solver.

## Verification completed here

- Four RefleX pencil-beam controls, 200,000 histories each, tested combined
  extinction, half column, absorption alone and scattering alone. The largest
  discrepancy from Beer–Lambert transmission is **0.70 binomial standard
  deviations**. Each run completed successfully. Counts, inputs and logs are
  retained in `evidence/`.
- The unchanged production host loading/validation functions at `7deb50c`
  accepted both generated tables. Their function/class ASTs were executed with
  NumPy, excluding JAX imports and JAX array construction.
- Solid-angle closure relative error: **3.6e-15**. Maximum CDF absolute difference
  from the analytic distribution: **2.10e-7**. Metadata checksums agree.
- The optional patch applies cleanly to the inspected driver/config files,
  and its modified Python parses successfully.

The initial host-only checks above were followed by actual JAX construction,
155 discovered regression/material tests, an 8,192-history patched-driver smoke
run, and a six-million-history external comparison. All passed. See
`RESULTS_2026-10-16.md` and `results_2026-10-16/` for the current evidence.

DSH predicts 5,997.26 halo counts versus RefleX's 5,976: a 0.356% measured
offset, or 0.267 combined standard errors. The nine-bin radial statistic is
chi-square=4.044 for 9 degrees of freedom, p=0.909, with no fitted amplitude.
The reference counting precision is about 1.3%, so the observed offset must
not be presented as a 0.36% accuracy bound. No caps or invalid states occurred.

## Files and installation

Extract the archive into the repository root. It contains a `reflex_adapter/`
directory. The hook targets `development/physics-improvements` at `7deb50c`;
it does not target the divergent configured-input refactor.

From the repository root:

```powershell
python reflex_adapter/install_adapter.py --repo .
python reflex_adapter/verify_materials.py --repository .
python -m unittest reflex_adapter.test_reflex_materials -v
```

The checks require the normal DSH dependencies, including JAX. They construct
the real physics table and check the explicit zero-absorption option. The installer
preserves existing science settings, handles an already applied driver hook,
and was tested with LF and Windows CRLF files. The original configuration-tail
patch context has been corrected in this revision.
The package also supports a NumPy-only audit:

```powershell
python reflex_adapter/verify_materials.py
```

To select the new material in `sim_config.py`:

```python
MATERIAL_MODEL = "reflex"
ENERGY_EDGES_KEV = None
REFLEX_MATERIAL_DIRECTORY = "reflex_adapter/materials"
REFLEX_ABSORPTION_ENABLED = True
```

The default `MATERIAL_MODEL = "default"` preserves the existing behavior.
For a scattering-only control, select `REFLEX_ABSORPTION_ENABLED = False` and
turn both gas and grain absorption off in RefleX. The hook deliberately rejects
continuous-energy runs with this initial three-energy package.

These flags select materials only; they are not a frozen benchmark scene.
The existing science config contains a billion-history GPU run. Use an explicit
small CPU smoke configuration and a separately checked comparison geometry
before a production run. Require zero interaction-guard terminations.

Reproduce the supplied tables using NumPy 2 or later:

```powershell
python reflex_adapter/build_reflex_materials.py --dump reflex_adapter/evidence/cross_sections.fits --output reflex_adapter/materials
```

The supported RefleX export input is `evidence/probe.par`; on Linux, run the
supplied executable with that input in a scratch directory. It performs one
history after writing the opacity dump. The original executable is not bundled
again; its SHA256 is recorded in `evidence/verification_summary.json`.

## Image evidence and comparison requirements

The attached run completed **100 million histories** in **27,637 seconds**.
It uses 4.9 keV, source distance 10 kpc, a midpoint disc of radius 30 pc and
thickness 30 pc, NH=1e22 cm^-2, and 10,000 disjoint cameras of aperture radius
0.1 pc. The stacked images have 60x60 pixels at approximately 5 arcsec/pixel.
The dust-only image contains **5,976 counts**. Subtracting it from the all-photon
image gives **313,691 direct counts**, confined to the four central pixels.
Their ratio is **0.0190506**.

A separate thin-screen, first-order estimate using the extracted angular law
gives a halo/direct ratio of approximately **0.01882** inside the square field.
This is about 0.94 approximate count standard deviations below the measured
all-order ratio. It is a consistency check, not an all-order transport test:
it neglects detailed camera response, finite slab geometry and higher orders.

For further image benchmarks:

1. Start with low column, absorption off, and first-order observables. Then
   restore NH=1e22, absorption, and compare all orders explicitly. Selecting
   `DUSTSCAT > 0` includes every scattering order.
2. Compare physical halo fluence with the camera stack's solid-angle/area
   normalization. Raw peel-off event counts and RefleX camera counts are
   different estimators. A halo/direct ratio is a useful additional observable.
3. Apply the camera aperture response: the midpoint-screen halo blur has radius
   approximately **4.13 arcsec**, comparable to a 5-arcsec pixel. An aperture
   response applied to the direct beam has a different angular scale.
4. Match the material volume separately from the displayed image FOV. A DSH
   cloud stored only over the central 300-arcsec image does not reproduce the
   full 30-pc RefleX disc for repeated transport. Outer scatterers can contribute
   to an inner image after additional scattering.
5. Integrate the same arrival-time support and use first/multiple/total fluence
   with history-based uncertainties. The current image pair alone cannot
   distinguish scattering orders or establish transport closure.

## Numerical qualifications

The supplied executable uses a hybrid energy grid. Its runtime lookup below
10 keV reads a row offset relative to the computed DUMP energy nodes, and
changes indexing at 10 keV. The producer explicitly reproduces the inspected
`get_index/get_all_sigma` behavior for 2–10 keV. This is a compatibility choice
for this binary, not a general physical definition of WD01 opacities. DUMP
rounds native values to float32. Relevant routine disassemblies are retained.

The analytic sampler formally permits alpha>pi; rotating by such angles folds
their physical deflections. This table instead conditions on alpha<=pi, sets
the endpoint phase value to zero, and renormalizes. The excluded probability
is at most **2.94e-8** at the three exported energies. The endpoint quadrature
effect is included in the reported CDF error. The dataset therefore makes a
controlled approximation, rather than claiming bit-for-bit identity with every
possible RefleX event.

Only the listed monochromatic nodes have been prepared and audited. A reliable
continuous 2–10 keV provider needs a RefleX-specific interpolation/edge study;
the old TBabs adaptive grid is not automatically appropriate. The optional driver
change was applied and smoke-tested in the isolated checkout before packaging.

## Reproduce the accepted comparison

From the repository root, after installation:

```powershell
python reflex_adapter/compare_reflex.py --output reflex_adapter/my_new_run
python reflex_adapter/plot_comparison.py reflex_adapter/my_new_run
python reflex_adapter/check_first_order.py reflex_adapter/my_new_run
```

The first command runs three seeds of two million histories, using the frozen
`comparison_config.json`. It calls the production launch, transport and observer
functions directly; it does not use the user's science scene in `sim_config.py`.
Do not point a new run at the supplied accepted results directory.

To recompute the statistics from the supplied saved moments without rerunning
transport:

```powershell
python reflex_adapter/compare_reflex.py --reuse --output reflex_adapter/results_2026-10-16
```

This checks configuration and collection provenance before reusing saved seeds.
Production source files must match the tested development branch. Source hashes permit only LF/Windows CRLF newline conversion; all other source
changes invalidate reuse. Graphs can always be rendered
from the saved arrays using `plot_comparison.py`, without JAX.

A small integration run is also available:

```powershell
python reflex_adapter/smoke_driver.py --output reflex_adapter/my_driver_smoke
```

It selects RefleX materials in memory, verifies the selected opacity, runs 8,192
histories on CPU and checks completion/output, without editing science settings.
