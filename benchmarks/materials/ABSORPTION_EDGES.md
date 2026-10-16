# Residual gas, H2 and one-sided energy edges

This milestone adds a portable neutral-gas/H2 absorption basis, a complete
chemically declared experimental WD01-shape material, and one-sided energy
edges in production transport and observer scoring. The existing V1, 2–10 keV
and RefleX providers retain their coefficients and default behavior.

## The gas configuration

The gas producer uses the author's pinned TBnew 2.3.2 C routines and Verner
PHFIT2 version 2 Fortran routine. The source archives are downloaded and
checksum-verified by `scripts.build_tbnew_gas`; upstream model source and
compiled libraries are **not redistributed** in this repository. The runtime
uses portable NumPy tables and needs neither XSPEC nor native compilers.

| Item | Explicit convention |
| --- | --- |
| Energy range | 2–10 keV, no extrapolation |
| Atoms | Neutral H through Zn, 30 separate cross sections in cm2/atom |
| H/He/O/Ne/Fe | TBnew's corresponding author routines |
| Other atoms | Author Verner PHFIT2 neutral-atom shell sum |
| H2 | TBnew/Yan prescription, cm2 per molecule |
| Molecular fraction | f_H2 = 2 n(H2)/n_H, where n_H counts total H nuclei |
| Residual atoms | Total abundances minus the declared grain inventory |
| Atomic H | Residual H minus the nuclei assigned to H2 |
| Solid absorption | Independent Mie Q_abs, added once |

The public TBnew interface exposes 18 atoms. In particular, it omits nonzero
Wilms entries for P, Ti and Mn. This basis includes them using the same Verner
prescription, and records that extension; it is not claimed to be an identical
copy of the standard 18-element model.

Setting TBnew's grain density to zero also switches off its internal depletion
handling. Therefore this producer derives residual gas from our audited
inventory first. Its native-model control uses those residual element columns,
rho=0, and the correct H2 fraction relative to residual gas hydrogen. It never
adds a full undepleted TBabs/TBnew opacity to the Mie grain term.

The portable basis has 184 energy nodes and 16 native neutral-shell thresholds.
Adaptive refinement checks every atom and H2, not just a particular abundance-
weighted mixture. Loader checks cover data checksum, element ordering, units,
energy domain, positive finite arrays and resolved edge brackets.

## The experimental complete material

The original [WD01/D03 bank](WD01.md) remains the published-normalization
reference. Its nominal grain inventory exceeds Wilms abundances and still
fails that combination's audit. This is a real abundance constraint; Zubko,
Dwek & Arendt (2004) also discuss excessive Mg/Si/Fe requirements in the related
Li–Draine models. Increasing a total abundance vector solely to hide an overdraw
would not validate the physical population.

`dsh/data/materials/wd01_wilm_c50_si90_v1/` is a **separately named experimental
recipe**, using WD01 shapes renormalized to explicit targets:

- Total elemental vector: Wilms, H=1 per hydrogen nucleus.
- Carbon: 50% gas, 50% solid, giving 120 ppm C in graphite.
- Silicon: 10% gas, 90% solid, giving 16.74 ppm Si in MgFeSiO4.
- Mg, Fe and O residuals follow that silicate stoichiometry. Other elements
  are gaseous because no additional solid species are declared.
- H2 nuclei fraction: 0.2, giving 0.8 H I atoms and 0.1 H2 molecules per H.

The C/Si targets correspond to those two parameters in TBnew's default input
file, but the complete depletion pattern differs from default TBabs: this model
contains only graphite and MgFeSiO4 solids. Its number factors are calculated
using the D03 optical material densities, graphite 2.2 and silicate 3.8 g/cm3.
The exact factors and derived gas fractions are in `recipe.json` and
`abundance_audit.json`.

This recipe preserves the shapes of the WD01 size laws, **not** their published
normalization or extinction fit. It has not been fitted to optical/IR extinction
or the user's sightlines. The H2 fraction is an explicit control value, not an
inference from the molecular-cloud cube. A justified sightline model can be
generated with other explicit inputs; the Python assembly API also accepts
other total elemental vectors. Bulk-graphite spheres remain the carbon optics
approximation, with no PAH molecular model or extra solid species.

The full material uses the previously checked per-component Mie/ADT arrays.
Changing component numbers/H scales their cross sections linearly. Regridding
log-interpolates totals and mixes **normalized** phase laws before reconstructing
the differential table and CDF. Component diagnostics must reproduce the source
bank's total/differential/grain-absorption arrays before any scaling is accepted.
Source checksums, exact factors and energy-grid checksum are recorded.

Atomic gas, H2 and grain absorption are separate checked files. The full loader
audits chemistry, requires disjoint complete phase coverage, verifies matching
definitions/grids/edges, and checks the stored total against the actual component
sum and component checksums. A zero H2 component remains an explicit phase;
unlabelled legacy tables retain their positive-opacity requirement.

## Edge semantics and outputs

Each discontinuity has distinct left/right samples in a narrow bracket, normally
E_edge*(1 +/- 2e-6). Both samples and the intervening threshold remain distinct
after float32 conversion. The builder rejects broad, overlapping, collapsed or
unbracketed edges.

Within that narrow interval, runtime evaluation uses the corresponding endpoint
on its own side; **the threshold itself takes the high-energy value**. It never
blends the two sides. Smooth intervals keep the established interpolation. The
energy displacement of a held endpoint is bounded by the bracket width;
cross-section accuracy is tested numerically. The same rule applies to interaction
opacities, transport CDFs, observer phase evaluation and observer escape extinction.
The independent host reference follows the stored runtime precision at thresholds.
No edge array means the existing behavior, checked against the old results.

`integrate_spectrum` splits Gauss quadrature at every supplied edge, table knot
or spectral break. This prevents spectral integration panels crossing a jump.
The Monte Carlo source continues to draw continuous intrinsic photon energies.
The existing transport still adds absorption and scattering for extinction on
both sides of each scattering event.

Edge-aware NPZ output is schema 7 and stores `physics_energy_edge_kev` plus the
interpolation convention. Legacy output remains schema 6. FITS `PHYSICS` adds
`EDGE_KEV` (threshold in the interval after each row, final zero) and `EDGEINT`.
Both output formats record the material-definition fingerprint (`MDEFSHA` in
the FITS primary header), alongside the table hashes.

Native source energies are preserved as physical input choices. In particular,
the Verner gas Fe K threshold is 7.124 keV; the D03 grain table contains its own
near-edge structure. No artificial shift aligns those two data sources. These
checks establish numerical handling of the chosen inputs, not laboratory-grade
edge calibration, mineralogical identification, ionization or molecular-metal
cross sections. Soft O/Ne/Fe-L edges are outside the current 2–10 keV domain.

## Run or regenerate

From the repository root, the new material is explicitly selected:

```powershell
python -m scripts.run_dsh_v1 `
  --materials declared `
  --material-directory dsh/data/materials/wd01_wilm_c50_si90_v1 `
  --source-spectrum hard-state-powerlaw `
  --packets 8192 --chunk-size 1024 `
  --output outputs/wd01_budget_smoke.npz
```

Generate another explicit Wilms-budget variant (H2 fraction can also be 0 or 1):

```powershell
python -m scripts.generate_wd01_ism `
  --output outputs/my_wd01_budget `
  --carbon-gas-fraction 0.5 --silicon-gas-fraction 0.1 --h2-fraction 0.2
```

To reproduce the atomic basis on GNU/Linux with GCC and GFortran:

```bash
python -m scripts.build_tbnew_gas --directory ../tbnew_backend
python -m scripts.generate_gas_basis \
  --backend ../tbnew_backend/libdsh_tbnew_gas.so \
  --phfit-source ../tbnew_backend/phfit2.f \
  --output dsh/data/absorption/tbnew_verner_basis_2_10_v1.npz
MIEPYTHON_USE_JIT=1 python -m scripts.validate_absorption_edges \
  --backend ../tbnew_backend/libdsh_tbnew_gas.so \
  --output outputs/absorption_checks.json \
  --direct-grain-check --transport-control --plot
```

Direct grain checks need `.[materials]`; plotting also needs Matplotlib. Core
table loading and simulations use the existing runtime dependencies.

## Independent XSPEC-process check: prepared, not run here

XSPEC is not installed in the generation environment. The author C/Fortran
checks below were executed; an independent XSPEC-process comparison remains
pending. Ten executable `.xcm` controls are saved in
`absorption_edges_2026-10-16/xspec_prepared/`, with `validation_executed=false`.
They use TBABSVERSION 2, abund wilm, rho=0 and explicit residual-gas abundances.
The comparison intentionally covers the native model's 18 atoms plus H2;
additional Verner atoms and Mie grains are not included in that XSPEC control.

In an initialized HEASoft environment:

```bash
python -m scripts.check_gas_with_xspec \
  --definition dsh/data/materials/wd01_wilm_c50_si90_v1/material_definition.json \
  --output outputs/xspec_gas_crosscheck
```

The script runs the processes, retains commands/logs and XSPEC version, checks
inferred coefficient agreement at two columns, and fails if its 0.1% gate is
not met. `--model tbnew` selects an externally loaded TBnew interface instead
of built-in `tbvarabs`; its installation remains the user's local XSPEC setup.

## Recorded validation

| Check | Result |
| --- | --- |
| Full unit suite | 211 pass; final targeted gas/edge/NPZ checks also pass |
| Portable basis versus author routines, 3,185 energies | Maximum atom error 0.02622%; H2 0.001724% |
| Residual core sum versus native TBnew gas transmission | Maximum relative difference 8.89e-16, seven energies and three columns |
| Actual JAX direct-beam spectral folds, Gamma=1.7 | Maximum relative difference 1.035e-5 across three bands and three columns |
| Folded reference order 32 versus 48 | Maximum relative difference 1.213e-9 |
| Between-node grain checks versus direct order-24 solves | Scattering/absorption errors <0.0082%; maximum CDF error 1.637e-4 |
| New-material first-flight control at NH=3e23 | 98,304 histories, maximum absolute z=1.454, no caps/invalid states |
| Multiple histories in that control | 2,243; termination checked, no new independent order-resolved halo reference claimed |
| Continuous 2–10 keV end-to-end smoke | 8,192 histories, NPZ/FITS written and verified |
| Fresh RefleX compatibility regression | 6,000,000 new histories; every comparison statistic equals the accepted result |

All preset gates pass. The saved reports distinguish native-model bookkeeping,
portable interpolation, grain interpolation, quadrature and transport checks.
They do not claim an external halo-image benchmark for the experimental material.
The next observational validation stage is evolving intrinsic spectra and
energy/angle/delay responses, after the independent XSPEC-process check.

## Primary sources

- [TBnew author documentation and source download](https://pulsar.sternwarte.uni-erlangen.de/wilms/research/tbabs/)
- [XSPEC TBabs-family interfaces and version control](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node318.html)
- [Verner author cross sections and PHFIT2 v2](https://www.pa.uky.edu/~verner/photo.html)
- [Zubko, Dwek & Arendt 2004, abundance constraints](https://arxiv.org/abs/astro-ph/0312641)
- [WD01/D03 implementation and optical-source references](WD01.md)
