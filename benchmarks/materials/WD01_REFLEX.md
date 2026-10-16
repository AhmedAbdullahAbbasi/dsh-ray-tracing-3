# Independent WD01/D03 material versus the RefleX benchmark

This comparison runs production transport with the complete
`wd01_wilm_c50_si90_v1` material. It uses the accepted RefleX geometry,
monochromatic source, finite camera aperture, image grid and predetermined
source/camera normalization. There is no fitted amplitude or column adjustment.

The new material uses WD01 R_V=3.1, b_C=60 ppm case-A size-distribution shapes,
D03 graphite/silicate optics, Mie integrated scattering and grain absorption,
and the checked hybrid Mie/ADT angular law. Its grain numbers are explicitly
renormalized to Wilms abundances: 50% of carbon and 90% of silicon in grains;
Mg/Fe/O follow MgFeSiO4 stoichiometry. This is an experimental abundance recipe,
not the nominal published WD01 extinction normalization. Absorption sums
residual neutral atomic gas from the TBnew/Verner basis, molecular H2, and Mie
grain absorption once. The H2 fraction is 0.2 by hydrogen nuclei.

The reference is the existing RefleX 3.0 simulation and its exported material
tables. Its total abundances are Wilms, H2 fraction 0.2, and DUST=0.691 specifies
the iron fraction in solids. The independent WD01 recipe has a different solid
inventory. Neither its absorption nor its angular law is replaced by the
RefleX adapter in the new transport run.

## Reproduce

From the repository root, with JAX, NumPy, Astropy, SciPy and Matplotlib installed:

```powershell
python scripts/compare_wd01_reflex.py --output outputs/wd01_reflex
```

This collects three independent runs of two million histories each, using
seeds 28611, 28612 and 28613. It writes the image and profile plots automatically.
No native TBnew/XSPEC or Mie build is needed; runtime uses the checked tables.
`--material PATH` selects another complete declared WD01 material directory.
`--reuse` requires matching inputs, collection code and runtime provenance.
Always use a separate output directory for another material or runtime.

To regenerate the committed figures and FITS without transport, including on a
different computer:

```powershell
python scripts/plot_wd01_reflex.py benchmarks/materials/wd01_reflex_2026-10-16
```

## Experiment and outputs

- Energy 4.9 keV; central N_H=10^22 cm^-2.
- Source 10 kpc; cloud center 5 kpc; thickness and radius both 30 pc.
- Unchanged 16-by-16 cloud angular discretization and 0.1 pc camera aperture.
- Native image: 60-by-60 pixels, 5 arcsec each, covering a 300 arcsec square.
- Reference: 100 million RefleX launched photons, yielding 5,976 halo counts.
- Three independent DSH seeds; all scattering orders retained, outgoing
  extinction included, and per-history moments include zero contributions.

The PNG/PDF comparison shows both images on a common absolute count scale,
rebinned only for display to 15 arcsec. The radial calculations retain native
pixel selection. The last annulus is clipped by the square image; its area is
the number of selected pixels times their area. RefleX Y is reversed to align
with increasing DSH sky Y. The previous passing DSH/RefleX-material profile is
shown as a separate reference curve when its committed evidence is present.

`halo_images.fits` contains the aligned raw reference, floating-point WD01
camera-equivalent counts, their difference, physical halo fluence, first and
multiple-scattering fluences, and per-pixel history MC standard error. These
retain the native 5 arcsec linear sky-coordinate grid. Camera-equivalent model
counts are predictions, not a second analog photon-count realization.
`radial_profile.csv` exports the nine annuli and their uncertainty estimates.
NPZ files retain full radial covariance and scattering-order moments.

The global radial statistic uses independent history MC covariance plus a
plug-in multinomial covariance from observed RefleX counts. Ratio errors use
both numerator and denominator variances with the actual model/reference
ratio. No geometry allowance is added to this statistical diagnostic.
Numerical quality gates (complete transport, sufficient sampling, independent
seed consistency, and order closure) are separate from physical model
agreement. A physical difference is not a transport failure.

This test covers the existing 4.9 keV image. Broadband edge and time-history
validation remains documented in `ABSORPTION_EDGES.md` and
`SPECTRAL_HISTORIES.md`; a single monochromatic image does not establish
agreement across energy or abundance choices.

## Results

Six million histories completed with zero capped or invalid transport histories.
All numerical quality gates and seven targeted covariance/export tests pass.

| Quantity | Result |
| --- | ---: |
| RefleX halo counts | 5,976 |
| WD01 predicted halo counts | 5,992.02 |
| WD01 / RefleX halo total | 1.00268 ± 0.01333 (combined statistical SE) |
| DSH aperture MC precision | 0.3085% |
| Total halo difference | +0.268%, 0.202 sigma |
| Global radial chi-square | 16.280 / 9 degrees of freedom |
| Global radial p-value | 0.06126 |
| Maximum absolute annulus discrepancy | 2.238 sigma |
| Independent seed consistency p-value | 0.4932 |
| Predicted / observed direct counts | 313,689.29 / 313,691 |

The 45–60 arcsec bin is 7.68% above RefleX; the 60–80 and 80–100 arcsec bins
are 8.04% and 12.06% below it. The aperture total therefore gives a closer match
than the resolved profile. The global radial diagnostic is marginal, with
p=0.061; this run does not establish interchangeable physical materials.
Increasing RefleX photon statistics and adding energy slices would better
resolve the model differences. This comparison changes neither the column nor
the material recipe to improve agreement.

| 4.9 keV material diagnostic | RefleX | WD01/D03 |
| --- | ---: | ---: |
| Scattering cross-section [cm²/H] | 1.97720e-24 | 2.00870e-24 |
| Total absorption cross-section [cm²/H] | 2.48326e-24 | 2.54036e-24 |
| Central direct transmission | 0.956376 | 0.955529 |
| Median physical scattering angle [arcsec] | 74.7624 | 75.3256 |

The normalized phase comparison exposes a depression at intermediate physical
deflection angles and a stronger far-angle tail. For this approximately
midway cloud, physical deflection is approximately twice the observed angular
offset in the small-angle limit. This explains why similar median angles can
coexist with different resolved radial profiles. The new mixture raises
scattering opacity by 1.593% and absorption opacity by 2.299% at this energy.

The machine-readable result and material audit are in
`wd01_reflex_2026-10-16/comparison_results.json` and
`material_diagnostics.json`. Collection fingerprints and source/input hashes
are in `provenance.json`; the benchmark manifest records the final code and
evidence hashes.
