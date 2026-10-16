# Material models versus the RefleX image

This experiment compares four complete material prescriptions with the same
existing 4.9 keV RefleX analog image. Every case uses the accepted cloud/source
geometry, finite camera aperture, native pixel selection and predetermined
absolute normalization. There are no fitted parameters, matched brightness
rescalings or column adjustments.

## Model definitions

| DSH case | Scattering | Absorption | Composition/normalization |
| --- | --- | --- | --- |
| RefleX-matched adapter | Exported RefleX dust opacity and inspected analytic angular law | Exported PHOTO + DUST × (1 − ALBEDO) | Original exported Wilms vector, H2 nuclei fraction 0.2, iron solid fraction 0.691; DUST column already depletion-scaled |
| MRN/RG–Drude + TBabs | Historical Gaussian Rayleigh–Gans/Drude with MRN radii 0.005–0.3 micron | Frozen full XSPEC TBabs, abund wilm, xsect vern | Historical effective dust mass/H = 2.32475e-26 g; density 3 g cm^-3; not a new elemental-budget prescription |
| WD01/D03 Mie–ADT | WD01 R_V=3.1, b_C=60 ppm case-A shapes, D03 graphite/silicate optics; Mie totals and hybrid Mie/ADT phase | Residual neutral TBnew/Verner atomic gas + H2 + Mie grain absorption | Experimental Wilms C50/Si90 budget; graphite and MgFeSiO4 inventory; H2 nuclei fraction 0.2 |
| WD01/D03 full Mie | Same WD01/D03 C50/Si90 inventory; full-Mie angular reference | Same residual gas/H2 and Mie grain absorption recipe | Same elemental counts, grain numbers and densities as hybrid; scattering method alone changes |

The reference is the unchanged RefleX analog simulation, rather than a fifth
DSH material choice. The MRN 2–10 keV bank and original three-node NewDust bank
agree in scattering and absorption cross sections at 4.9 keV to better than
1e-7 relative. They are one physical prescription here, not two independent
model alternatives. The historical 4.46% difference between the integrated
RG/Drude differential law and NewDust's separately reported efficiency total
is preserved; interaction and angular-law normalization use the same integral.

For the WD01 pair, C50 means 50% of total Wilms carbon in solids, and Si90 means
90% of total silicon in solids. The graphite and silicate number multipliers
are 0.4813853948538391 and 0.3184324217049859 relative to nominal WD01. D03
densities are 2.2 and 3.8 g cm^-3. Bulk graphite uses 1/3 parallel plus 2/3
perpendicular orientation averaging, including at the smallest grain sizes;
PAH molecular optics are not included. This is an abundance-declared
experimental WD01-shape recipe, not nominal published WD01 normalization.

The complete full-Mie material is composed from the already checked dust
reference with source energies 2, 3.3, 4.9, 6.9 and 10 keV and the same residual
gas basis. No new native Mie or XSPEC solves are needed. The comparison uses
the exact 4.9 keV source node. Its sparse dust source grid does not establish
accurate broadband edge behavior. The generated material, complete audit and
source hashes are retained under `model_comparison_2026-10-16/full_mie_material`.
The checked source angular grids have 2,561 full-Mie nodes and 1,153 hybrid
nodes. Differences between their runtime phase tables therefore include both
the angular solver approximation and tabulation/interpolation. The physical
grain inventory, optics and absorption are held fixed. This experiment does
not replace the direct solver/grid checks in `WD01.md`.

Nominal WD01 dust overdraws the selected Wilms elemental inventory. It is
therefore not combined with an inconsistent residual Wilms gas in this test.
ZDA04 has not yet been implemented.

## Experiment

- Source distance 10 kpc; cloud midpoint 5 kpc; cloud thickness and radius 30 pc.
- Central hydrogen-nuclei column 10^22 cm^-2; monochromatic 4.9 keV source.
- Same 16-by-16 angular cloud discretization, production launch importance
  sampler, transport kernel, peel-off scorer and outgoing extinction.
- Camera grid 100-by-100 with 0.2 pc spacing and 0.1 pc aperture radius;
  source cone half-angle 1 degree; original RefleX launch count 100 million.
- Native halo image 60-by-60 pixels of 5 arcsec, spanning ±150 arcsec.
- Three seeds, two million independent histories each, for each material.
  All scattering orders are scored; the eight-interaction guard is checked
  for capped/invalid histories, not used to truncate accepted histories.

The RefleX-matched adapter uses verified saved seeds 8261–8263, and the WD01
hybrid uses verified saved seeds 28611–28613. New MRN histories use 38611–38613;
new full-Mie histories use 48611–48613. This gives 24 million DSH histories,
including 12 million freshly collected for this comparison. All four seed sets
are disjoint. Existing cache input/code hashes, configuration and available
actual material/cloud/launch/runtime fingerprints are checked before reuse.

## Statistics and scope

All moments combine scores belonging to the same photon history, including
zero-contribution histories. Radial MC covariance retains correlations across
annuli. RefleX counts use a plug-in multinomial covariance from the observed
counts, reflecting its fixed number of analog launched histories. The global
radial statistic uses the sum of the independent model MC and reference count
covariances. No geometry allowance or fitted normalization is added.

Ratio standard errors propagate both model and reference variances using the
actual numerator/denominator ratio. Different model-to-RefleX ratio curves
share one reference denominator and are correlated; they must not be treated
as independent measurements. The full-Mie-versus-hybrid statistic uses only
their independent DSH MC covariances, with no RefleX count noise added.

Numerical quality (complete transport, sampling precision, seed consistency,
order closure) is evaluated separately from agreement between physical
prescriptions. A discrepant historical dust model is not a transport failure.
The matched-adapter diagnostic p-value can differ slightly from the immutable
original benchmark because this comparison consistently uses covariance from
observed reference counts instead of the matched-model null expectation.

This is an ideal-observer halo comparison with no telescope PSF, exposure,
effective area or detector response. It uses one existing monochromatic image;
new energy slices and larger RefleX statistics would be required to assess
energy-dependent discrepancies and edge behavior.

## Reproduce and inspect

With JAX, NumPy, Astropy, SciPy and Matplotlib installed, run from repository root:

```powershell
python scripts/compare_material_models.py --output outputs/model_comparison
```

Figures and FITS are automatic. The default reuses the verified adapter/hybrid
benchmarks and collects the missing model cases. Use `--fresh` with a new
output directory to collect all four cases again. Use `--reuse` to resume an
interrupted collection or regenerate products with an unchanged collection
runtime and inputs; changed fingerprints require a fresh output directory.
If a different machine/runtime fails the strict existing-cache verification,
use `--fresh` with a new output directory to collect all four cases locally.
Rendering the saved results below does not require the collection runtime.

To re-render the committed results on another computer without transport:

```powershell
python scripts/plot_material_models.py benchmarks/materials/model_comparison_2026-10-16
```

`model_radial_comparison` shows all four absolute profiles and their ratios to
RefleX. `wd01_radial_detail` provides a closer ratio view of the WD01 and
matched-adapter cases. `model_image_comparison` shows all five images on a
common absolute logarithmic scale. Display pixels are summed to 15 arcsec;
all statistics retain the native 5 arcsec grid. `model_phase_comparison`
shows normalized intrinsic phase laws, and `mie_hybrid_phase_difference`
isolates the numerical scattering-approximation difference.

The outer annulus is clipped by the square image. Its mean surface fluence
uses actual selected pixel area. The RefleX Y axis is reversed to align with
DSH sky Y. The FITS file retains the native linear angular grid, aligned raw
reference counts, each model's floating-point camera-equivalent counts, and
per-pixel model MC standard errors. Camera-equivalent model counts are
predictions rather than an independent analog count realization.

JSON, NPZ and CSV products retain material definitions, scalar diagnostics,
per-model provenance, full radial covariance, scattering-order images and
selected-pixel radial measurements. The final manifest records source and
evidence checksums. Original RefleX and earlier WD01 benchmark files remain
unchanged.

## Results

The final numerical summary is `model_comparison_2026-10-16/comparison_results.json`.

All four numerical-quality checks pass for all cases, with zero capped or invalid
transport histories. Twelve targeted checks pass. The original 63-file RefleX
manifest and the 25-file earlier WD01 comparison manifest remain unchanged.

| DSH material | Predicted halo counts | Model / RefleX ± combined SE | Radial chi-square / 9 dof | Radial p |
| --- | ---: | ---: | ---: | ---: |
| RefleX-matched adapter | 5,997.26 | 1.00356 ± 0.01336 | 4.154 / 9 | 0.90097 |
| MRN/RG–Drude + TBabs | 16,422.17 | 2.74802 ± 0.03597 | 16065.917 / 9 | underflow (extremely small) |
| WD01/D03 Mie–ADT | 5,992.02 | 1.00268 ± 0.01333 | 16.280 / 9 | 0.061261 |
| WD01/D03 full Mie | 5,999.83 | 1.00399 ± 0.01335 | 20.445 / 9 | 0.015356 |

The MRN model predicts 2.748 times the halo fluence, with a broader normalized
phase law. Its scattering opacity is 2.874 times the exported RefleX value;
its median physical deflection is 84.73 arcsec versus 74.76 arcsec. These
differences reproduce the historical factor-of-2–3 discrepancy under unchanged
geometry and column. Changes to grain population, normalization, optics and
absorption accompany the move to the new WD01 material; this is not a test of
RG versus Mie with all other physical choices fixed.

The WD01 variants predict aperture totals within 0.4% of RefleX, but their
resolved profiles retain a bump around 45–60 arcsec and a deficit around
60–100 arcsec. Maximum individual discrepancies are 2.24 sigma for hybrid
and 2.65 sigma for full Mie. Their respective global radial p-values of
0.061 and 0.015 show that matching the total does not establish identical
materials or profiles. No physical-agreement threshold is used as a numerical
quality gate.

Full Mie / hybrid aperture ratio is 1.001302 ± 0.004363 using only
independent DSH MC errors, and their pairwise radial p-value is 0.5463.
They are statistically consistent at this energy and sampling precision.
Differences in their p-values against RefleX do not demonstrate that one
scattering solver is physically better. The runtime normalized-phase difference
is at most 0.0943% over physical deflections 0.5–450 arcsec; the maximum
CDF difference over all angles is 6.42501e-05. Larger pointwise differences
occur in faint far-angle tails and include the different angular tabulations.
These diagnostics are saved in `controlled_mie_diagnostics.json`.
