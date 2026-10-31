# Cold neutral Compton material checkpoint

Date: 1 November 2026. Branch: `feature/compton-scattering`.
Parent checkpoint: `b65cf9d718dde396383a9e277398991578cce8df`.

This checkpoint implements an offline, independently testable cold-material
provider. Production simulations still use the accepted dust-scattering and
photoabsorption transport. No production Compton switch is introduced here.

## Physical prescription

For each neutral target, the unpolarized differential cross section is the
free-electron Klein–Nishina law multiplied by its incoherent scattering
function S. Atomic S is per atom; the bundled H2 function is per H nucleus
IN H2. The inventory counts H2 molecules and multiplies that S by two exactly
once. At f_H2=0.2, the targets per total H nucleus are 0.8 isolated H atoms
and 0.1 H2 molecules. Helium and metals follow the material's abundances.

The momentum-transfer argument retains stationary Compton recoil, following
Hubbell et al. (1975), Eq. (2):

`x = (E/hc) sqrt[u (1 + (epsilon^2+2 epsilon)u)] / (1+2 epsilon u)`,

where u=(1-mu)/2, epsilon=E/(m_e c^2), and hc=12.398419843320026 keV Angstrom.
This is half the photon momentum transfer divided by h, expressed in inverse
Angstrom. Its small-angle/low-energy limit is sin(theta/2)/lambda. Retaining
the recoil correction matters when comparing with integrated Hubbell totals.
An independent vector-momentum test checks this conversion.

Isolated hydrogen uses the analytic Pirenne result, S=1-F^2, with
F=[1+(2 pi a0 x)^2]^-2 and the CODATA 2022 Bohr radius. expm1 avoids losing
the small-x limit. All other source functions tend quadratically to S(0)=0.

Photon energy redistribution is explicitly the stationary-electron recoil
law already accepted in Stage 1. The provider does not implement shell
selection, binding-energy thresholds, Compton profiles, electron Doppler
broadening, thermal electrons, or polarization. A bound-electron angular
correction is not a complete shell-resolved redistribution model.

## Source data and accuracy

`dsh/data/materials/cold_compton_v1/scattering_functions.json` contains:

- A numeric xraylib 4.2.0 SF_Compt snapshot for Z=1..30 at listed source
  nodes. The first four helium nodes use the corrected Hubbell data hosted
  by ORNL. Source references and these substitutions are recorded explicitly.
- Selected unambiguous Table IV H2 values from the NIST-hosted Hubbell
  reprint, printed page 536. Nodes with conflicting OCR readings were
  omitted rather than guessed. These values are per H nucleus in H2.
- A canonical JSON content checksum, independent of LF/CRLF checkout style.

Positive atomic nodes use log-log linear interpolation. H2 uses monotone
log-log cubic Hermite interpolation (PCHIP) to preserve its shape across the
selected source nodes. Both use a declared quadratic continuation below the
first positive x node; no positive forward floor or high-x extrapolation is
introduced. No xraylib, SciPy, or network access is needed to build or use a
material from this snapshot. The numeric snapshot can be reproduced from
the declared xraylib version, source nodes and helium substitutions; the
selected H2 source nodes remain explicit transcribed data.

Independent published-total checks use Hubbell Table V: isolated-H totals at
2, 5 and 10 keV within 0.07%; H2 totals per nucleus at 2, 4, 5 and 10 keV
within 0.4%. These are source/model comparisons, not claims of 0.4% physical
accuracy. Numerical angular quadrature has a separate 5e-4 acceptance gate.
The incoming-energy domain for this checkpoint is explicitly 2–10 keV.
Future transport must handle a downscattered continuation leaving common
material support; this module does not authorize clipping or extrapolation.

## Material and grain inventory

Every builder call requires a complete MaterialDefinition and an explicit
`grain_electrons` selection:

- `omit`: gas atoms and H2 contribute; all solid-phase incoherent scattering
  is explicitly omitted. This is the control used for the supplied RefleX
  comparison, whose gas Compton opacity decreases with dust depletion.
- `atomic_incoherent`: gas and solid atoms both contribute, each using the
  same neutral isolated-atom incoherent function. Gas plus solid numbers
  close to the declared elemental abundance. This is an explicit additive
  approximation for solid electrons, not a solid-state validation.

Neither policy duplicates coherent dust scattering. Rayleigh scattering is
a distinct coherent channel and is excluded from these Compton tables.
The new provider alone is therefore not a complete neutral-gas transfer model.

The reference material definition is a separate inventory: Wilms abundances,
f_H2=0.2, Fe dust fraction 0.691, graphite C/Si=4, and MgFeSiO4. Our existing
C50/Si90 material remains unchanged and can be passed to the same builder.
The reference definition does not claim to replace its coherent dust tables.

## Independent RefleX controls and discrepancy

The frozen controls in `benchmarks/compton/cold_material_v1` record the
supplied executable SHA256, exact parameters, logs, selected opacity rows,
and 10,282 selected single-Compton escaping photons from 2,000,000 histories.
The executable is not redistributed. Three effects are kept separate:

1. The supplied binary's init_ph multiplies its tabulated coherent and
   incoherent cross sections by **1.0195979722863812e-6**. Relative to the
   expected barn-to-Mb factor 1e-6, this includes a **1.019598** multiplier.
   Pure-H controls at exact native energy nodes independently recover the
   same factor. It is an observed executable behavior, not a fitted DSH
   correction. Raw exports and effective opacities are preserved separately.
2. RefleX linearly interpolates its coarse energy-tabulated total cross
   sections. Its below-10-keV runtime lookup also refers to DUMP row
   floor(E_eV), whose stored calculation energy is (row+1)eV. The selected
   evidence retains requested energy, row and calculation energy separately.
   After explicitly removing the recorded initialization scale for diagnosis
   and matching native energy interpolation, mixture opacity differs by at
   most **0.3942%** over the selected 2–10 keV grid. Direct evaluation of our
   continuous kernel instead differs by up to **1.755%**, predominantly from
   the reference's coarse interpolation. The DSH material is never rescaled.
3. The controlled angular experiment agrees with an uncorrected KN phase:
   KS distance **0.00927**. It disagrees with the bound H KN*S phase:
   KS distance **0.12866**. The sphere is optically thin (tau~0.005), so
   residual single-order escape-selection bias is far smaller than this
   discrepancy. The same photons satisfy stationary Compton recoil to
   about **7.3e-6 eV** using the Stage-1 constants.

Consequently, this checkpoint does **not** claim angular or full transport
equivalence to the supplied RefleX binary. This qualifies the earlier plan
to use RefleX angular distributions as an acceptance target. The physical
bound angular law is validated against analytic limits, published source
values, independent integration, and sampler/evaluator agreement instead.
The earlier Compton-on/off count measurements remain unchanged; their
exported opacity was the executable's effective opacity, including its scale.

## Use from the repository root

Run all acceptance checks, including the existing Stage-1 and dust-only checks:

```powershell
.\.venv\Scripts\python.exe -m scripts.validate_cold_compton
```

With an activated environment, `python -m scripts.validate_cold_compton`
also works. Reports are written under `outputs\compton-cold-material`, with
`compton_cold_material_report.zip` containing logs, source identities, both
explicit-grain-policy tables, and the reference diagnostic.

Build a material archive from the reference inventory:

```powershell
python -m scripts.build_cold_compton_material `
  --material-definition dsh/data/materials/cold_compton_v1/reflex_reference_definition.json `
  --grain-electrons omit `
  --output outputs/cold-compton/reference.npz
```

The builder writes the checksummed Stage-1-compatible NPZ plus a JSON ledger
of material identity, species contributions, assumptions and support.
Select `atomic_incoherent` explicitly to include the declared solid-electron
approximation. Existing archives still require the matching material hash.

Optional Linux reference recapture (requires Astropy and the inspected binary):

```bash
python -m scripts.capture_cold_compton_reflex \
  --executable /path/to/reflex_3.0_Linux_x86_64_2.17 \
  --output-directory outputs/fresh-cold-reflex
```

The directory must be empty and the executable hash must match. Acceptance
does not require this recapture or a locally installed RefleX executable.

## Remaining work

The next implementation milestone is transport plus observer scoring:
evolving packet energy, three competing interactions, channel-specific
peel-off energies and escape opacity, component histories, and cutoff
bookkeeping. It must preserve the accepted Compton-off path. A subsequent
inelastic response needs separate source and observed energies. The RefleX
angular/normalization findings must remain visible in future comparisons.

## Primary sources

- Hubbell et al. 1975, JPCRD 4, 471, doi:10.1063/1.555523:
  https://srd.nist.gov/jpcrdreprint/1.555523.pdf
- Corrected low-x atomic data hosted by ORNL:
  https://web.ornl.gov/~peplowde/p006/hubbs.dat
- xraylib project and data provenance: https://github.com/tschoonj/xraylib
- Paltani & Ricci 2017: https://arxiv.org/pdf/1906.08824
- Ricci & Paltani 2023: https://arxiv.org/pdf/2301.10268
