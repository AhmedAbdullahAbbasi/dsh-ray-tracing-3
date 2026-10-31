# Configured Compton simulation and physical halo responses

This checkpoint connects the accepted coupled transport and peel-off APIs to
`python -m dsh.command check/run`, with material checks, versioned NPZ/FITS
products, response collection/folding and completion diagnostics. The numerical
transport/scoring kernels and accepted cold prescription remain unchanged.

## Select the prescription in TOML

Existing configurations omit `[compton]` and retain the elastic execution path.
An explicit `[compton]` with `kind = "disabled"` selects that same path. For a
matched declared material bundle:

```toml
[materials]
kind = "declared_bundle"
directory = "../../dsh/data/materials/wd01_wilm_c50_si90_v1"

[compton]
kind = "cold_material"
grain_electrons = "atomic_incoherent"
energy_cutoff_kev = 2.0
```

The grain choice is required: `omit` excludes solid incoherent scattering;
`atomic_incoherent` uses the declared isolated-atom approximation for grain
electrons. Neither is an implicit free-electron prescription. The loader
builds the cold table from the same material definition as dust and absorption,
using their energy nodes within 2--10 keV and the cold support endpoints. It
checks matching material-definition hashes before conversion to JAX arrays.
The cold source-function JSON is included in input provenance and change checks.

For a previously written Compton material file:

```toml
[compton]
kind = "external_table"
file = "cold_compton.npz"
energy_cutoff_kev = 2.0
```

The file's checksums and material-definition hash must match both selected dust
and absorption tables. Legacy effective table pairs without a shared declared
composition cannot be combined with an arbitrary Compton prescription. The
file declares its own free/bound state and redistribution assumptions.

Enabled modes require a positive explicit continuation cutoff within common
material support, below its upper endpoint. Initial source support must fit
both materials and lie at or above the cutoff. Unknown, missing and irrelevant
mode settings are rejected. The host-only TOML parser still imports no JAX,
NumPy, Astropy or Matplotlib.

## Run the examples

From the repository root, using the existing Python environment:

```powershell
python -m dsh.command check configs/compton/cold_smoke.toml
python -m dsh.command run configs/compton/cold_smoke.toml
python -m dsh.command run configs/compton/collect.toml
python -m dsh.command run configs/compton/fold.toml
```

The first simulation uses the three anchors 3.3, 4.9 and 6.9 keV. Collection
uses a continuous 3.3--8 keV proposal; folding applies a new spectral/time
history within that proposal. These bounded examples demonstrate the pipeline,
not scientific convergence. Output directories are fresh by default; use a
new configured directory or explicitly select the existing overwrite policy.
Relative paths are resolved against each TOML file.

## Simulation outputs and quality

Compton-enabled NPZ products use schema 9. The original total/first/multiple
fluence and photon-grouped moments remain in their usual fields. Prefix
components add `<dust_only|compton_only|mixed>__<product_field>` for every
existing binned-product field. Compton energy, cosine, opacity, normalized
phase and CDF arrays are stored alongside prescription metadata. Observer
energy bins refer to outgoing virtual photon energies; source energy nodes
and source spectra remain separate input arrays.

FITS adds `DUST4D`, `COMPT4D`, `MIXED4D`, component first/multiple/count/moment
HDUs, the `COMPONENTS` scalar summary, numeric `COMPT_*` tables/images,
`COMPT_DIAGNOSTICS` and `COMPT_METADATA`. `STATUS` includes cutoff and unsupported
continuation codes. The primary total image retains its normal viewer behavior.
Disabled runs retain their existing output schemas and numerical results.

`inelastic_*` NPZ diagnostics and the JSON report record real dust/Compton
counts, recoil and absorption, residual energies in escape/cutoff/unsupported/
safety-limit histories, and maximum per-history energy closure error. Energy
sums describe unweighted simulated photon histories; they are not an
isotropic-source calorimetric estimate. Virtual observer recoil is not heat.

Completion requires complete terminal histories, finite products, consistent
component/history accounting, energy closure within 2e-5 keV per photon and no
unsupported observer energies or missing phase laws. Cutoff and support losses
remain explicitly incomplete. An otherwise supported earlier peel-off score
survives such termination, but cannot certify the missing continuation.
Publication withholds production arrays and `complete.json` on failure. The
existing diagnostic option saves clearly labeled incomplete products instead.
Response collection reports evaluated/requested histories, terminal counts
and partial recoil/absorption/cutoff/support energy diagnostics when it aborts.

Observer-window losses remain separate from numerical failures. Unsupported
virtual escapes are conservatively failures even when their energies would
fall outside the requested observer window. In particular, starting a continuous
source exactly at the cold provider's 2 keV lower boundary can expose cutoff
or unsupported virtual rays; no lower-energy opacity is invented. The supplied
examples start above that boundary. Extending the cold prescription below its
accepted support or declaring a justified observable-specific truncation policy
would be separate scientific changes.

## Physical halo response schema 2

This response includes cloud transport, angular geometry, delay and outgoing
extinction. Detector ARF/RMF, PSF, counting noise and the direct unscattered
beam remain outside it.

Elastic schema-1 archives remain readable and writable. Compton collection
writes schema 2 with observed `energy_kev`, initial `source_energy_kev`,
`incident_energy_kev`, prefix dust/Compton orders and interaction type. Every
new array participates in the archive checksum. Source energy support describes
the initial proposal, so an observed photon can fall below the proposal's
lower energy while remaining within material support.

For each sample the response weight is `G / (N q(E_source))`. Folding therefore
evaluates the target source spectral density at **source energy**, while
selecting observer bins at **observed energy**. It integrates the shifted source
time intervals using exact stored geometric delays. No extra opacity, recoil
energy deposition or energy-bin Jacobian is applied during folding.

All events from a launched photon stay grouped. Folded output includes total
fluence/covariance and each prefix component's fluence and standard error.
Total covariance includes cross-component terms; component variances must not
be added to obtain it. Independent source-fit errors and detector counting
noise are not included in this finite-response Monte Carlo uncertainty.

## Acceptance and remaining work

```powershell
python -m scripts.validate_configured_compton
```

The report ZIP is `outputs/compton-configured/compton_configured_report.zip`.
Acceptance runs the whole unittest suite in isolated groups, repeats the coupled
core/configured/response tests in float64, repeats the original Compton physics
tests in float64, and runs all three saved-product TOML examples in both
precisions. Optional dependency skips are reported explicitly. It checks the
146 frozen benchmark hashes and byte-preserves the accepted numerical core
relative to `546e6cd` (normalizing checkout line endings only).

Earlier validators that seal pre-integration orchestration source are historical
checkpoint validators. Use the new validator for this stage, which deliberately
updates configuration, response orchestration and output writers while checking
legacy numerical behavior through regression tests.

Next is the scientific Compton-on/off comparison against the user's RefleX
simulations, with aligned source spectra, geometry, material inventory, angular
and time windows, scattering-order selection and uncertainty treatment. The
accepted model still uses stationary-electron recoil without Doppler broadening
or shell selection and makes no claim of RefleX bound-angular equivalence.
Detector modeling and cluster execution testing remain separate work.
