# Coupled Compton transport and observer core

This checkpoint connects the accepted Compton sampler and cold material
provider to voxel transport and peel-off scoring through explicit low-level
APIs. The existing configured runner continues to use its accepted elastic
path. Configuration, response transport and production archive integration
are the next checkpoint.

## API

```python
import jax
from dsh.transport.inelastic import transport_inelastic_batch
from dsh.observer.inelastic import score_inelastic_peeloff, bin_inelastic_events

transport = jax.jit(transport_inelastic_batch, static_argnames=("max_interactions",))
result = transport(
    key, launched.position_pc, launched.momentum_kev, cloud, dust, compton,
    energy_cutoff_kev=2.0, max_interactions=64,
)
scored = jax.jit(score_inelastic_peeloff)(launched, result, cloud, dust, compton)
products = jax.jit(bin_inelastic_events)(scored, observer_bins)
```

`dust` is the existing numeric DustPhysicsTable containing dust scattering,
total photoabsorption, and its normalized dust angular law. `compton` is a
ComptonPhysicsTable obtained with `to_compton_physics`. Construct both from
the **same declared elemental budget** and common per-H-nucleus normalization.
Numeric JAX tables do not retain material hashes; the calling host loader
must verify matching `material_definition_sha256` values before conversion.
The example below performs this check. Do not combine the default MRN/TBabs
recipe with a different cold material simply because the energy grids overlap.

Supplying `compton=None` directly calls the accepted transport and observer
functions. Their original fields and random streams are bitwise preserved;
the added metadata has zero Compton counts. The numerical energy cutoff is
not applied to this disabled path.

## Transport and energy accounting

At every real interaction, the current incident energy selects dust scattering,
incoherent scattering and photoabsorption opacities. Their sum sets the sampled
optical depth and their ratios select the process. Dust is elastic. Compton
changes direction and energy using the accepted full-sphere sampler and
stationary-electron recoil. All three opacities are reevaluated after recoil,
including photoelectric edge handling inherited from the dust material table.

Interaction type 3 denotes Compton scattering; types 0, 1 and 2 retain their
existing meanings. Records include incident and outgoing four-momenta,
cumulative geometry, prefix dust/Compton orders, recoil and deposition.

For each valid input photon, within the active floating-point precision:

`source_energy_kev = deposited_energy_kev + momentum_kev[0]`

Deposition is real recoil plus real photoabsorption. A virtual observer photon
does not add recoil heat. Final momentum is zero for absorption; escaping and
incomplete photons retain their residual energy. Transport does not apply
physical packet weights to these per-photon energy diagnostics.

| Status | Meaning | Required treatment |
| --- | --- | --- |
| 1, 2 | Observer plane / outer boundary | Completed analog escape |
| 3 | Absorption | Completed; remaining energy deposited |
| 4 | Interaction safety limit | Incomplete; residual energy retained |
| 5 | Unsupported initial energy | Invalid for this common material support |
| 6 | Invalid state or nonpositive/nonfinite cutoff | Reject input |
| 7 | Below explicit continuation cutoff | Incomplete numerical loss; residual in `cutoff_energy_kev` |
| 8 | Continuation outside common table support | Incomplete model support; residual in `unsupported_energy_kev` |

The cutoff is a declared numerical continuation limit. It does not imply
photoabsorption or an escaped photon. Table indices are clipped only to keep
inactive calculations safe; unsupported physical requests receive no
extrapolated cross-sections. The accepted cold provider still covers 2--10 keV.
Support for transport at a given energy requires **both** material tables.

## Observer scoring and components

Each real dust or Compton event creates an alternative ray aimed at the
observer. The real channel-selection probability has already been sampled,
so the score uses that channel's normalized phase, without another opacity
factor. A Compton observer ray receives its own recoil energy from its
observer-directed angle, independently of the sampled continuation direction.
Its escape extinction includes dust, Compton and absorption evaluated at that
virtual outgoing energy. Dust events also include Compton escape extinction.

The existing geometry, source importance correction, inverse-square dilution
and arrival delay convention are preserved. A small-angle formulation retains
forward bound-electron suppression in float32. Source, incident, observed and
actual continuation energies are separate quantities.

`scored.events.energy_kev` is **observed outgoing energy**. Additional arrays
retain source and incident energy and per-event interaction type. Component
values are 1 dust-only, 2 Compton-only and 3 mixed, based on the history prefix
through that event. A later scatter, absorption or truncation cannot relabel
or erase an earlier valid observer alternative.

An observer escape energy outside common table support sets
`unsupported_escape` and records the rejected virtual energy; its score is
zero. A missing dust differential law with nonzero dust opacity sets
`unsupported_phase`. Both require explicit acceptance diagnostics. Numerical
continuation cutoff does not suppress otherwise supported virtual photons,
including virtual energies below that cutoff. Continuing histories can still
be incomplete, so supported earlier scores alone do not certify full-run
convergence.

`bin_inelastic_events` returns total, dust-only, Compton-only and mixed physical
fluence cubes using observed energy. Total history second moments include
cross-component terms. Do not obtain total variance by adding component
variances. Cross-component covariance is not exported by this core API.
Instrument response, FITS and configured-run metadata are not added here.

## Acceptance and executable example

From the repository root:

```powershell
python -m scripts.validate_compton_transport
```

This produces `outputs/compton-transport/compton_transport_report.zip` with:

- 32 focused tests in both float32 and float64: current-energy process
  competition; independent recoil/opacity controls; energy closure; terminal
  statuses; prefix classification; virtual energy, escape opacity and timing;
  component binning and complete-history uncertainty; exact disabled behavior.
- A 262,144-history free-electron single-scatter analog/peel-off comparison in
  each precision. Analog hits in a 500 pc observer disk are compared with
  central peel-off density. Acceptance requires at least 100 analog hits and
  agreement within five paired standard errors plus a declared 2% finite-disk
  allowance. This checks estimator normalization, not a cold-material model.
- A 4,096-history matched WD01/cold-material example in each precision, using
  the bundled `wd01_wilm_c50_si90_v1` budget and explicitly selected
  `atomic_incoherent` grain policy. It requires all three components, complete
  terminal statuses, finite closed cubes, no unbinned fluence, no unsupported
  virtual rays and energy closure better than 2e-5 keV per history.
- All 140 earlier test executions and the 146 frozen benchmark hashes, using
  the unchanged prior validators. A full pass totals **204 test executions**.

The example writes `material/core_example.npz` and `material/cold_compton.npz`
under each controls directory. NPZ prefixes are `packet__` for per-history
diagnostics, `event__` for valid observer events (with packet indices), `axis__`
for bin axes, and `<component>__` for existing binned-product fields. Source
fluence totals 1 ph/cm² across the three anchors 3.3, 4.9 and 6.9 keV; it is a
reproducible low-level integration example, not an observational fit. It does
not sample an instrument response or propose a stable production file schema.

The declared material prescription retains its previous limitations:
stationary-electron recoil without Doppler broadening, explicitly chosen grain
treatment, and no claimed equivalence to RefleX bound-electron angular physics.
External transport/response benchmarks remain to be performed after the
configured-run integration.
