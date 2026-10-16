# Material definitions and abundance accounting

This is the first milestone after the accepted RefleX benchmark at `8a5073c`.
It adds chemical input definitions and an opt-in table-composition contract on
`development/post-reflex-improvements`. The following [WD01/D03 milestone](WD01.md)
now adds actual size distributions, Mie/forward-ADT scattering and grain
absorption. The [absorption/edge milestone](ABSORPTION_EDGES.md) adds a pinned
gas/H2 basis, an explicit experimental budget variant and runtime edge handling;
the user subsequently reported the independent XSPEC-process checks passing.
The [spectral-history milestone](SPECTRAL_HISTORIES.md) adds changing intrinsic
spectra and validation of a reusable physical halo response. ZDA04 alternatives
remain the next physics milestone. The description below documents the first
milestone's contract.

## Definition and normalization

`dsh.physics.material_definitions` provides immutable `ElementalAbundances`,
`GrainComponent`, and `MaterialDefinition` objects. Definitions can be written
with `definition.as_dict()` and loaded using `load_material_definition(path)`.
The JSON schema requires `schema_version=1`, `normalization="per H nucleus"`,
and the declared mass-conversion convention. Unknown constructor fields fail.

The abundance vector contains total gas-plus-solid elemental counts per
hydrogen nucleus, with H=1. A supplied XSPEC `wilm` vector includes all 30
elements H through Zn, including explicit zeros. Custom vectors are supported
within those elements; a new dust population must use its actual assumed
abundance set instead of silently inheriting Wilms.

Every grain component declares its stoichiometry, material density, mass in
g per H nucleus, size-distribution identifier/parameters, optical-constant
identifier, and reference. The population mass is an input to this milestone:
a size-distribution generator must independently verify

    mass/H = integral [(4*pi/3) * a^3 * rho * (dn/da per H)] da.

The identifier and shape parameters alone do not implement or validate a size
distribution. Density and optical constants are recorded, not inferred from
the name "WD01", "graphite", or "silicate". Composition, size distribution,
optical constants, and scattering method remain separate recorded choices.

For a component with formula weight W and stoichiometric coefficient nu_Z,
the solid inventory is

    atoms_Z/H = (mass/H) * nu_Z / (W * atomic_mass_unit).

The audit sums all solids, subtracts them from the total elemental vector,
and derives the residual gas abundance and gas fraction of each element.
It rejects an overdrawn element; it does not rescale the grain population.
Only floating-point roundoff (32 ulps per elemental comparison) is tolerated.
Gas-plus-grain mass must close to relative tolerance 1e-13. Grain mass/H and
dust-to-gas mass ratio are reported separately and must not be interchanged.
Atomic weights use CIAAW's 2024 abridged values; these specify a reproducible
conversion convention rather than an astrophysical isotope distribution.

The molecular fraction counts hydrogen nuclei:

    f_H2 = 2*n(H2)/n_H,
    n(H I)/n_H + 2*n(H2)/n_H + n(H in grains)/n_H = 1.

Thus f_H2=0.2 gives 0.1 H2 molecules/H, and 0.8 H I atoms/H when no hydrogen
is assigned to solids. Hydrogen already in solids cannot also be put into H2.
Ionized gas and non-H2 gas molecules are outside this first contract.

## Audit command

Run from the repository root:

```powershell
python -m scripts.audit_material_definition benchmarks/materials/illustrative_definition.json
```

The supplied graphite/olivine mixture is deliberately an **illustrative chemical
inventory**, not a WD01, ZDA04, or RefleX population. No optical constants or
cross sections are supplied for it. `illustrative_audit.json` contains its
closed inventory. Use `--output path.json` to save another audit; an invalid
definition exits with an error before emitting an accepted report.

`definition.sha256` fingerprints the canonical definition, including the
elemental vector, grain masses/compositions/densities, population parameters,
optical-constant identifiers, method, references, and mass convention.
JSON formatting and grain-component ordering do not change the fingerprint.

## Declared material table composition

`dsh.physics.materials.build_declared_material_tables(definition, scattering,
absorption_components)` returns the existing `(scattering, absorption, physics)`
tuple and constructs the real JAX physics table through the established builder.
It requires:

- A successful elemental audit.
- The same `material_definition_sha256` in scattering and every absorption
  component; scattering also declares a matching `scattering_method`.
- Identical energy arrays and matching shared-grid checksums, when present.
- Intrinsic photoelectric absorption in `cm2 H-1`, with finite nonnegative
  component arrays. A phase with zero opacity still requires an explicit
  declaration and array.
- Disjoint, complete `absorption_coverage` declarations drawn from
  `atomic_gas`, `molecular_hydrogen`, and `grains`.

A total ISM absorption table declares all three phases. Combining it with a
second grain-absorption table fails. Alternatively, separately computed gas,
H2 and grain terms can be added once each. The sum is absorption, not total
extinction: the existing transport adds scattering to form interaction and
escape extinction. A Mie producer must supply Q_abs here, not Q_ext.

The numerical builder continues to check total-scattering, differential-law,
and CDF closure. Chemical declarations bind the inputs and reject incompatible
or overlapping combinations. They cannot prove that an external producer
actually calculated its declared physics; that requires producer-specific
and independent physical validation.

## Existing providers and validation

Existing NewDust/MRN+TBabs and RefleX effective tables remain on their accepted
loading paths. Their coefficients and the transport/observer code are unchanged.
They have not been retroactively assigned unverified chemical fingerprints.
The Drude material lacks an explicit elemental composition. RefleX exports
effective opacities; its `DUST=0.691` iron-depletion setting alone is insufficient
to reconstruct a complete grain inventory. Neither is advertised as passing
the new elemental audit.

Run the new checks with:

```powershell
python -m unittest tests.test_material_definitions -v
python -m unittest discover -v
```

Checks exercise independent stoichiometric counts, elemental/mass/H2 closure,
overdraw and malformed inputs, immutable/fingerprinted definitions, the entire
Wilms vector against RefleX's independent export, and actual JAX construction
from split absorption. A total-ISM-plus-grain combination must fail. Synthetic
test coefficients exercise this contract without claiming a new grain model.

`results_2026-10-16/` records the full suite and a reuse check of the accepted
six-million-history RefleX moments. The latter verifies the production/input
fingerprint and reproduces every accepted statistic exactly; it does not
collect new photon histories or extend the original physical validation scope.

## Following milestones

1. Implemented in [WD01/D03](WD01.md): the published size law, explicit
   normalization variants, mass integrals, Mie and bounded forward ADT.
   Nominal WD01 does not fit Wilms abundances; that combination is rejected.
3. Implemented in [absorption and edges](ABSORPTION_EDGES.md): residual gas/H2,
   explicit grain accounting, one-sided interpolation and split spectral folds.
   Independent XSPEC-process verification remains pending.
4. Validate evolving intrinsic spectra and energy-angle-delay responses,
   followed by instrument/extraction-region predictions.
5. Add selected ZDA04 alternatives; then undertake the larger refactor and
   speed work using the accepted physics regressions.

## Sources

- [XSPEC elemental abundance vectors, including Wilms et al. (2000)](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XSabund.html)
- [CIAAW 2024 abridged standard atomic weights](https://ciaaw.org/abridged-atomic-weights.htm)
- [NIST CODATA atomic mass constant](https://physics.nist.gov/cgi-bin/cuu/Value?u)
- [TBnew configuration and its gas-fraction/depletion convention](https://pulsar.sternwarte.uni-erlangen.de/wilms/research/tbabs/)
