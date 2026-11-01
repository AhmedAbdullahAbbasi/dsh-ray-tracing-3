# Windows acceptance: material-inventory roundoff

The uploaded Windows report at `80d8d5f` passes both 54-case Compton suites,
the 44-case float64 physics suite and regression groups 1 and 2. Regression
group 3 fails one historical material-comparison assertion:

`test_full_mie_composition_preserves_hybrid_inventory_and_total_opacity`

It compares derived inventory dictionaries exactly. Windows produced
`0.00011999999999999999` versus `0.00012000000000000003` atoms/H, a difference
of approximately 4.1e-20 atoms/H, or 3.4e-16 relative to that grain inventory.
The original test is frozen benchmark evidence and remains unchanged.

The current acceptance command selects a portable variant that inherits all
five original test cases. Only the inventory assertion changes: element names,
field names and total elemental abundances remain exact. Derived gas/solid
amounts allow 32 ulps of the element's total abundance; gas fractions allow
32 ulps at unity. Nonfinite values fail. This is the same roundoff scale used
by the production material audit, without changing that audit or any physics.
The original total-opacity and distinct-definition assertions remain intact.

An added guard reproduces the reported rounding pair and verifies that changed
abundances, 1e-10 relative inventory shifts, changed gas fractions, NaNs and
unexpected fields still fail. All 146 frozen files and their manifests remain
unchanged. The validator reports the active test mapping explicitly. Running
the frozen module directly still runs its historical exact-equality assertion;
use `scripts.validate_configured_compton` for current acceptance.

After applying the hotfix, rerun:

```powershell
python -m scripts.validate_configured_compton
```

This is a validation portability correction. No transport, Compton prescription,
material table, source normalization or saved-product format changes.
