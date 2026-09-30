# Monochromatic flare through the realistic four-cloud test cube

Use a clean checkout of `refactor/inputs-and-config`. The earlier checkout
containing uncommitted Stage 9F files does not need to be switched or cleaned.
The existing `realistic_nh_cube_500asec_10kpc_dz0p05.fits` has 200 radial
cells over 10 kpc and a 500×500 arcsecond field. Its outer radial face is
10 kpc, so place the source strictly farther away; this setup uses 10.5 kpc.
The cube is synthetic gas, not a measured column map.

From PowerShell in the clean refactor checkout:

```powershell
python -m scripts.prepare_realistic_line_run `
  --cloud-fits 'C:\path\to\realistic_nh_cube_500asec_10kpc_dz0p05.fits' `
  --output-dir 'outputs\realistic_line_5p35'

python -m dsh.command check 'outputs\realistic_line_5p35\pilot.toml'
python -m dsh.command check 'outputs\realistic_line_5p35\full.toml'
python -m dsh.command run 'outputs\realistic_line_5p35\pilot.toml'
python -m dsh.command audit 'outputs\realistic_line_5p35\pilot.toml'
```

The generated source is one hour at exactly 5.35 keV with an illustrative
intrinsic photon flux of `0.038 ph cm^-2 s^-1`, or `136.8 ph cm^-2` total
fluence. The default pilot uses 8,192 packets; the full run uses 2,500,000,
in chunks of 256 and 512 respectively, with a 32-interaction cap. The
source, material tables, cloud and run settings are recorded in schema-7
outputs. The observer grid has one energy bin `[5.25,5.45]` keV and arrival
time edges `[0,3,4,6,7,9,10,60]` days. In particular, it preserves the
one-day windows `[3,4)`, `[6,7)` and `[9,10)` for the three requested images.

Inspect `pilot.run_report.json` and `pilot.audit_report.json`. Both
`numerical_passed` and product integrity must be true before committing
resources to the full run. Sparse or empty individual one-day images in an
8,192-packet pilot are possible; they do not establish the reliability of a
2,500,000-packet image. The `dsh check` step loads the actual cloud and
verifies the native angular/distance WCS, column units, source position, and
material support. It will reject an outer cloud face at or beyond 10.5 kpc.

Once the pilot passes:

```powershell
python -m dsh.command run 'outputs\realistic_line_5p35\full.toml'
python -m dsh.command audit 'outputs\realistic_line_5p35\full.toml'
python -m dsh.command snapshot 'outputs\realistic_line_5p35\full.toml'
```

The extractor audits the full source, cloud, materials, NPZ and FITS again
before writing `snapshots/line_day_003_to_004.fits`, `...006_to_007.fits`,
and `...009_to_010.fits` with a JSON manifest. Each FITS primary is
ideal-observer fluence in `ph cm^-2` per pixel; `FIRSTIMG`, `MULTIIMG`,
`EVENTIMG` and `STDIMG` record order split, scored event count and per-pixel
photon-history standard error. `COARSEFL` and `COARSEEV` sum 20×20 native
pixels for the 500×500 cube; no coarse uncertainty is inferred without the
spatial history covariance. An empty snapshot is written as an empty image
and reported with zero events. It is not a science detection.

The earlier `python -m scripts.extract_configured_line_snapshots --config ...`
command remains a compatibility entry point to the same extractor.

The file-input audit verifies product integrity and numerical terminal
states. The independent Stage 9F comparison establishes the controlled
finite-cone reference at 5.35 keV, but the realistic full-cloud cone and
one-day images still require packet-count/seed convergence and checks of
their spatial and time-bin uncertainties. Use the first full run as a pilot
for those diagnostics before interpreting morphology or flux physically.
