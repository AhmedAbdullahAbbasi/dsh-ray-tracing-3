# Matched NewDust thin-cloud benchmarks

Run this after installing the repository and the optional benchmark dependencies:

```bash
python -m pip install -e ".[benchmark]"
python benchmarks/newdust/compare_newdust.py --repo . --config benchmarks/newdust/configs/single_f0.25.json --output validation_outputs/radial_single_0p25
python benchmarks/newdust/compare_newdust.py --repo . --config benchmarks/newdust/configs/single_f0.5.json --output validation_outputs/radial_single_0p5
python benchmarks/newdust/compare_newdust.py --repo . --config benchmarks/newdust/configs/single_f0.5_high_energy.json --output validation_outputs/radial_single_0p5_high_energy
python benchmarks/newdust/compare_newdust.py --repo . --config benchmarks/newdust/configs/single_f0.75.json --output validation_outputs/radial_single_0p75
python benchmarks/newdust/compare_newdust.py --repo . --config benchmarks/newdust/narrow_screen_config.json --output validation_outputs/radial_narrow
python benchmarks/newdust/compare_newdust.py --repo . --config benchmarks/newdust/configs/multiple.json --output validation_outputs/radial_multiple
```

These cover f=d/D=0.25, 0.5, 0.75 at 3.3, 4.9, 6.9 keV. The narrowest
f=0.75, 6.9-keV case uses 10-arcsec annuli; the other cases use 20-arcsec
annuli. The f=0.5, 6.9-keV case uses six million histories per seed; the
other matrix cases use three million per seed, each with three seeds. The
original nine-million-history f=0.5, 6.9-keV trial did not meet the ten-bin
sampling gate and is retained as a diagnostic in the result archive.
The matched thin-cloud setup is D=11.5 kpc, total NH=8e20 cm^-2,
source fluence 1 ph cm^-2, 1-pc DSH screens, no absorption, all arrival times,
4-arcsec raw pixels, and a +/-600-arcsec field. Multiple-cloud runs split NH
equally among the three distances and retain explicit vacuum gaps. The two
convergence probes use thickness_half_config.json and column_half_config.json.

NewDust is distributed as xdust. The runner independently computes
ScreenGalHalo.norm_int for each cloud using x_ND=1-f. This already contains
column and geometry normalization; only source fluence multiplies it. No fitted
scale, unit-sum normalization, PSF, or detector counts are introduced. The
stored FITS values are ph cm^-2 per pixel, while radial surface fluence is
ph cm^-2 arcsec^-2. Monochromatic tests have no arbitrary energy-bin division.
Preview images sum the raw pixels into equal 20-arcsec display pixels.

The runner compares first-order scores. DSH retains incident and escaping
extinction, whereas NewDust's thin-screen profile is linear in column. The
aperture tolerance is 3 MC standard errors plus the recorded total tau,
conservative thickness/geometry allowances, and reference numerical error.
Resolved radial bins use 4 standard errors plus the same allowances. A case
also requires at least ten bins with effective sample size >=100, <=3% relative
aperture MC error, three seeds, seed consistency p>=0.001, reference refinement
<=0.3%, clean numerical terminal states, and first-order image coverage.
The geometry/thickness allowances are initial regression heuristics, not a
rigorous proof of convergence.

Each case additionally tests azimuth in radial windows 0-40, 40-80, 80-160,
160-320, and 320-600 arcsec. The masks are identical for both models. A window
is resolved when its effective sample size is >=300, all 16 sectors have
effective sample size >=15, and the four-dimensional covariance has full rank.
The joint m=2 and m=4 cosine/sine test uses covariance from weighted first-order
history contributions and p>=0.001 per resolved window. At least three windows
must resolve. The m=4 term directly probes a fourfold/square error. Unresolved
windows and sparse individual pixels remain diagnostic, even when a case passes.
This test does not establish convergence of every individual raw pixel.

All MC errors include histories with zero score. Each first-order history has
at most one contribution, so the saved square-weight moments support both
aperture and angular covariance calculations. Higher-order images are retained
without an uncertainty or validation claim in this harness.

The packaged differential law agrees directly with NewDust at the table energy
nodes. Its solid-angle integral is about 1.04463 times NewDust's separately
reported total RG/Drude coefficient. This pre-existing discrepancy is recorded;
neither image is rescaled using the reported total or Halo.fhalo.

Set launch_proposal to radial (default), nested_near, or nested_far in a config.
The latter two preserve the old nested-box proposal for controlled comparisons.
The radial proposal uses the public production sampler and its exact PDF;
the benchmark does not substitute a separate validation-only implementation.
Use --packets and --seed only for pilots or explicit count overrides. Exit 0
means every gate passed; 2 means a comparison or sampling gate did not pass.

Outputs include matched FITS/NPZ images, first-history error images, saved seed
moments, radial_profiles.csv, azimuth_profiles.csv, comparison.png,
azimuth_comparison.png, report.json, and source/configuration provenance. Reuse
of saved moments with reanalyze_saved.py requires matching physical parameters,
material checksum, histories, edges, and executed DSH module hashes.

```bash
python -m unittest tests.test_radial_source_launch -v
python -m unittest discover -s benchmarks/newdust -p "test_*.py" -v
```

Primary implementation reference:
https://github.com/eblur/xdust/blob/9e167c00f9af35a12cf7011056e54a9cf2b7a882/src/xdust/halos/galhalo.py
