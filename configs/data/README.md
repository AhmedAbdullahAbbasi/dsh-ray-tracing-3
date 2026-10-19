# Synthetic configuration inputs

These tiny files support the runnable `configs/example_*.toml` integration
examples. They are synthetic and have no observational provenance. Source
fluxes are intrinsic observer-equivalent band-integrated photons in
ph cm^-2 s^-1. They are not count rates or energy fluxes.

| File | Contract |
| --- | --- |
| `example_flux.txt` | Whitespace rows: MJD(TT), ignored ancillary column, F(2–4 keV), F(4–6 keV), F(6–10 keV). Times 60000, 60000.5, 60001; flux rows `[.020,.012,.006]`, `[.010,.006,.003]`, `[.005,.003,.0015]`. TOML holds each row to the next, then adds a final 0.5 day interval. |
| `example_history.json` | Version-1 spectral history. Synthetic epoch MJD 60000, TT; relative edges `[0,43200,86400]` s; energy edges `[2,4,6,10]` keV. Two independently integrated band-flux rows and per-cell photon indices. |
| `example_cloud.fits` | Native column cube, NumPy shape `(z,y,x)=(4,9,9)`, FITS axes `(x,y,z)`. `BUNIT=cm-2`; `CTYPE1/2=XOFFSET/YOFFSET` in arcsec and `CTYPE3=DISTANCE` in kpc. |

The FITS recipe is intentionally simple and recorded in its header. Sky
centers are `-80,-60,...,80` arcsec on each axis; radial centers are
`[2,3,4,5]` kpc. The four spatially uniform planes contain
`[1e20,2e20,1e20,0]` cm^-2 per voxel, stored as float32. Every sightline
therefore has total column `4e20` cm^-2 within float32 rounding. Radial cell
edges are `[1.5,2.5,3.5,4.5,5.5]` kpc. The example TOML selects a 10 kpc
source explicitly. The header's `DSOURCE` documents that intended value; the
configured runner uses the TOML selection.

Sky coordinates already use the source sightline as zero. The loader does
not recenter from an image pixel or celestial WCS. FITS checksums are included.
The file is intentionally committed as a tiny fixture despite the repository's
general exclusion of large FITS inputs. Tests independently check its axes,
column sum and `n_H * delta_r` closure.
