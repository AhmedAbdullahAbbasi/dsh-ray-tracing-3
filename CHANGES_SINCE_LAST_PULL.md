# Local changes on `dsh` not yet in the repository

Summary written 2026-10-06 by Atakan Saraç. The branch is level with `origin/dsh` at `e701700` (last pull: 2026-10-05 12:08 +03).

Nothing under `dsh/`, `scripts/` or any other tracked path has been modified. The change is three new files that sit alongside the package and only call its public functions.

| File | What it is |
| --- | --- |
| `run_simulation.py` | Batch driver: runs one simulation from `sim_config.py` and writes images |
| `sim_config.py` | All run parameters for the driver |
| `submit_simulation.sh` | SLURM submission script for the driver |

Together they are a non-interactive way to run long GPU simulations on the cluster: edit `sim_config.py`, then run `bash submit_simulation.sh`.

## New capabilities

- **Photon count images.** Alongside the fluence image, each run now saves the number of scored Monte Carlo events per pixel, as PNGs (`outputs/dsh_count_image_<scale>.png`) and as a `(y, x)` integer matrix (`outputs/dsh_count_image.npy`). This shows directly how well each pixel is sampled.
- **Per-time-bin images.** Setting `IMAGE_EPOCH_DAYS = (start, stop)` restricts the images to the arrival-time bins inside that interval, down to a single bin (for example `(100.0, 101.0)` with one-day bins). The fluence image then becomes the mean specific intensity over that epoch, in ph cm⁻² s⁻¹ keV⁻¹ arcsec⁻², and the count image covers the same epoch. Titles give the epoch in days and, for a flux-file source, in MJD. A run produces images for one epoch; with `IMAGE_EPOCH_DAYS = None` it sums over all arrival times as before.

## `run_simulation.py`

- **Cloud:** either a `delta_NH` FITS cube (`CLOUD_FITS_PATH`) or a uniform-density disc built from the grid and disc parameters in the config.
- **Source models:** `constant-flare`, `exponential-decay`, `custom` (time edges and fluxes typed into the config) and `flux-file` (a light curve read from a text file of MJD plus flux columns, each row's flux held until the next row's MJD).
- **Energy bands:** with `ENERGY_EDGES_KEV = None`, photons are emitted at exactly 3.3/4.9/6.9 keV using the newdust and photoelectric tables. With band edges set, the driver switches to `load_2_10_material_tables()`, draws energies from a power law (`PHOTON_INDEX`) inside each band, and uses the photon-weighted mean energy as the band's effective energy. This mode needs `SOURCE_MODEL = "flux-file"`.
- **Device:** `DEVICE` (`"cpu"` or `"gpu"`) sets `JAX_PLATFORMS` before JAX is imported.
- **Progress:** one line per `PROGRESS_EVERY_CHUNKS` chunks to stdout and `logs/progress.log`, with elapsed time and ETA.
- **Diagnostics:** transport terminal states, interaction and scattering counts, scored against binned fluence, and how many events fell outside the sky, energy or arrival-time bins.
- **Outputs**, one set per entry in `IMAGE_SCALES` (log and linear):
  - `outputs/dsh_image_<scale>.png`: fluence summed over all arrival times, or mean specific intensity over `IMAGE_EPOCH_DAYS` when an epoch is set.
  - `outputs/dsh_count_image_<scale>.png`: scored Monte Carlo events per pixel.
  - `outputs/dsh_count_image.npy`: the count image as a `(y, x)` integer matrix.
- **Image options:** select one energy band or sum all, crop to a centred field of view, restrict to an arrival-time epoch.

## `sim_config.py`

A flat module of constants, grouped as geometry, disc cloud, source model, flux file, energy bands, custom light curve, Monte Carlo run, observer binning, progress log and output image. The values currently in it are those of the last cluster run:

- Cloud `clouds/4u1630_NH_111111111100001_DSH_11p5kpc.fits`, source at 11.5 kpc.
- Source from flux column 2 of `fluxes/1630_allflux.txt`, in a 2.25–3.15 keV band with photon index 2.
- 10⁹ packets in chunks of 10,000, at most 4 interactions, seed 2026, on GPU.
- 200 one-day arrival-time bins.

## `submit_simulation.sh`

- Requests 1 node, 4 CPUs, 16 GB and 24 hours, and writes `logs/dsh_sim_<jobid>.out`.
- Reads `DEVICE` from `sim_config.py` and, for GPU runs, adds `--gres=gpu:1 --exclude=cn05,cn06` (those nodes carry Tesla K80s, too old for JAX's CUDA 12 build).
- Works with either `bash` or `sbatch`. Run with `bash`, it submits itself with the right options. Submitted with plain `sbatch` and given no GPU, it resubmits itself once with the GPU request.
- Hardcodes values specific to my cluster account: the SLURM account and partition (`mdbf`), a Python path under `/cta/users/atakans/` and a notification email address.

## Not included

The config points at two inputs that are not part of this change: the trimmed cloud cube under `clouds/` (FITS files are ignored by git) and the light curve `fluxes/1630_allflux.txt`.
