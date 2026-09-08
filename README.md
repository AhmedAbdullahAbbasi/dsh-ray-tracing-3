# Basic Monte Carlo radiative transfer, parallelized with JAX

Open [`monte_carlo_rt.ipynb`](monte_carlo_rt.ipynb) for the simulation
walkthrough. Run [`physics_checkpoints.ipynb`](physics_checkpoints.ipynb) at
each development milestone: it states the analytic expectation, checks
numerical closure, and only then marks that physics stage as passing. Helper
code lives in [`utils/`](utils/).

## Source-flux convention

[`utils/source.py`](utils/source.py) provides the first physical input layer.
The primary test source is generic and separable,
`F(E,t) = F_band(t) S(E)`: its light curve is piecewise constant, while its
spectrum is sampled exactly from a power law. `F_band(t)` is the unabsorbed
observer-equivalent photon flux integrated over the simulated energy range in
`ph cm^-2 s^-1`. Every packet receives a real energy, emission time, and an
observer-fluence weight in `ph cm^-2`; packet weights sum to the integrated
fluence of the light curve.

```python
import jax
import numpy as np
from jax import random
from utils.source import (
    build_decay_observation_window,
    build_variable_powerlaw_source,
    fred_outburst_flux,
    sample_variable_powerlaw_source,
)

DAY = 86_400.0
time_edges_s = np.linspace(0.0, 120.0 * DAY, 241)
peak_time_s = 10.0 * DAY

# Generic fast-rise, exponential-decay X-ray transient.
photon_flux = fred_outburst_flux(
    time_edges_s,
    baseline_flux=1.0e-2,
    peak_excess_flux=9.0e-2,
    peak_time_s=peak_time_s,
    rise_time_s=2.0 * DAY,
    decay_time_s=25.0 * DAY,
)
source = build_variable_powerlaw_source(
    time_edges_s,
    photon_flux,
    energy_min_kev=1.0,
    energy_max_kev=10.0,
    photon_index=2.0,
)

# A 28.8 ks observation beginning 35 days after the outburst peak.
observation = build_decay_observation_window(
    start_s=45.0 * DAY,
    stop_s=45.0 * DAY + 28_800.0,
    outburst_peak_s=peak_time_s,
)

sample_jit = jax.jit(
    sample_variable_powerlaw_source,
    static_argnames=("n_packets",),
)
packets = sample_jit(random.PRNGKey(0), source, n_packets=100_000)
```

The same module retains a tabulated-band source for later observational
adapters. NaNs are rejected deliberately: missing intervals must be handled
by an explicit gap policy before source construction.

## Transporting physical source packets

`simulate_source_packets` connects those sampled packets to the existing JAX
voxel transport. It preserves each packet's physical energy in keV, emission
time, observer-fluence weight, and source-bin indices in the returned result.
Assuming `density_grid`, `box_min`, and `box_max` have already been prepared:

```python
from utils.transport import simulate_source_packets

transport_jit = jax.jit(
    simulate_source_packets,
    static_argnames=(
        "n_bounces",
        "n_substeps",
        "scattering_model",
        "grain_radius_um",
    ),
)
transported = transport_jit(
    random.PRNGKey(1),
    packets,
    n_bounces=4,
    n_substeps=256,
    density_grid=density_grid,
    box_min=box_min,
    box_max=box_max,
    beam_radius=0.0,
    scattering_model="mie",
    grain_radius_um=0.1,
)

# Transport outputs and unchanged physical source metadata:
transported.path
transported.energy_kev
transported.emission_time_s
transported.weight_observer_fluence
```

The Mie option is still a qualitative Henyey-Greenstein placeholder. Its
dimensionless grain-size parameter is now derived internally from physical
keV energy instead of replacing the packet's energy field. Arrival-time
selection is intentionally deferred: the next layer must add the geometric
excess-path delay to `emission_time_s` before applying `observation`.

## Native cloud-input convention

`utils/clouds.py` is the physical adapter for an angular--distance hydrogen
column cube. The native array order is `(z, y, x)`, where each value is the
column increment contributed by one radial voxel, `delta_NH` in `cm^-2`. It
is not a Cartesian density and is never rescaled to an arbitrary peak. The
adapter derives the radial-average number density through

```text
n_H [cm^-3] = delta_NH [cm^-2] / delta_r [cm]
```

and requires all radial cell edges to lie between observer and source. Input
axes may increase or decrease; decreasing FITS axes are flipped together with
the data so the physical scene is unchanged. The JAX-ready object retains the
native frustum axes, original column increments, derived number density, and
source distance.

```python
from utils.clouds import cloud_from_loaded_fits, total_column_map_cm2
from utils.fits_cube import load_cube

loaded = load_cube("my_delta_nh_cube.fits")
cloud = cloud_from_loaded_fits(loaded, source_distance_kpc=10.0)
nh_map = total_column_map_cm2(cloud)
```

This milestone deliberately stops before off-axis ray traversal. Its required
physics checks are voxel column closure and Beer--Lambert closure; both are in
`physics_checkpoints.ipynb` and `tests/test_clouds.py`.

## Physical coordinate convention

[`utils/coordinates.py`](utils/coordinates.py) defines the geometry used by
all forthcoming physical transport and delay calculations:

- Cartesian vectors are ordered `(line_of_sight_pc, sky_x_pc, sky_y_pc)`.
- The observer is at the origin.
- The source lies on the positive line-of-sight axis.
- Source photons initially travel toward the negative line-of-sight axis.
- Cartesian lengths are always parsecs; radial inputs are kiloparsecs; sky
  offsets are arcseconds.

```python
from utils.coordinates import build_sightline_geometry, sky_position_pc

geometry = build_sightline_geometry(source_distance_kpc=10.0)
cloud_center_pc = sky_position_pc(
    distance_kpc=5.3,
    sky_x_arcsec=30.0,
    sky_y_arcsec=-15.0,
)
```

The angular-distance FITS cube is a frustum: the physical width of an angular
pixel increases with distance. The legacy `voxels.from_fits_cube` function
only rescales that data into a cubic toy box and must not be used for physical
time delays. `utils.clouds` now preserves the native frustum and its column;
the next transport milestone will traverse that geometry directly.

Run the automated source and transport checks from the repository root with:

```bash
python -m unittest discover -v
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install matplotlib jupyter astropy

# then install the JAX build that matches your GPU:
pip install jax-metal "jax==0.4.34" "jaxlib==0.4.34"   # Apple Silicon Mac (Metal backend)
pip install -U "jax[cuda12]"                             # Nvidia GPU (CUDA backend)
pip install jax jaxlib                                   # no GPU / CPU only

jupyter notebook monte_carlo_rt.ipynb
```

JAX picks the fastest backend it finds automatically — the notebook code itself doesn't change between a Mac GPU, an Nvidia GPU, or CPU. Cell 0 of the notebook prints which one it's using.

**Mac GPU note:** `jax-metal`'s PyPI metadata only declares a *minimum* JAX version, so a plain `pip install jax-metal` will happily pull in the newest `jax`/`jaxlib` — which the actual compiled Metal plugin (last updated by Apple for the 0.4.34-era StableHLO format) can't run, and you'll see `JaxRuntimeError: ... unknown attribute code ...`. Pin `jax==0.4.34 jaxlib==0.4.34` alongside `jax-metal` as shown above to avoid it.

## Using a real column-density cube

[`utils/fits_cube.py`](utils/fits_cube.py) reads a FITS column-density cube (extract/print/plot only) and [`utils/voxels.py`](utils/voxels.py)'s `from_fits_cube` turns one into a `simulate_photons`-ready density grid. FITS files aren't tracked in this repo (too large for a normal git push) — supply your own cube (a `(n_dist, n_y, n_x)` primary HDU with linear `CRPIX`/`CRVAL`/`CDELT` axis keywords) and point `fits_cube.load_cube(...)` at it.

For physical work, pass the loaded dictionary to
`clouds.cloud_from_loaded_fits`; it preserves the native frustum and physical
column. `voxels.from_fits_cube` remains a legacy visualization adapter: it
downsamples and rescales the array and must not be used for optical depth,
path length, or arrival-time calculations.
