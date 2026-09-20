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

The native-frustum kernel in `utils.voxel_transport` also returns a fixed-size
interaction history.  Every slot records whether an interaction occurred, its
type, position, incoming and outgoing four-momenta, cumulative path and excess
path, and scattering order.  Unused slots are explicitly masked and zeroed.
These records feed the observer/peel-off estimator described below; crossing
the observer plane by itself is not treated as a telescope detection.

## Physical source launch

`utils/source_launch.py` converts physical `SourcePackets` into the positions
and null four-momenta accepted by the native-frustum transport kernel. Version
1 treats the source as an isotropic point at the central source position. It
importance-samples only a conservative rectangular cone enclosing the cloud
frustum instead of wasting packets over the full sphere.

```python
from jax import random
from utils.source_launch import (
    build_cloud_launch_geometry,
    sample_source_launches,
)
from utils.voxel_transport import transport_photon_batch

launch_geometry = build_cloud_launch_geometry(cloud)
launched = sample_source_launches(random.PRNGKey(3), packets, launch_geometry)
transported = transport_photon_batch(
    random.PRNGKey(4),
    launched.position_pc,
    launched.momentum_kev,
    cloud,
    physics,
)
```

Launch directions are uniform in tangent-plane slopes `(u, v)`, so their
solid-angle density is evaluated with the exact Jacobian,
`q(Omega) = (1 + u^2 + v^2)^(3/2) / slope_area`. The result retains both
`q(Omega)` and the isotropic-source importance factor `1/(4*pi*q)` without
modifying the packet's observer-fluence weight. This makes the absolute halo
normalization independent of the chosen enclosing cone. For a scattering
event at observer distance `r`, the peel-off scorer uses

```text
w_event = w_observer * (D / r)^2
          * 4*pi*isotropic_importance * phase_pdf * transmission,
```

where `D` is the source distance and `phase_pdf = (d sigma/d Omega)/sigma_sca`.
The cloud's outer radial edge must lie strictly in front of the point source;
a frustum touching the source has no finite rectangular tangent-plane cone.

## Peel-off observer scoring

`utils/observer.py` turns every recorded dust scattering into one virtual
observer photon. This next-event estimator is necessary because a point
observer subtends essentially zero solid angle: waiting for an analog
scattered direction to hit it would be impractical. The scorer instead aims a
virtual ray from each event toward the observer and evaluates its statistical
weight without changing the actual outgoing direction used for later
multiple scattering.

```python
from utils.observer import score_peeloff_events

observer_events = score_peeloff_events(
    launched,
    transported,
    cloud,
    physics,
)

observer_events.valid
observer_events.sky_x_arcsec
observer_events.sky_y_arcsec
observer_events.arrival_time_s
observer_events.weight_observer_fluence
observer_events.scattering_order
```

The phase density is the normalized intrinsic NewDust quantity
`(d sigma/d Omega)/sigma_sca`; the analog transport already sampled the
scattering opacity, so the scorer does not multiply by `sigma_sca` again. The
remaining event-to-observer column is integrated exactly and attenuated by

```text
exp[-N_H,event->observer * (sigma_sca + sigma_abs)].
```

Arrival times use the exact geometric excess path. Both arcsecond scattering
angles and small delays are evaluated with float32-stable identities. Output
arrays have shape `(n_packets, max_interactions)`; absorption events and unused
slots are masked and exactly zero. Image, energy, and arrival-time binning are
handled by the next layer.

## Weighted DSH observer products

`utils/observer_binning.py` converts the fixed-shape peel-off event arrays into
physical four-dimensional DSH cubes. The canonical array order is
`(arrival time, energy, sky y, sky x)`, and each cell contains observer fluence
in `ph cm^-2`. First-scatter and multiple-scatter fluences are retained
separately, alongside their sum and the raw Monte Carlo event count.

```python
from utils.observer_binning import (
    bin_observer_events,
    build_observer_bin_geometry,
    fluence_surface_brightness_per_sr,
)

image_bins = build_observer_bin_geometry(
    sky_x_edges_arcsec=[-120.0, -60.0, 0.0, 60.0, 120.0],
    sky_y_edges_arcsec=[-120.0, -60.0, 0.0, 60.0, 120.0],
    energy_edges_kev=[3.0, 4.0, 6.0, 8.0],
    arrival_time_edges_s=[0.0, 86_400.0, 172_800.0],
)
products = bin_observer_events(observer_events, image_bins)
surface_brightness = fluence_surface_brightness_per_sr(
    products.total_fluence,
    image_bins,
)
```

All axes use left-inclusive, right-exclusive bins, except that the final upper
edge is included. Sky-pixel solid angles are calculated exactly for the
gnomonic angular coordinates rather than assumed to be constant. The result
also records valid, binned, and out-of-range event counts and weights, making
fluence loss at requested image, energy, or time limits explicit. Independent
packet chunks can be combined with `add_binned_observer_products` without
retaining their event arrays. This stage intentionally applies no detector
response, exposure map, point-spread function, background, or Poisson draw.

## Complete Version-1 ideal-observer pipeline

`utils/simulation.py` composes the independently tested layers into one
source-to-observer calculation:

```text
source sampling -> importance launch -> voxel transport
                -> peel-off scoring -> weighted DSH binning
```

Both tabulated-band and variable power-law sources have JIT-compatible
single-batch entry points. Production-sized calculations should use the
host-side chunked entry points, which retain only the accumulated cubes and
diagnostics. Packet weights are normalized to the requested total packet
count across all chunks, so chunking does not duplicate the source fluence.
The result reports every transport terminal state, the number of analog
interactions and scatterings, scored observer-event totals, and the binned
closure fields from the preceding section.

The first local end-to-end smoke run uses a 10 kpc source, a synthetic
500-arcsec four-cloud plus diffuse-H scene, the Version-1 energies
3.3/4.9/6.9 keV, multiple scattering, and an eight-interaction safety guard:

```powershell
python -m scripts.run_dsh_v1
```

The default is deliberately moderate: 4096 packets in chunks of 256. It
writes `outputs/dsh_v1_ideal_observer.npz`, containing total, first-scatter,
and multiple-scatter fluence cubes; event counts; all bin edges and solid
angles; transport-state counts; closure diagnostics; and run metadata. Once
that smoke run is clean, increase the Monte Carlo statistics explicitly:

```powershell
python -m scripts.run_dsh_v1 --packets 100000 --chunk-size 1024
```

A physical `delta_NH` FITS cube can replace the built-in scene:

```powershell
python -m scripts.run_dsh_v1 `
  --cloud-fits "C:\path\to\cloud.fits" `
  --source-distance-kpc 10.5
```

The source distance must lie strictly beyond the FITS cube's outer radial
cell edge so the point-source launch cone remains finite. This Version-1
runner produces ideal-observer physical fluence only. Telescope effective
area, PSF, exposure maps, detector energy redistribution, background, and
counting noise are deferred to Version 2.

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

`utils/ray_integrals.py` integrates columns along arbitrary finite straight
rays through this frustum. It solves intersections with radial spheres and
angular boundary planes analytically, sorts those crossings, and sums
`n_H * ds` within each crossed cell. It therefore has no fixed spatial-step
error. Checkpoint 4 requires every radial pixel sightline to reproduce the
native `sum(delta_NH)` map and verifies a ray outside the field returns zero.

## One-event Monte Carlo checkpoint

`utils/first_interaction.py` connects physical source packets to the native
cloud geometry without introducing a dust-model approximation. For each ray
it integrates the available column exactly, samples a target optical depth
`-log(xi)`, inverts cumulative column to obtain the interaction position, and
branches between scattering and absorption according to their opacity ratio.
It deliberately stops after one event.

```python
from jax import random
from utils.first_interaction import simulate_first_interactions

result = simulate_first_interactions(
    random.PRNGKey(2), packets, cloud,
    origin_pc=[10_000.0, 0.0, 0.0],
    direction=[-1.0, 0.0, 0.0],
    max_distance_pc=10_000.0,
    scattering_cross_section_cm2_per_h=3.0e-22,
    absorption_cross_section_cm2_per_h=2.0e-22,
)
```

The temporary scattered direction is isotropic. That is a test phase function,
not a physical X-ray dust kernel. Checkpoint 5 compares outcome fractions with
the exact finite-slab probabilities, verifies the truncated exponential
free-path law and exact event locations, and checks conservation of all source
packet metadata. Checkpoint 6 will replace this temporary angular sampler with
a normalized energy-dependent differential dust-scattering cross-section.

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
`utils.ray_integrals` traverses that geometry directly.

Run the automated source and transport checks from the repository root with:

```bash
python -m unittest discover -v
```

## NewDust scattering table

The Version-1 dust phase function is precomputed with NewDust/xdust and stored
as an intrinsic differential cross-section per H atom. It uses the legacy MRN
power-law grain population, Rayleigh-Gans scattering with the Drude
approximation, and representative energies 3.3, 4.9, and 6.9 keV. The table is
loaded and validated on the host; the resulting fixed arrays are then passed
to the JAX transport kernel.

```python
from utils.newdust import (
    build_dust_physics_from_tables,
    load_newdust_scattering_table,
)
from utils.absorption import load_photoelectric_absorption_table

scattering = load_newdust_scattering_table()
absorption = load_photoelectric_absorption_table()
physics = build_dust_physics_from_tables(scattering, absorption)
```

The checked-in absorption table is source-independent XSPEC ``tbabs``
physics: it stores ``sigma_abs(E)`` in ``cm^2/H`` for the same representative
energies, using TBabs version 2 and Wilms abundances.  It is not a table of
broad-band transmissions.  A band transmission depends on the source
spectrum and changes with column as the transmitted spectrum hardens, while
the transport material coefficient obeys

```text
T(E, NH) = exp[-NH * sigma_abs(E)].
```

Version 1 requires the absorption and scattering energy grids to match
exactly.  Their schemas already accept arbitrary one-dimensional energy axes,
so both can later be regenerated on the same dense grid without changing the
voxel-transport API.

The supplied legacy `x_0_007.dat`--`x_0_010.dat` files are not intrinsic
cross-sections. They are the 3.3-keV thin-screen halo kernel and already contain
the `(1-f)^-2` observer geometry, where `f=d_dust/d_source`. The regression
tests reconstruct those files from the intrinsic table and reapply the
geometry exactly once.

Regenerating the table is an offline provenance task, not a runtime step:

```bash
pip install xdust astropy scipy
python scripts/generate_newdust_table.py
```

The generator fixes the grain-radius sampling explicitly because current
xdust defaults differ from the historical NewDust v1 defaults used to produce
the legacy kernels. The transport cross-section is the solid-angle integral
of the same tabulated differential cross-section used to build the phase CDF;
this gives exact opacity/phase-function closure and retains the legacy halo
normalization. The adjacent JSON records this choice, the complete
configuration, the original NewDust-reported `tau_sca`, and a checksum of the
compressed array file.

Regenerating the absorption table is also an offline provenance operation and
requires an initialized XSPEC/HEASoft environment:

```bash
python scripts/generate_tbabs_table.py
```

The generator evaluates ``tbabs*powerlaw`` and ``powerlaw`` in the same narrow
dummy-response bin.  Their ratio removes the dummy source and yields the
monochromatic cross-section.  It repeats the extraction at two reference
columns and refuses to write the table unless the inferred material
coefficient is column-independent.

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
