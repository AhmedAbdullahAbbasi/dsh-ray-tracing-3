# Basic Monte Carlo radiative transfer, parallelized with JAX

Open [`monte_carlo_rt.ipynb`](monte_carlo_rt.ipynb) — it's self-contained and explains the physics and the JAX parallelization as you go. Helper code lives in [`utils/`](utils/).

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
by an explicit gap policy before source construction. The current toy
transport has not yet been connected to either source model.

Run the automated source checks from the repository root with:

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

## Using a real density cube

[`utils/fits_cube.py`](utils/fits_cube.py) reads a FITS column-density cube (extract/print/plot only) and [`utils/voxels.py`](utils/voxels.py)'s `from_fits_cube` turns one into a `simulate_photons`-ready density grid. FITS files aren't tracked in this repo (too large for a normal git push) — supply your own cube (a `(n_dist, n_y, n_x)` primary HDU with linear `CRPIX`/`CRVAL`/`CDELT` axis keywords) and point `fits_cube.load_cube(...)` at it.
