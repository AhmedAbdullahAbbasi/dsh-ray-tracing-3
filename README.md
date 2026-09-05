# Basic Monte Carlo radiative transfer, parallelized with JAX

Open [`monte_carlo_rt.ipynb`](monte_carlo_rt.ipynb) — it's self-contained and explains the physics and the JAX parallelization as you go. Helper code lives in [`utils/`](utils/).

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
