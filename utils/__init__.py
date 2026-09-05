"""Tiny helper package for the Monte Carlo radiative transfer notebook.

Everything here is intentionally minimal: a couple of random-sampling
functions (`sampling.py`), the photon physics and the JAX
vmap/scan machinery that parallelizes it (`transport.py`), some
matplotlib helpers (`plotting.py`), and a one-function device check
(`device.py`).
"""
