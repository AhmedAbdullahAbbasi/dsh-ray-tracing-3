"""Tiny helper package for the Monte Carlo radiative transfer notebook.

Everything here is intentionally modular: physical source-flux sampling
(`source.py`), source-cloud-observer coordinates (`coordinates.py`), native
angular-distance cloud input (`clouds.py`), exact ray-column integration
(`ray_integrals.py`), one-event physical Monte Carlo validation
(`first_interaction.py`), random interaction sampling (`sampling.py`),
source-packet transport and JAX parallelization (`transport.py`), voxel
geometry, imaging, plotting, and device reporting.
"""
