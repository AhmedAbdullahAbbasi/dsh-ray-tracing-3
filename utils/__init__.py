"""Tiny helper package for the Monte Carlo radiative transfer notebook.

Everything here is intentionally modular: physical source-flux sampling
(`source.py`), importance-sampled physical source launch (`source_launch.py`),
source-cloud-observer coordinates (`coordinates.py`), native angular-distance
cloud input (`clouds.py`), exact ray-column integration (`ray_integrals.py`),
one-event physical Monte Carlo validation (`first_interaction.py`), random
interaction sampling (`sampling.py`), native voxel transport
(`voxel_transport.py`), next-event observer scoring (`observer.py`), legacy
source-packet transport and JAX parallelization (`transport.py`), weighted
observer-product binning (`observer_binning.py`), voxel geometry, imaging,
complete ideal-observer simulation orchestration (`simulation.py`), plotting,
and device reporting.
"""
