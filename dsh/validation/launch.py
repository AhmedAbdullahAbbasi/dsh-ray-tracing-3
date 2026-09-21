"""Variance-reduced source launches for controlled uniform-screen experiments.

The wide rectangular cone retains support over the entire cloud.  Two nested
cones concentrate photons at the forward scattering angles that dominate a
DSH image.  Every sampled direction carries the *full mixture density* so
the production peel-off scorer stays unbiased and needs no special handling.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import random

from ..sources.launch import (
    SourceLaunchGeometry,
    build_rectangular_launch_geometry,
    sample_source_launches,
)
from ..sources.models import SourcePackets

MIXTURE_PROBABILITIES = (0.80, 0.15, 0.05)


def nested_screen_launch_geometries(
    outer: SourceLaunchGeometry,
    *,
    source_distance_kpc: float,
    fractional_distance: float,
    median_scattering_angle_rad: float,
) -> tuple[SourceLaunchGeometry, SourceLaunchGeometry, SourceLaunchGeometry]:
    """Choose two central launch cones inside the full screen launch cone.

    A launch angle of approximately ``x * scattering_angle`` illuminates the
    observed halo.  The full outer cone remains in the mixture and guarantees
    coverage of every part of the rectangular screen.
    """

    edge = min(
        float(np.min(np.abs(np.asarray(outer.slope_x_bounds)))),
        float(np.min(np.abs(np.asarray(outer.slope_y_bounds)))),
    )
    characteristic = fractional_distance * median_scattering_angle_rad
    inner = min(0.40 * edge, 2.5 * characteristic)
    middle = min(0.80 * edge, max(2.0 * inner, 8.0 * characteristic))
    if not (0.0 < inner < middle < edge):
        raise ValueError("outer launch cone must be symmetric and wider than the halo")
    geometries = []
    for width in (inner, middle):
        geometries.append(
            build_rectangular_launch_geometry(
                source_distance_kpc, [-width, width], [-width, width]
            )
        )
    return geometries[0], geometries[1], outer


def sample_mixture_source_launches(key, packets: SourcePackets, geometries):
    """Draw source directions from nested boxes with the exact joint PDF.

    ``q(Omega) = (1+u**2+v**2)**(3/2) * sum_k p_k 1_box_k / area_k``.
    This replaces the sampler's single-box importance factor by
    ``1/(4*pi*q)``.  Other packet fields retain their original values.
    """

    if len(geometries) != 3:
        raise ValueError("the validation proposal requires three nested boxes")
    selection_key, *launch_keys = random.split(key, 4)
    candidates = [
        sample_source_launches(launch_key, packets, geometry)
        for launch_key, geometry in zip(launch_keys, geometries, strict=True)
    ]
    choice = random.categorical(
        selection_key,
        jnp.log(jnp.asarray(MIXTURE_PROBABILITIES, dtype=jnp.float32)),
        shape=(packets.energy_kev.shape[0],),
    )
    momentum = jnp.where(
        (choice == 0)[:, None],
        candidates[0].momentum_kev,
        jnp.where(
            (choice == 1)[:, None],
            candidates[1].momentum_kev,
            candidates[2].momentum_kev,
        ),
    )
    u = -momentum[:, 2] / momentum[:, 1]
    v = -momentum[:, 3] / momentum[:, 1]
    jacobian = (1.0 + u * u + v * v) ** 1.5
    slope_pdf = jnp.zeros_like(u)
    for probability, geometry in zip(MIXTURE_PROBABILITIES, geometries, strict=True):
        inside = (
            (u >= geometry.slope_x_bounds[0])
            & (u <= geometry.slope_x_bounds[1])
            & (v >= geometry.slope_y_bounds[0])
            & (v <= geometry.slope_y_bounds[1])
        )
        slope_pdf += probability * inside / geometry.slope_area
    q_per_sr = jacobian * slope_pdf
    return candidates[-1]._replace(
        momentum_kev=momentum,
        launch_pdf_per_sr=q_per_sr,
        isotropic_importance=1.0 / (4.0 * jnp.pi * q_per_sr),
    )
