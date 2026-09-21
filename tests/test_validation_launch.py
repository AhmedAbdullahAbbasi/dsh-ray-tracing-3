"""Measure validation-proposal normalization without running photon transport."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from dsh.sources.launch import (
    build_rectangular_launch_geometry,
    sample_source_launches,
)
from dsh.sources.models import SourcePackets
from dsh.validation.launch import (
    MIXTURE_PROBABILITIES,
    nested_screen_launch_geometries,
    sample_mixture_source_launches,
)


class TestMixtureLaunch(unittest.TestCase):
    def test_overlap_pdf_and_forward_phase_estimator(self):
        n = 20_000
        packets = SourcePackets(
            energy_kev=jnp.full(n, 3.3),
            emission_time_s=jnp.zeros(n),
            weight_observer_fluence=jnp.ones(n),
            time_index=jnp.zeros(n, dtype=jnp.int32),
            spectral_bin_index=jnp.zeros(n, dtype=jnp.int32),
        )
        outer = build_rectangular_launch_geometry(10.0, [-0.01, 0.01], [-0.01, 0.01])
        geometries = nested_screen_launch_geometries(
            outer,
            source_distance_kpc=10.0,
            fractional_distance=0.5,
            median_scattering_angle_rad=0.0006,
        )
        mixed = jax.jit(sample_mixture_source_launches)(
            random.PRNGKey(33), packets, geometries
        )
        uniform = jax.jit(sample_source_launches)(random.PRNGKey(34), packets, outer)

        def contribution(sample):
            p = np.asarray(sample.momentum_kev)
            u, v = -p[:, 2] / p[:, 1], -p[:, 3] / p[:, 1]
            norm_cubed = (1.0 + u * u + v * v) ** 1.5
            expected_q = np.zeros(n)
            for probability, geometry in zip(
                MIXTURE_PROBABILITIES, geometries, strict=True
            ):
                bounds_u = np.asarray(geometry.slope_x_bounds)
                bounds_v = np.asarray(geometry.slope_y_bounds)
                inside = (
                    (u >= bounds_u[0])
                    & (u <= bounds_u[1])
                    & (v >= bounds_v[0])
                    & (v <= bounds_v[1])
                )
                expected_q += probability * inside / float(geometry.slope_area)
            expected_q *= norm_cubed
            np.testing.assert_allclose(sample.launch_pdf_per_sr, expected_q, rtol=1e-6)
            np.testing.assert_allclose(
                sample.isotropic_importance,
                1.0 / (4.0 * np.pi * expected_q),
                rtol=6e-7,
            )
            # Gaussian is a separate analytic proxy for a forward phase
            # function, evaluated on exactly the same rectangular domain.
            width = 0.0003
            forward = np.exp(-0.5 * (u * u + v * v) / width**2)
            return forward / expected_q

        mixed_weights = contribution(mixed)
        # Uniform proposals have one component; use their stored PDF directly.
        p = np.asarray(uniform.momentum_kev)
        u, v = -p[:, 2] / p[:, 1], -p[:, 3] / p[:, 1]
        uniform_weights = np.exp(-0.5 * (u * u + v * v) / 0.0003**2) / np.asarray(
            uniform.launch_pdf_per_sr
        )
        mixed_eff = mixed_weights.sum() ** 2 / np.sum(mixed_weights**2)
        uniform_eff = uniform_weights.sum() ** 2 / np.sum(uniform_weights**2)
        self.assertGreater(mixed_eff, 20.0 * uniform_eff)
        # Integral of exp(-r^2/(2 sigma^2)) over solid angle at small angle.
        self.assertAlmostEqual(
            np.mean(mixed_weights) / (2.0 * np.pi * 0.0003**2), 1.0, delta=0.06
        )


if __name__ == "__main__":
    unittest.main()
