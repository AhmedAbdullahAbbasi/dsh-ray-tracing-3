"""Physical dust tables used by the DSH voxel-transport kernel.

The transport kernel needs intrinsic material quantities:

* total dust-scattering cross-section per H atom;
* total absorption cross-section per H atom; and
* a normalized CDF of the physical scattering angle.

The phase CDF is defined by

    C(E, theta) = (1 / sigma_sca) * integral_0^theta
                  2 pi sin(theta') (d sigma / d Omega) d theta'.

Observer-space DSH kernels must not be passed here.  In particular, tables
that already include the geometric ``(1-x)**-2`` factor are not intrinsic
cross-sections and must first be converted back to ``d sigma / d Omega`` as a
function of the physical scattering angle.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class DustPhysicsTable(NamedTuple):
    """Fixed-shape JAX arrays for energy-dependent DSH interactions."""

    energy_kev: jnp.ndarray
    scattering_cross_section_cm2_per_h: jnp.ndarray
    absorption_cross_section_cm2_per_h: jnp.ndarray
    scattering_angle_rad: jnp.ndarray
    scattering_angle_cdf: jnp.ndarray
    differential_cross_section_cm2_per_sr_per_h: jnp.ndarray


def build_dust_physics_table(
    energy_kev,
    scattering_cross_section_cm2_per_h,
    absorption_cross_section_cm2_per_h,
    scattering_angle_rad,
    scattering_angle_cdf,
    differential_cross_section_cm2_per_sr_per_h=None,
) -> DustPhysicsTable:
    """Validate host arrays and construct a JAX-ready dust table.

    ``scattering_angle_cdf`` has shape ``(n_energy, n_angle)``.  Its first
    value must be zero, its last value one, and every row must be monotonic.
    When supplied, ``differential_cross_section_cm2_per_sr_per_h`` must have
    the same two-dimensional shape and must integrate to both the supplied
    total scattering cross-section and CDF.  It is required by the observer
    peel-off estimator but may be omitted by transport-only synthetic tests.
    The table builder deliberately rejects NaNs, negative cross-sections, and
    incomplete or inconsistently normalized tables rather than repairing
    them silently.
    """

    energy = np.asarray(energy_kev, dtype=np.float64)
    sigma_sca = np.asarray(scattering_cross_section_cm2_per_h, dtype=np.float64)
    sigma_abs = np.asarray(absorption_cross_section_cm2_per_h, dtype=np.float64)
    angle = np.asarray(scattering_angle_rad, dtype=np.float64)
    cdf = np.asarray(scattering_angle_cdf, dtype=np.float64)

    if energy.ndim != 1 or energy.size < 2:
        raise ValueError("energy_kev must be one-dimensional with at least two values")
    if angle.ndim != 1 or angle.size < 2:
        raise ValueError(
            "scattering_angle_rad must be one-dimensional with at least two values"
        )
    if sigma_sca.shape != energy.shape or sigma_abs.shape != energy.shape:
        raise ValueError("cross-section arrays must have shape (n_energy,)")
    if cdf.shape != (energy.size, angle.size):
        raise ValueError("scattering_angle_cdf must have shape (n_energy, n_angle)")
    if not np.all(np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("energy_kev must contain finite positive values")
    if not np.all(np.diff(energy) > 0.0):
        raise ValueError("energy_kev must be strictly increasing")
    if not np.all(np.isfinite(angle)) or np.any(np.diff(angle) <= 0.0):
        raise ValueError("scattering_angle_rad must be finite and strictly increasing")
    if angle[0] < 0.0 or angle[-1] > np.pi:
        raise ValueError("scattering angles must lie in [0, pi]")
    if not np.all(np.isfinite(sigma_sca)) or not np.all(np.isfinite(sigma_abs)):
        raise ValueError("cross-sections must be finite")
    if np.any(sigma_sca < 0.0) or np.any(sigma_abs < 0.0):
        raise ValueError("cross-sections cannot be negative")
    if not np.all(np.isfinite(cdf)):
        raise ValueError("scattering_angle_cdf must be finite")
    if np.any(np.diff(cdf, axis=1) < -1.0e-12):
        raise ValueError("every scattering-angle CDF row must be nondecreasing")
    if not np.allclose(cdf[:, 0], 0.0, rtol=0.0, atol=1.0e-10):
        raise ValueError("every scattering-angle CDF row must start at zero")
    if not np.allclose(cdf[:, -1], 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError("every scattering-angle CDF row must end at one")
    if np.any((cdf < -1.0e-12) | (cdf > 1.0 + 1.0e-12)):
        raise ValueError("scattering-angle CDF values must lie in [0, 1]")

    if differential_cross_section_cm2_per_sr_per_h is None:
        differential = np.empty((0, 0), dtype=np.float64)
    else:
        differential = np.asarray(
            differential_cross_section_cm2_per_sr_per_h,
            dtype=np.float64,
        )
        if differential.shape != cdf.shape:
            raise ValueError(
                "differential cross-section must have shape "
                "(n_energy, n_angle)"
            )
        if not np.all(np.isfinite(differential)) or np.any(differential < 0.0):
            raise ValueError(
                "differential cross-section must be finite and nonnegative"
            )
        integrated_sigma, integrated_cdf = (
            phase_cdf_from_differential_cross_section(angle, differential)
        )
        if not np.allclose(
            integrated_sigma, sigma_sca, rtol=5.0e-10, atol=0.0
        ):
            raise ValueError(
                "scattering cross-section does not match the differential table"
            )
        if not np.allclose(
            integrated_cdf, cdf, rtol=5.0e-10, atol=5.0e-12
        ):
            raise ValueError(
                "scattering-angle CDF does not match the differential table"
            )

    return DustPhysicsTable(
        energy_kev=jnp.asarray(energy),
        scattering_cross_section_cm2_per_h=jnp.asarray(sigma_sca),
        absorption_cross_section_cm2_per_h=jnp.asarray(sigma_abs),
        scattering_angle_rad=jnp.asarray(angle),
        scattering_angle_cdf=jnp.asarray(cdf),
        differential_cross_section_cm2_per_sr_per_h=jnp.asarray(differential),
    )


def phase_cdf_from_differential_cross_section(
    scattering_angle_rad,
    differential_cross_section_cm2_per_sr_per_h,
):
    """Construct ``sigma_sca`` and angular CDFs from intrinsic ``dσ/dΩ``.

    This is a NumPy preprocessing function, not part of the jitted hot path.
    The input has shape ``(n_energy, n_angle)`` and must be evaluated on a
    strictly increasing angular grid.  The integration uses the exact solid
    angle measure ``2*pi*sin(theta)*dtheta``.
    """

    theta = np.asarray(scattering_angle_rad, dtype=np.float64)
    differential = np.asarray(
        differential_cross_section_cm2_per_sr_per_h, dtype=np.float64
    )
    if theta.ndim != 1 or theta.size < 2 or np.any(np.diff(theta) <= 0.0):
        raise ValueError("scattering_angle_rad must be strictly increasing")
    if differential.ndim != 2 or differential.shape[1] != theta.size:
        raise ValueError(
            "differential cross-section must have shape (n_energy, n_angle)"
        )
    if not np.all(np.isfinite(differential)) or np.any(differential < 0.0):
        raise ValueError("differential cross-section must be finite and nonnegative")

    integrand = 2.0 * np.pi * np.sin(theta)[None, :] * differential
    increments = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * np.diff(theta)
    cumulative = np.concatenate(
        [np.zeros((differential.shape[0], 1)), np.cumsum(increments, axis=1)],
        axis=1,
    )
    sigma_sca = cumulative[:, -1]
    if np.any(sigma_sca <= 0.0):
        raise ValueError("every energy row must have positive integrated scattering")
    cdf = cumulative / sigma_sca[:, None]
    cdf[:, 0] = 0.0
    cdf[:, -1] = 1.0
    return sigma_sca, cdf


def remove_small_angle_dsh_geometry_factor(weighted_values, fractional_distance):
    """Remove an explicitly embedded ``(1-x)**-2`` DSH geometry factor.

    This only removes the multiplicative factor.  The caller must separately
    map an observer-angle axis to physical scattering angle using
    ``theta_sca = theta_obs / (1-x)`` in the small-angle approximation.
    """

    values = np.asarray(weighted_values)
    x = np.asarray(fractional_distance)
    if np.any(~np.isfinite(x)) or np.any((x < 0.0) | (x >= 1.0)):
        raise ValueError("fractional_distance must satisfy 0 <= x < 1")
    return values * (1.0 - x) ** 2
