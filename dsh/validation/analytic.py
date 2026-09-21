"""Independent analytic references for dust-scattering-halo validation.

These NumPy helpers are deliberately kept outside the JAX transport path.
They provide equations and summary statistics against which simulations are
tested; none is used to generate a simulated photon history.

The fractional dust distance ``x`` is measured from the observer:
``x = d_dust / D_source``. Angles supplied to the delay equations are the
observed source--halo separation, not the physical scattering angle.
"""

from __future__ import annotations

import numpy as np

from ..geometry.coordinates import ARCSEC_TO_RAD, PC_PER_KPC
from ..observer.scoring import PC_LIGHT_TRAVEL_TIME_S


def _validated_geometry(source_distance_kpc, fractional_distance, angle_rad=None):
    distance = np.asarray(source_distance_kpc, dtype=np.float64)
    fraction = np.asarray(fractional_distance, dtype=np.float64)
    if distance.ndim != 0 or not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("source_distance_kpc must be one finite positive scalar")
    if np.any(~np.isfinite(fraction)) or np.any((fraction <= 0.0) | (fraction >= 1.0)):
        raise ValueError("fractional_distance must satisfy 0 < x < 1")
    if angle_rad is None:
        return float(distance), fraction
    angle = np.asarray(angle_rad, dtype=np.float64)
    if np.any(~np.isfinite(angle)) or np.any(angle < 0.0) or np.any(angle >= np.pi):
        raise ValueError("observed angles must be finite and lie in [0, pi)")
    return float(distance), fraction, angle


def exact_single_scatter_excess_path_pc(
    source_distance_kpc,
    fractional_distance,
    observed_angle_rad,
):
    r"""Return the exact Euclidean path excess for one scattering event.

    For source distance ``D``, event radius ``r=xD``, and observed angle
    ``theta``, the broken path is

    ``sqrt(D**2 + r**2 - 2*D*r*cos(theta)) + r``.

    The implementation uses a rationalized form to retain DSH-scale delays
    without subtracting nearly equal source-distance terms.
    """

    distance_kpc, fraction, angle = _validated_geometry(
        source_distance_kpc,
        fractional_distance,
        observed_angle_rad,
    )
    distance_pc = distance_kpc * PC_PER_KPC
    event_radius_pc = fraction * distance_pc
    source_to_event_pc = np.sqrt(
        (distance_pc - event_radius_pc) ** 2
        + 4.0 * distance_pc * event_radius_pc * np.sin(0.5 * angle) ** 2
    )
    numerator = 4.0 * distance_pc * event_radius_pc * np.sin(0.5 * angle) ** 2
    denominator = source_to_event_pc + distance_pc - event_radius_pc
    return numerator / denominator


def exact_single_scatter_delay_s(
    source_distance_kpc,
    fractional_distance,
    observed_angle_rad,
):
    """Return the exact source--event--observer delay in seconds."""

    return (
        exact_single_scatter_excess_path_pc(
            source_distance_kpc,
            fractional_distance,
            observed_angle_rad,
        )
        * PC_LIGHT_TRAVEL_TIME_S
    )


def small_angle_single_scatter_delay_s(
    source_distance_kpc,
    fractional_distance,
    observed_angle_rad,
):
    r"""Return the standard thin-screen small-angle delay.

    ``Delta t = (D/c) * x * theta**2 / (2 * (1-x))``.
    """

    distance_kpc, fraction, angle = _validated_geometry(
        source_distance_kpc,
        fractional_distance,
        observed_angle_rad,
    )
    distance_pc = distance_kpc * PC_PER_KPC
    excess_pc = distance_pc * fraction * angle**2 / (2.0 * (1.0 - fraction))
    return excess_pc * PC_LIGHT_TRAVEL_TIME_S


def small_angle_ring_radius_arcsec(
    delay_s,
    source_distance_kpc,
    fractional_distance,
):
    """Invert the small-angle delay relation to obtain the ring radius."""

    distance_kpc, fraction = _validated_geometry(
        source_distance_kpc,
        fractional_distance,
    )
    delay = np.asarray(delay_s, dtype=np.float64)
    if np.any(~np.isfinite(delay)) or np.any(delay < 0.0):
        raise ValueError("delay_s must be finite and nonnegative")
    distance_pc = distance_kpc * PC_PER_KPC
    angle_rad = np.sqrt(
        2.0
        * (delay / PC_LIGHT_TRAVEL_TIME_S)
        * (1.0 - fraction)
        / (distance_pc * fraction)
    )
    return angle_rad / ARCSEC_TO_RAD


def finite_screen_ring_bounds_arcsec(
    delay_s,
    source_distance_kpc,
    fractional_distance_bounds,
):
    """Return inner/outer ring radii for a finite radial screen.

    At fixed positive delay, radius decreases monotonically with ``x``.
    The returned pair is therefore ``(radius_at_x_max, radius_at_x_min)``.
    """

    bounds = np.asarray(fractional_distance_bounds, dtype=np.float64)
    if bounds.shape != (2,) or not np.all(np.diff(bounds) > 0.0):
        raise ValueError("fractional_distance_bounds must be two increasing values")
    _validated_geometry(source_distance_kpc, bounds)
    inner = small_angle_ring_radius_arcsec(delay_s, source_distance_kpc, bounds[1])
    outer = small_angle_ring_radius_arcsec(delay_s, source_distance_kpc, bounds[0])
    return np.stack((inner, outer), axis=-1)


def phase_containment_angle_rad(angle_rad, phase_cdf, enclosed_fraction):
    """Interpolate the angle enclosing a requested phase probability."""

    angle = np.asarray(angle_rad, dtype=np.float64)
    cdf = np.asarray(phase_cdf, dtype=np.float64)
    fraction = np.asarray(enclosed_fraction, dtype=np.float64)
    if angle.ndim != 1 or angle.size < 2 or not np.all(np.diff(angle) > 0.0):
        raise ValueError("angle_rad must be one-dimensional and increasing")
    if cdf.shape[-1:] != angle.shape or np.any(~np.isfinite(cdf)):
        raise ValueError("phase_cdf final dimension must match angle_rad")
    if np.any(np.diff(cdf, axis=-1) < -1.0e-12):
        raise ValueError("phase_cdf must be nondecreasing")
    if np.any(~np.isfinite(fraction)) or np.any((fraction < 0.0) | (fraction > 1.0)):
        raise ValueError("enclosed_fraction must lie in [0, 1]")
    flat_cdf = cdf.reshape((-1, angle.size))
    flat_fraction = np.broadcast_to(fraction, cdf.shape[:-1]).reshape(-1)
    result = np.asarray(
        [
            np.interp(value, row, angle)
            for row, value in zip(flat_cdf, flat_fraction, strict=True)
        ]
    )
    return result.reshape(cdf.shape[:-1])


def log_log_power_law_slope(x, y):
    """Return the least-squares slope of ``log(y)`` against ``log(x)``."""

    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.ndim != 1 or y_values.shape != x_values.shape or x_values.size < 2:
        raise ValueError("x and y must be matching one-dimensional arrays")
    if (
        np.any(~np.isfinite(x_values))
        or np.any(~np.isfinite(y_values))
        or np.any(x_values <= 0.0)
        or np.any(y_values <= 0.0)
    ):
        raise ValueError("x and y must contain finite positive values")
    return float(np.polyfit(np.log(x_values), np.log(y_values), 1)[0])


def azimuthal_harmonic_amplitudes(azimuth_rad, weight=None, max_order=4):
    """Return normalized Fourier amplitudes used to quantify asymmetry."""

    azimuth = np.asarray(azimuth_rad, dtype=np.float64)
    if azimuth.ndim != 1 or azimuth.size == 0 or np.any(~np.isfinite(azimuth)):
        raise ValueError("azimuth_rad must be a nonempty finite vector")
    if not isinstance(max_order, int) or max_order <= 0:
        raise ValueError("max_order must be a positive integer")
    if weight is None:
        weights = np.ones_like(azimuth)
    else:
        weights = np.asarray(weight, dtype=np.float64)
        if weights.shape != azimuth.shape or np.any(~np.isfinite(weights)):
            raise ValueError("weight must be finite and match azimuth_rad")
        if np.any(weights < 0.0):
            raise ValueError("weight cannot be negative")
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("weight must have a positive sum")
    orders = np.arange(1, max_order + 1, dtype=np.float64)
    coefficients = np.sum(
        weights[:, None] * np.exp(1j * azimuth[:, None] * orders[None, :]),
        axis=0,
    )
    return np.abs(coefficients) / total


def causal_discrete_convolution(source_fluence, impulse_response):
    """Convolve nonnegative, equally spaced source and response sequences."""

    source = np.asarray(source_fluence, dtype=np.float64)
    response = np.asarray(impulse_response, dtype=np.float64)
    for values, name in ((source, "source_fluence"), (response, "impulse_response")):
        if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
            raise ValueError(f"{name} must be a nonempty finite vector")
        if np.any(values < 0.0):
            raise ValueError(f"{name} cannot be negative")
    return np.convolve(source, response, mode="full")
