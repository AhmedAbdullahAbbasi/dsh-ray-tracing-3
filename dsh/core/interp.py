"""Named production interpolation policies, preserving V1 evaluation order."""

from __future__ import annotations

import jax.numpy as jnp


def transport_energy_bracket(grid, value):
    upper = jnp.searchsorted(grid, value, side="right")
    upper = jnp.clip(upper, 1, grid.size - 1)
    lower = upper - 1
    log_grid = jnp.log(grid)
    fraction = (jnp.log(value) - log_grid[lower]) / (log_grid[upper] - log_grid[lower])
    return lower, upper, jnp.clip(fraction, 0.0, 1.0)


def transport_nonnegative(values, lower, upper, fraction):
    low = values[lower]
    high = values[upper]
    linear = low + fraction * (high - low)
    both_positive = (low > 0.0) & (high > 0.0)
    log_value = jnp.exp(
        jnp.log(jnp.maximum(low, jnp.finfo(values.dtype).tiny))
        + fraction
        * (
            jnp.log(jnp.maximum(high, jnp.finfo(values.dtype).tiny))
            - jnp.log(jnp.maximum(low, jnp.finfo(values.dtype).tiny))
        )
    )
    return jnp.where(both_positive, log_value, linear)


def observer_grid_bracket(grid, value):
    upper = jnp.searchsorted(grid, value, side="right")
    upper = jnp.clip(upper, 1, grid.size - 1)
    return upper - 1, upper


def observer_nonnegative_pair(low, high, fraction):
    """Log-interpolate positive values and linearly handle exact zeros."""

    linear = low + fraction * (high - low)
    dtype = jnp.result_type(low, high, fraction)
    tiny = jnp.finfo(dtype).tiny
    logarithmic = jnp.exp(
        jnp.log(jnp.maximum(low, tiny))
        + fraction
        * (jnp.log(jnp.maximum(high, tiny)) - jnp.log(jnp.maximum(low, tiny)))
    )
    return jnp.where((low > 0.0) & (high > 0.0), logarithmic, linear)


def observer_energy_fraction(energy_grid, energy):
    lower, upper = observer_grid_bracket(energy_grid, energy)
    log_grid = jnp.log(energy_grid)
    fraction = (jnp.log(energy) - log_grid[lower]) / (log_grid[upper] - log_grid[lower])
    return lower, upper, jnp.clip(fraction, 0.0, 1.0)


def observer_angle_fraction(angle_grid, angle):
    lower, upper = observer_grid_bracket(angle_grid, angle)
    low_angle = angle_grid[lower]
    high_angle = angle_grid[upper]
    linear_fraction = (angle - low_angle) / (high_angle - low_angle)
    tiny = jnp.finfo(angle_grid.dtype).tiny
    log_fraction = (
        jnp.log(jnp.maximum(angle, tiny)) - jnp.log(jnp.maximum(low_angle, tiny))
    ) / (jnp.log(jnp.maximum(high_angle, tiny)) - jnp.log(jnp.maximum(low_angle, tiny)))
    fraction = jnp.where(low_angle > 0.0, log_fraction, linear_fraction)
    return lower, upper, jnp.clip(fraction, 0.0, 1.0)
