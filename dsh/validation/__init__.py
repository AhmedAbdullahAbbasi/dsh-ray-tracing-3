"""Analytic references and statistical diagnostics for DSH validation."""

from .analytic import (
    azimuthal_harmonic_amplitudes,
    causal_discrete_convolution,
    exact_single_scatter_delay_s,
    exact_single_scatter_excess_path_pc,
    finite_screen_ring_bounds_arcsec,
    log_log_power_law_slope,
    phase_containment_angle_rad,
    small_angle_ring_radius_arcsec,
    small_angle_single_scatter_delay_s,
)

__all__ = (
    "azimuthal_harmonic_amplitudes",
    "causal_discrete_convolution",
    "exact_single_scatter_delay_s",
    "exact_single_scatter_excess_path_pc",
    "finite_screen_ring_bounds_arcsec",
    "log_log_power_law_slope",
    "phase_containment_angle_rad",
    "small_angle_ring_radius_arcsec",
    "small_angle_single_scatter_delay_s",
)
