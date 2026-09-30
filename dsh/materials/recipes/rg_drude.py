"""Independent NumPy implementation of the V1 Gaussian RG/Drude grain model.

This implements the *same approximation* and grain normalization as the
frozen NewDust table. It is not a Mie calculation, nor a dust model suitable
for arbitrarily low energies. All angles here are physical scattering angles,
not observed halo offsets. Generation is offline; the photon hot path still
consumes a precomputed table and CDF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Numerical conventions in the pinned xdust RGscattering/CmDrude implementation.
CLASSICAL_ELECTRON_RADIUS_CM = 2.8179403227e-13
PROTON_MASS_G = 1.67262192369e-24
HC_KEV_CM = 1.2398419843320026e-7
CM_PER_MICRON = 1.0e-4
ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)


@dataclass(frozen=True)
class RGDrudeMRN:
    minimum_radius_micron: float = 0.005
    maximum_radius_micron: float = 0.3
    size_power_law_index: float = 3.5
    radius_samples: int = 100
    material_density_g_cm3: float = 3.0
    dust_mass_per_h_g: float = 2.32475e-26


def gaussian_rg_drude_table(
    energy_kev,
    scattering_angle_rad,
    *,
    model: RGDrudeMRN | None = None,
):
    """Return (dSigma/dOmega, Sigma, CDF) for the configured grain population.

    Differential cross sections have units cm² sr⁻¹ H⁻¹. The integrated
    opacity is the solid-angle integral of that differential *Gaussian*
    rather than the separate xdust Qsca efficiency, preserving the V1
    interaction/phase normalization. Shape of differential/CDF is (E,angle).
    """

    model = RGDrudeMRN() if model is None else model
    energy = np.asarray(energy_kev, dtype=np.float64)
    theta = np.asarray(scattering_angle_rad, dtype=np.float64)
    if (
        energy.ndim != 1
        or energy.size < 2
        or not np.all(np.isfinite(energy))
        or energy[0] < 2.0
        or np.any(np.diff(energy) <= 0.0)
        or energy[-1] > 10.0
    ):
        raise ValueError("model extension requires increasing energies in [2,10] keV")
    if (
        theta.ndim != 1
        or theta.size < 2
        or not np.all(np.isfinite(theta))
        or theta[0] != 0.0
        or theta[-1] != np.pi
        or np.any(np.diff(theta) <= 0.0)
    ):
        raise ValueError("angles must increase from 0 through pi radians")
    if (
        model.minimum_radius_micron <= 0.0
        or model.maximum_radius_micron <= model.minimum_radius_micron
        or model.radius_samples < 2
        or model.material_density_g_cm3 <= 0.0
        or model.dust_mass_per_h_g <= 0.0
        or not np.isfinite(model.size_power_law_index)
    ):
        raise ValueError("invalid MRN/Drude grain parameters")

    radius_micron = np.linspace(
        model.minimum_radius_micron,
        model.maximum_radius_micron,
        model.radius_samples,
    )
    radius_cm = radius_micron * CM_PER_MICRON
    grain_mass = (4.0 * np.pi / 3.0) * radius_cm**3 * model.material_density_g_cm3
    size_law = radius_micron ** (-model.size_power_law_index)
    grains_per_h_per_micron = (
        model.dust_mass_per_h_g
        * size_law
        / np.trapezoid(size_law * grain_mass, radius_micron)
    )

    differential = np.empty((energy.size, theta.size), dtype=np.float64)
    for index, e in enumerate(energy):
        wavelength_cm = HC_KEV_CM / e
        size_parameter = 2.0 * np.pi * radius_cm / wavelength_cm
        refractive_index_minus_one = (
            model.material_density_g_cm3
            / (2.0 * PROTON_MASS_G)
            * CLASSICAL_ELECTRON_RADIUS_CM
            / (2.0 * np.pi)
            * wavelength_cm**2
        )
        # Pinned xdust rgscat.py uses _dsig * _thdep / (pi*a**2):
        # 2*a²*z⁴*|m-1|² * (2/9)*exp[-alpha²/(2*s²)].
        amplitude = (
            (4.0 / 9.0)
            * radius_cm**2
            * size_parameter**4
            * refractive_index_minus_one**2
        )
        width_rad = 1.04 * ARCMIN_TO_RAD / (e * radius_micron)
        phase = np.exp(-0.5 * (theta[:, None] / width_rad[None, :]) ** 2)
        differential[index] = np.trapezoid(
            phase * (amplitude * grains_per_h_per_micron)[None, :],
            radius_micron,
            axis=1,
        )

    integrand = 2.0 * np.pi * np.sin(theta)[None, :] * differential
    increments = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * np.diff(theta)
    cumulative = np.concatenate(
        [np.zeros((energy.size, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    sigma = cumulative[:, -1]
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("scattering integral is not finite and positive")
    cdf = cumulative / sigma[:, None]
    cdf[:, 0] = 0.0
    cdf[:, -1] = 1.0
    return differential, sigma, cdf


def default_angle_grid(positive_samples: int = 8192, minimum_arcsec: float = 1e-3):
    """V1-compatible 0-to-pi grid with logarithmic small-angle sampling."""

    if positive_samples < 2 or not 0.0 < minimum_arcsec < 180.0 * 3600.0:
        raise ValueError("invalid angular-grid settings")
    return np.r_[
        0.0,
        np.geomspace(
            minimum_arcsec * np.pi / (180.0 * 3600.0), np.pi, positive_samples
        ),
    ]
