"""Offline Mie cross sections and a bounded forward-angle ADT approximation.

Indices use the passive-material convention m=n+i*kappa, kappa>=0. The Mie
backend uses its conjugate convention. Differential results are cm2/sr per
grain. Graphite orientation averaging is done by the population producer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .optical_constants import HC_KEV_UM

ADT_MINIMUM_SIZE_PARAMETER = 300.0
ADT_MAXIMUM_INDEX_CONTRAST = 0.01
ADT_MAXIMUM_PHASE_SHIFT = 8.0
ADT_MAXIMUM_ANGLE_RAD = 0.1


@dataclass(frozen=True)
class SphereCrossSections:
    differential_cm2_per_sr: np.ndarray
    scattering_cm2: float
    absorption_cm2: float
    extinction_cm2: float
    phase_method: str


def _inputs(radius_um, energy_kev, index, angle_rad):
    radius, energy, index = float(radius_um), float(energy_kev), complex(index)
    if not np.isfinite(radius) or radius <= 0 or not np.isfinite(energy) or energy <= 0:
        raise ValueError("radius and energy must be finite and positive")
    if not np.isfinite(index) or index.real <= 0 or index.imag < 0:
        raise ValueError("index must describe a passive material, n>0 and kappa>=0")
    angle = np.atleast_1d(np.asarray(angle_rad, dtype=np.float64))
    if (
        angle.ndim != 1
        or not angle.size
        or np.any(~np.isfinite(angle))
        or np.any((angle < 0) | (angle > np.pi))
    ):
        raise ValueError("angles must be a nonempty finite 1D array in [0, pi]")
    wave_um = HC_KEV_UM / energy
    x = 2 * np.pi * radius / wave_um
    wave_number = 2 * np.pi / (wave_um * 1e-4)
    if not np.isfinite(x + wave_number) or x > 2e5:
        raise ValueError("sphere solver supports finite size parameters up to 200000")
    return radius, index, angle, x, wave_number


def mie_sphere(radius_um, energy_kev, index, angle_rad):
    """Exact homogeneous-sphere Mie solution using pinned miepython 3.3.0.

    Set MIEPYTHON_USE_JIT=1 before importing miepython for its optional Numba
    backend. That affects table-generation time, not physical normalization.
    """
    try:
        import miepython
    except ImportError as exc:
        raise RuntimeError(
            'Install offline producer dependencies with pip install -e ".[materials]"'
        ) from exc
    radius, index, angle, x, wave_number = _inputs(
        radius_um, energy_kev, index, angle_rad
    )
    qext, qsca, _, _ = miepython.efficiencies_mx(np.conj(index), x)
    tolerance = 64 * np.finfo(float).eps * max(abs(qext), abs(qsca), 1)
    if not np.isfinite(qext + qsca) or qsca < -tolerance or qext - qsca < -tolerance:
        raise ValueError("Mie backend returned invalid passive-material efficiencies")
    area = np.pi * (radius * 1e-4) ** 2
    s1, s2 = miepython.S1_S2(np.conj(index), x, np.cos(angle), norm="wiscombe")
    differential = (np.abs(s1) ** 2 + np.abs(s2) ** 2) / (2 * wave_number**2)
    if np.any(~np.isfinite(differential)):
        raise ValueError("Mie differential law is nonfinite")
    return SphereCrossSections(
        differential,
        max(0.0, float(qsca)) * area,
        max(0.0, float(qext - qsca)) * area,
        max(0.0, float(qext)) * area,
        "mie",
    )


@lru_cache(maxsize=4)
def _unit_quadrature(order):
    node, weight = np.polynomial.legendre.leggauss(order)
    return (node + 1) / 2, weight / 2


def _adt_domain(x, index):
    rho = 2 * x * (index - 1)
    return (
        x >= ADT_MINIMUM_SIZE_PARAMETER
        and abs(index - 1) <= ADT_MAXIMUM_INDEX_CONTRAST
        and abs(rho) <= ADT_MAXIMUM_PHASE_SHIFT
    )


def adt_sphere_efficiencies(x, index, *, quadrature_order=96):
    """Positive ADT Q_sca, Q_abs, Q_ext from shadow-function integrals.

    Hoffman & Draine (2016), equations (6)--(8), evaluated without subtracting
    nearly equal Q_ext and Q_abs or using cancellation-prone small-rho formulae.
    The deliberately conservative supported domain is shared with the hybrid.
    """
    x, index = float(x), complex(index)
    if (
        not np.isfinite(x)
        or not np.isfinite(index)
        or index.imag < 0
        or not _adt_domain(x, index)
    ):
        raise ValueError("ADT requires x>=300, |m-1|<=0.01 and |2*x*(m-1)|<=8")
    t, weight = _unit_quadrature(quadrature_order)
    rho = 2 * x * (index - 1)
    shadow = -np.expm1(1j * rho * t)
    qsca = float(2 * np.sum(weight * t * np.abs(shadow) ** 2))
    qabs = float(2 * np.sum(weight * t * -np.expm1(-2 * rho.imag * t)))
    return qsca, qabs, qsca + qabs


def _shadow_transform(rho, q):
    """Hankel transform of the spherical shadow, with stable oscillatory tails.

    For q<=40 use smooth Gauss quadrature in chord length t. For larger q,
    integrate the exponential power series analytically with
    integral_0^1 u*(1-u^2)^(n/2)*J0(q*u)du
      = 2^(n/2)*Gamma(n/2+1)*J_(n/2+1)(q)/q^(n/2+1).
    Fixed 60 terms are conservative for the enforced |rho|<=8 domain.
    """
    from scipy.special import gammaln, j0, jv

    q = np.asarray(q, dtype=np.float64)
    answer = np.zeros(q.shape, dtype=complex)
    smooth = q <= 40
    if np.any(smooth):
        t, weight = _unit_quadrature(96)
        integrand = weight * t * -np.expm1(1j * rho * t)
        answer[smooth] = j0(q[smooth, None] * np.sqrt(1 - t[None, :] ** 2)) @ integrand
    if np.any(~smooth):
        value = q[~smooth]
        coefficient = 1 + 0j
        total = np.zeros(value.shape, dtype=complex)
        for n in range(1, 61):
            coefficient *= 1j * rho / n
            power = n / 2
            moment = np.exp(
                power * np.log(2) + gammaln(power + 1) - (power + 1) * np.log(value)
            ) * jv(power + 1, value)
            total -= coefficient * moment
        answer[~smooth] = total
    return answer


def adt_forward_differential(radius_um, energy_kev, index, angle_rad):
    """Forward ADT dC/dOmega, valid only for theta<=0.1 radians.

    Uses x*sin(theta) as in corrected Hoffman--Draine (2016) equation (15).
    Extending this planar approximation to pi would create an unphysical
    backward peak. The hybrid uses exact Mie outside this forward interval.
    """
    radius, index, angle, x, wave_number = _inputs(
        radius_um, energy_kev, index, angle_rad
    )
    if not _adt_domain(x, index) or np.any(angle > ADT_MAXIMUM_ANGLE_RAD):
        raise ValueError(
            "forward ADT called outside the supported size/index/angle domain"
        )
    transform = _shadow_transform(2 * x * (index - 1), x * np.sin(angle))
    return wave_number**2 * (radius * 1e-4) ** 4 * np.abs(transform) ** 2


def hybrid_sphere(radius_um, energy_kev, index, angle_rad):
    """Exact Mie totals; forward ADT where valid and exact Mie at other angles.

    This approximation is opt-in and retains a complete all-sky phase law.
    The producer records its raw angular closure against the Mie total.
    """
    _, index, angle, x, _ = _inputs(radius_um, energy_kev, index, angle_rad)
    if not _adt_domain(x, index):
        return mie_sphere(radius_um, energy_kev, index, angle)
    forward = angle <= ADT_MAXIMUM_ANGLE_RAD
    if not np.any(forward):
        return mie_sphere(radius_um, energy_kev, index, angle)
    # At least one angle is needed by the backend. The forward-only case uses
    # zero as a cheap placeholder solely to obtain exact integrated efficiencies.
    tail = mie_sphere(
        radius_um, energy_kev, index, angle[~forward] if np.any(~forward) else [0.0]
    )
    differential = np.empty(angle.shape)
    differential[forward] = adt_forward_differential(
        radius_um, energy_kev, index, angle[forward]
    )
    if np.any(~forward):
        differential[~forward] = tail.differential_cm2_per_sr
    return SphereCrossSections(
        differential,
        tail.scattering_cm2,
        tail.absorption_cm2,
        tail.extinction_cm2,
        "mie_totals_forward_adt_mie_tail",
    )
