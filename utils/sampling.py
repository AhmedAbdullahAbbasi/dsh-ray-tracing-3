"""The random draws this simulation needs: how far a photon goes, and which way it scatters.

Photons now travel through a real volumetric medium -- a grid of
voxels, each with its own density (see `voxels.py`) -- instead of
empty vacuum with one hard object in it. That brings back a random
free path to sample, exactly like the very first version of this
notebook's slab model, just now integrated through a spatially varying
density along a 3D ray instead of a uniform 1D medium (see
`transport.photon_step` for how). All four functions below use inverse
transform sampling: draw xi ~ Uniform(0, 1), then invert the CDF of the
quantity you actually want.
"""

import math

import jax.numpy as jnp
from jax import random


def sample_free_path(key):
    """Draw a random *optical depth* to travel before the next scattering event.

    The probability of travelling an optical depth tau without
    scattering is exp(-tau) (the Beer-Lambert law), so tau is
    exponentially distributed with mean 1: tau = -log(xi). This is a
    target in *optical depth*, not physical distance -- `photon_step`
    converts it into an actual 3D stopping point by marching along the
    ray and accumulating optical depth = density x distance as it
    passes through however many voxels it takes to reach that target.
    """
    xi = random.uniform(key)
    return -jnp.log(xi)


def sample_isotropic_mu(key):
    """Draw a direction cosine for an isotropic scattering event.

    "Isotropic" scattering means the photon forgets its incoming
    direction and re-emits equally likely into any direction on the
    sphere. Relative to *any* fixed axis, the cosine of the angle to
    that axis then comes out uniformly distributed on [-1, 1]:

        mu = 2 * xi - 1

    Scattering off a voxel of the medium isn't a surface bounce off a
    solid object -- there's no normal to stay on the outside of -- so
    `photon_step` uses the full range here, unlike a hard-object model
    that would restrict it to an outward hemisphere.
    """
    xi = random.uniform(key)
    return 2.0 * xi - 1.0


def sample_azimuth(key):
    """Draw a uniformly random angle around a reference axis, in [0, 2*pi).

    Isotropic scattering has no preferred direction *around* the axis
    either -- combined with `sample_isotropic_mu` for the angle *to*
    the axis, this is the second of the two numbers needed to fully
    specify a random direction in 3D.
    """
    xi = random.uniform(key)
    return 2.0 * jnp.pi * xi


def _real_cbrt(x):
    """The real cube root of `x`, including negative `x` -- see `sample_rayleigh_mu` for why not `jnp.cbrt`."""
    return jnp.sign(x) * jnp.abs(x) ** (1.0 / 3.0)


def sample_rayleigh_mu(key):
    """Draw a direction cosine from the Rayleigh scattering phase function.

    Rayleigh scattering applies when the scatterer (a dust grain, a
    free electron) is much smaller than the photon's wavelength. Its
    angular distribution is the classic dipole pattern,

        p(mu) = (3/8) * (1 + mu^2),    mu in [-1, 1],

    unlike isotropic scattering's flat p(mu) = 1/2 -- photons are
    somewhat more likely to keep going forward/backward (mu near +-1)
    than to scatter sideways (mu near 0). Inverting its CDF, xi = F(mu),
    means solving a depressed cubic mu^3 + 3*mu + (4 - 8*xi) = 0; since
    its derivative 3*mu^2 + 3 is always positive, there's exactly one
    real root, given by Cardano's formula:

        A = 2 - 4*xi
        mu = cbrt(sqrt(A^2 + 1) - A) + cbrt(-sqrt(A^2 + 1) - A)

    (checked against the analytic CDF -- see the notebook's scattering
    phase function section). Uses a hand-rolled real cube root
    (`sign(x) * |x|^(1/3)`) rather than `jnp.cbrt`: the latter doesn't
    lower on the experimental Metal backend (`mhlo.cbrt` fails to
    legalize and takes the whole run down with it), while a plain
    `**(1/3)` on the absolute value does.
    """
    xi = random.uniform(key)
    a = 2.0 - 4.0 * xi
    d = jnp.sqrt(a * a + 1.0)
    return _real_cbrt(d - a) + _real_cbrt(-d - a)


def sample_henyey_greenstein_mu(key, g):
    """Draw a direction cosine from the Henyey-Greenstein phase function, asymmetry `g`.

    Exact Mie scattering (a grain comparable to or larger than the
    wavelength) needs a size- and composition-dependent sum of Bessel
    functions with no closed form to sample from directly. The
    Henyey-Greenstein function is the standard practical stand-in used
    throughout radiative transfer (dust, clouds, ocean optics) when a
    full Mie phase-function table isn't available -- a single-parameter
    curve that reproduces Mie's characteristic *forward-peaked* shape:

        p(mu) = (1 - g^2) / (2 * (1 + g^2 - 2*g*mu)^(3/2)),  mu in [-1, 1]

    `g` is the mean cosine of the scattering angle: `g = 0` is
    isotropic (falls back to `sample_isotropic_mu`'s formula exactly),
    `g -> 1` is almost pure forward scattering (a large grain), `g < 0`
    biases backward. Inverting its CDF gives a closed form (checked
    against the analytic CDF -- see the notebook's scattering phase
    function section):

        mu = (1 + g^2 - ((1 - g^2) / (1 - g + 2*g*xi))^2) / (2*g)
    """
    xi = random.uniform(key)
    g_safe = jnp.where(jnp.abs(g) < 1e-3, 1.0, g)  # dodge the g=0 division; overwritten below anyway
    mu_hg = (1.0 + g_safe**2 - ((1.0 - g_safe**2) / (1.0 - g_safe + 2.0 * g_safe * xi)) ** 2) / (2.0 * g_safe)
    return jnp.where(jnp.abs(g) < 1e-3, 2.0 * xi - 1.0, mu_hg)


def mie_asymmetry_from_size_parameter(x, g_max=0.85):
    """Map a dimensionless grain size parameter to a Henyey-Greenstein asymmetry `g`.

    `x` stands in for `2*pi*grain_radius / wavelength`: small `x` means
    a grain much smaller than the photon's wavelength (the Rayleigh
    regime), large `x` means a grain comparable to or bigger than it
    (the Mie regime, increasingly forward-peaked). This notebook has no
    real grain-size distribution or refractive index to compute an
    exact `g(x)` from, so this is a deliberately simple, monotonically
    saturating stand-in -- `g` rises from 0 at `x = 0` towards `g_max`
    as `x` grows -- picked only to give believable *qualitative*
    behavior (near-isotropic for small grains/long wavelengths,
    strongly forward-scattering for large grains/short wavelengths),
    not a fitted physical result:

        g(x) = g_max * x / (x + 1)
    """
    return g_max * x / (x + 1.0)


_HC_KEV_ANGSTROM = 12.398419843320026  # h*c: wavelength[angstrom] = this / energy[keV]


def size_parameter_from_energy_kev(energy_kev, grain_radius_um):
    """Convert a real photon energy (keV) and an assumed grain radius (microns) into `x`.

    Nothing in this simulation ties its internal `energy` field to any
    physical unit on its own -- see `mie_asymmetry_from_size_parameter`'s
    `x` -- so this is a plain, non-JAX preprocessing helper *you* call
    yourself, once, to decide what that tie should be: pick a photon
    energy in keV and a grain radius in microns, and it works out the
    corresponding size parameter `x = 2*pi*grain_radius / wavelength`
    via the standard X-ray relation `wavelength[angstrom] = 12.398 /
    energy[keV]`. Use its output as `energy_min`/`energy_max` in
    `simulate_photons` (still just called "energy" there for
    historical reasons -- it's really "size parameter" once you're
    using the "mie"/"rayleigh_mie" scattering models).

    For calibration: typical ISM dust grains are roughly 0.005-0.25 um
    in radius; visible light is ~2-3 eV (wavelength ~4000-7000
    angstrom); soft X-rays are ~0.1-10 keV (wavelength ~1-100
    angstrom). A 0.1 um grain is deep in the Mie regime (x >> 1) for
    visible light, but near or below the Rayleigh/Mie transition for
    hard X-rays -- i.e. the same dust can look very different to
    photons of different energies, which is exactly the effect
    `scattering_model="rayleigh_mie"` is there to capture.
    """
    wavelength_angstrom = _HC_KEV_ANGSTROM / energy_kev
    grain_radius_angstrom = grain_radius_um * 1e4  # 1 um = 1e4 angstrom
    return 2.0 * math.pi * grain_radius_angstrom / wavelength_angstrom


def sample_mu_in_cone(key, cos_half_angle):
    """Draw a direction cosine uniformly (by solid angle) within a cone around some axis.

    Same idea as `sample_isotropic_mu` -- uniform solid angle means
    uniform in `mu` -- just restricted to `mu` in `[cos_half_angle, 1]`
    instead of the full `[-1, 1]`. Used to give the injected photon
    beam some angular spread instead of every photon starting out
    perfectly parallel. Two useful extremes: `cos_half_angle=1.0`
    (a zero-width cone) always returns `mu=1`, i.e. no spread at all;
    `cos_half_angle=-1.0` recovers full-sphere isotropic emission,
    e.g. for a point source instead of a beam.
    """
    xi = random.uniform(key)
    return cos_half_angle + (1.0 - cos_half_angle) * xi
