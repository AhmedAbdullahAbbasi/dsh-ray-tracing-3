"""Regression probes for real 9F failure modes found by the code audit."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from dsh.sources.models import _sample_powerlaw_energy


class TestValidation9FSafeguards(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("astropy"), "requires Astropy")
    def test_cloud_degree_axes_are_converted_and_celestial_wcs_rejected(self):
        from astropy.io import fits

        from dsh.io.cloud_fits import load_cube

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "degrees.fits"
            image = fits.PrimaryHDU(np.ones((2, 2, 2)) * 1e20)
            for axis, scale, unit, name in (
                (1, 1 / 3600, "deg", "XOFFSET"),
                (2, 1 / 3600, "deg", "YOFFSET"),
                (3, 1000, "pc", "DISTANCE"),
            ):
                image.header[f"CRVAL{axis}"] = 0
                image.header[f"CRPIX{axis}"] = 1
                image.header[f"CDELT{axis}"] = scale
                image.header[f"CUNIT{axis}"] = unit
                image.header[f"CTYPE{axis}"] = name
            image.header["BUNIT"] = "cm-2"
            image.writeto(path)
            loaded = load_cube(path)
            np.testing.assert_allclose(np.diff(loaded["x_arcsec"]), [1.0])
            np.testing.assert_allclose(np.diff(loaded["y_arcsec"]), [1.0])
            np.testing.assert_allclose(np.diff(loaded["z_kpc"]), [1.0])
            image.header["CTYPE1"] = "GLON-SIN"
            image.writeto(path, overwrite=True)
            with self.assertRaisesRegex(ValueError, "celestial WCS"):
                load_cube(path)
            image.header["CTYPE1"] = "XOFFSET"
            image.header["PC1_2"] = 0.5
            image.writeto(path, overwrite=True)
            with self.assertRaisesRegex(ValueError, "CD/PC"):
                load_cube(path)

    def test_source_inverse_cdf_near_gamma_one(self):
        n = 10000
        key = random.PRNGKey(109)
        uniforms = np.asarray(random.uniform(key, (n,)), dtype=np.float64)
        sample = jax.jit(_sample_powerlaw_energy, static_argnames=("n_packets",))
        for gamma in (0.99999, 1.0, 1.000002, 1.00001, 1.7):
            alpha = 1 - np.float32(gamma).item()
            expected = (
                2 * np.exp(np.log1p(uniforms * np.expm1(alpha * np.log(5))) / alpha)
                if alpha != 0
                else 2 * np.exp(uniforms * np.log(5))
            )
            actual = np.asarray(
                sample(
                    key,
                    jnp.float32(2),
                    jnp.float32(10),
                    jnp.float32(gamma),
                    n_packets=n,
                )
            )
            with self.subTest(gamma=gamma):
                self.assertLess(np.max(np.abs(actual - expected) / expected), 2e-6)


if __name__ == "__main__":
    unittest.main()
