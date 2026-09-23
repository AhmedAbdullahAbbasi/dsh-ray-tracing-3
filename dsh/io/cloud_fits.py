"""Read a physical angular--distance hydrogen-column FITS cube."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits


def _axis_coordinates(header, axis_number: int, size: int) -> np.ndarray:
    """Return linear-WCS coordinates for one FITS axis."""

    pixel = np.arange(1, size + 1, dtype=np.float64)
    allowed = {
        1: {"XOFFSET", "OFFSET", "LINEAR", "X"},
        2: {"YOFFSET", "OFFSET", "LINEAR", "Y"},
        3: {"DISTANCE", "LINEAR", "Z", "DIST"},
    }[axis_number]
    if header.get(f"CTYPE{axis_number}", "").upper() not in allowed:
        raise ValueError(
            "cloud FITS requires native local-offset/distance axes; celestial WCS needs an adapter"
        )
    try:
        scale = u.Unit(header[f"CUNIT{axis_number}"]).to(
            u.arcsec if axis_number < 3 else u.kpc
        )
    except (KeyError, ValueError) as error:
        raise ValueError(
            "cloud FITS requires explicit angular/distance CUNIT values"
        ) from error
    return scale * (
        header[f"CRVAL{axis_number}"]
        + (pixel - header[f"CRPIX{axis_number}"]) * header[f"CDELT{axis_number}"]
    )


def load_cube(path: str | Path) -> dict[str, object]:
    """Load the primary ``delta_NH`` cube and its physical coordinate axes.

    The primary array must be three-dimensional with NumPy order
    ``(distance, sky_y, sky_x)``.  Voxel values are returned unchanged as
    ``delta_nh_cm2``; validation and conversion to radial-average number
    density belong to :func:`dsh.geometry.clouds.cloud_from_loaded_fits`.
    """

    with fits.open(path) as hdul:
        primary = hdul[0]
        delta_nh_cm2 = np.asarray(primary.data, dtype=np.float64)
        header = primary.header.copy()

    if any(
        key.startswith(("CD1_", "CD2_", "CD3_", "PC1_", "PC2_", "PC3_", "CROTA"))
        for key in header
    ):
        raise ValueError(
            "cloud FITS CD/PC/rotation WCS is unsupported; use native linear axes"
        )

    if delta_nh_cm2.ndim != 3:
        raise ValueError("primary FITS array must have shape (z, y, x)")

    n_z, n_y, n_x = delta_nh_cm2.shape
    return {
        "delta_nh_cm2": delta_nh_cm2,
        "x_arcsec": _axis_coordinates(header, 1, n_x),
        "y_arcsec": _axis_coordinates(header, 2, n_y),
        "z_kpc": _axis_coordinates(header, 3, n_z),
        "unit": header.get("BUNIT", "").strip(),
        "header": header,
    }
