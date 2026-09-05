"""Inspect the input FITS density cube: extract its contents and print/plot them.

This is a read-only companion to `voxels.py`, written for
`realistic_nh_cube_500asec_10kpc_dz0p05.fits` -- a synthetic
turbulent-ISM sightline cube supplied as an example of a *real* (as
opposed to hand-built) density field. Everything here just reads the
file and reports what's in it; nothing here builds a
`simulate_photons`-ready density grid or otherwise touches
`transport.py` -- that conversion, if wanted, is a separate step.

The file's structure, for reference:

- HDU 0 ("TOTAL_NH", primary): the 3D cube itself, shape
  (n_dist, n_y, n_x) -- the hydrogen column density contributed by
  each voxel (NHI + 2*NH2, in cm^-2). This is the "cloud voxel density
  map" this module is mainly for.
- HDU 1-3 ("NH_MAP", "HI_NH_MAP", "MOL_NH_MAP"): 2D maps -- the
  total/atomic/molecular column density integrated along the full
  line of sight.
- HDU 4-7 ("MC1_NH" .. "MC4_NH"): 2D column-density maps for each
  individual molecular cloud complex.
- HDU 8 ("CLOUDS"): a binary table, one row per molecular cloud
  complex, giving its position, size, and peak column density.

The cube's 3 axes are angular offset in x, angular offset in y (both
arcsec), and line-of-sight distance (kpc); the primary header's
CRPIX/CRVAL/CDELT keywords give the physical coordinate of every voxel
along each axis.
"""

import numpy as np
from astropy.io import fits


def _axis_coords(header, axis_num, n):
    """Physical coordinate of every pixel along one FITS axis (1-indexed, as in the header).

    Plain linear WCS: coord = CRVAL + (pixel - CRPIX) * CDELT, with
    pixel running 1..n in the FITS convention.
    """
    crpix = header[f"CRPIX{axis_num}"]
    crval = header[f"CRVAL{axis_num}"]
    cdelt = header[f"CDELT{axis_num}"]
    pixel = np.arange(1, n + 1)
    return crval + (pixel - crpix) * cdelt


def _decode(value):
    """Turn a FITS byte-string table entry into a plain, stripped Python str; pass everything else through."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode().strip()
    return value


def load_cube(fits_path):
    """Open the FITS file and pull out the main density cube, its physical axis coordinates, and its unit.

    Returns a plain dict:
      - "density": the (n_z, n_y, n_x) TOTAL_NH array, as float64 numpy (z = distance, y/x = angular offsets)
      - "x_arcsec", "y_arcsec", "z_kpc": 1D physical coordinate arrays for each axis
      - "unit": the BUNIT string (e.g. "cm-2")
      - "header": the primary HDU header, for anything not already pulled out above
    """
    with fits.open(fits_path) as hdul:
        primary = hdul[0]
        density = np.asarray(primary.data, dtype=np.float64)
        header = primary.header.copy()

    n_z, n_y, n_x = density.shape  # FITS axis order (3, 2, 1) -> numpy axis order (z, y, x)
    return {
        "density": density,
        "x_arcsec": _axis_coords(header, 1, n_x),
        "y_arcsec": _axis_coords(header, 2, n_y),
        "z_kpc": _axis_coords(header, 3, n_z),
        "unit": header.get("BUNIT", "").strip(),
        "header": header,
    }


def print_fits_info(fits_path):
    """Print a plain-language summary of every HDU in the FITS file.

    Just `astropy.io.fits`'s own `.info()`, plus for each HDU: its
    shape/dtype/unit and data min/max/mean (image HDUs), or its rows
    and column values (the "CLOUDS" binary table) -- enough to see
    what's actually in the file without opening it elsewhere.
    """
    with fits.open(fits_path) as hdul:
        hdul.info()
        print()
        for hdu in hdul:
            name = hdu.name or "PRIMARY"
            print(f"--- {name} ---")
            if hdu.data is None:
                print("  (no data)")
                print()
                continue

            if hdu.is_image:
                data = np.asarray(hdu.data)
                unit = hdu.header.get("BUNIT", "").strip()
                print(f"  shape: {data.shape}   dtype: {data.dtype}   unit: {unit or '(none)'}")
                print(f"  min={np.nanmin(data):.4e}  max={np.nanmax(data):.4e}  "
                      f"mean={np.nanmean(data):.4e}")
                for key in ("BTYPE", "SIMTYPE", "NMCLOUD", "ZMINKPC", "ZMAXKPC", "DZKPC"):
                    if key in hdu.header:
                        print(f"  {key}: {hdu.header[key]}")
            else:
                table = hdu.data
                print(f"  {len(table)} rows, columns: {list(table.columns.names)}")
                for row in table:
                    fields = ", ".join(f"{c}={_decode(row[c])}" for c in table.columns.names)
                    print(f"    {fields}")
            print()


def print_cube_summary(cube):
    """Print basic stats about the loaded density cube (`load_cube`'s return value).

    Grid shape and physical extent along each axis, the density
    range/unit, and how many voxels actually carry nonzero density --
    a quick sanity check before looking at the cube any further.
    """
    density = cube["density"]
    n_z, n_y, n_x = density.shape
    nonzero = density > 0

    print(f"grid shape (z, y, x): {density.shape}")
    print(f"x extent: [{cube['x_arcsec'][0]:.2f}, {cube['x_arcsec'][-1]:.2f}] arcsec  ({n_x} voxels)")
    print(f"y extent: [{cube['y_arcsec'][0]:.2f}, {cube['y_arcsec'][-1]:.2f}] arcsec  ({n_y} voxels)")
    print(f"z extent: [{cube['z_kpc'][0]:.3f}, {cube['z_kpc'][-1]:.3f}] kpc  ({n_z} voxels)")
    print(f"density unit: {cube['unit'] or '(none)'}")
    print(f"density min/max/mean: {density.min():.4e} / {density.max():.4e} / {density.mean():.4e}")
    print(f"nonzero voxels: {nonzero.sum():,} / {density.size:,} ({100 * nonzero.mean():.2f}%)")


def plot_cloud_3d(cube, ax=None, density_threshold_frac=0.05, max_points=20_000, seed=0, cmap="inferno"):
    """3D scatter plot of the cloud: one point per occupied voxel, colored (and sized) by density.

    matplotlib has no true volume renderer, so -- same honest
    substitute used for the simulation's own density fields in
    `plotting.plot_voxel_grid` -- this draws one point per voxel above
    `density_threshold_frac` of the cube's peak density, subsampled to
    `max_points` if there are more than that many. Since reading off
    the density itself is the point here, color (log-scaled, with a
    colorbar) carries the density value directly, rather than only
    point size.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    density = cube["density"]
    d_max = density.max()
    if d_max <= 0:
        raise ValueError("cube is all zero -- nothing to plot")

    mask = density > density_threshold_frac * d_max
    Z, Y, X = np.meshgrid(cube["z_kpc"], cube["y_arcsec"], cube["x_arcsec"], indexing="ij")
    xs, ys, zs, ds = X[mask], Y[mask], Z[mask], density[mask]

    if xs.size > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(xs.size, size=max_points, replace=False)
        xs, ys, zs, ds = xs[keep], ys[keep], zs[keep], ds[keep]

    if ax is None:
        ax = plt.figure(figsize=(9, 7)).add_subplot(projection="3d")

    sizes = 4.0 + 20.0 * (ds / d_max)
    sc = ax.scatter(xs, ys, zs, c=ds, s=sizes, cmap=cmap,
                     norm=LogNorm(vmin=ds.min(), vmax=ds.max()), alpha=0.5, linewidths=0)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(f"density ({cube['unit'] or 'unitless'})")

    ax.set_xlabel("x offset [arcsec]")
    ax.set_ylabel("y offset [arcsec]")
    ax.set_zlabel("distance [kpc]")
    ax.set_title(f"Cloud voxel density (voxels above {density_threshold_frac:.0%} of peak)")
    return ax


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    path = sys.argv[1] if len(sys.argv) > 1 else "realistic_nh_cube_500asec_10kpc_dz0p05.fits"

    print_fits_info(path)
    cube = load_cube(path)
    print_cube_summary(cube)
    plot_cloud_3d(cube)
    plt.show()
