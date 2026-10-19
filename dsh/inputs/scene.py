"""Source-centered native FITS or the established four-cloud example."""

import numpy as np

from ..examples import build_synthetic_four_cloud_scene
from ..geometry.clouds import (
    KPC_TO_CM,
    build_angular_distance_cloud,
    centers_to_edges,
    cloud_from_loaded_fits,
)


def load_scene(config):
    if config.kind == "example_four_cloud":
        return build_synthetic_four_cloud_scene(config.source_distance_kpc)
    if config.kind == "uniform_disc":
        x, y, z = [
            np.arange(*values)
            for values in (
                config.grid_x_arcsec,
                config.grid_y_arcsec,
                config.grid_z_kpc,
            )
        ]
        xx, yy = np.meshgrid(x, y, indexing="xy")
        radius = np.sqrt(
            (xx - config.disc_center_x_arcsec) ** 2
            + (yy - config.disc_center_y_arcsec) ** 2
        )
        inside = (radius[None] <= config.disc_radius_arcsec) & (
            np.abs(z[:, None, None] - config.disc_center_z_kpc)
            <= 0.5 * config.disc_thickness_kpc
        )
        widths = np.diff(centers_to_edges(z)) * KPC_TO_CM
        columns = np.where(inside, config.disc_n_h_cm3, 0.0) * widths[:, None, None]
        return build_angular_distance_cloud(
            columns, x, y, z, config.source_distance_kpc
        )
    from ..io.cloud_fits import load_cube

    return cloud_from_loaded_fits(
        load_cube(config.cloud_file), source_distance_kpc=config.source_distance_kpc
    )
