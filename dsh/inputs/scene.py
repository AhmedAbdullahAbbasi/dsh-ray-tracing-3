"""Source-centered native FITS or the established four-cloud example."""

from ..examples import build_synthetic_four_cloud_scene
from ..geometry.clouds import cloud_from_loaded_fits


def load_scene(config):
    if config.kind == "example_four_cloud":
        return build_synthetic_four_cloud_scene(config.source_distance_kpc)
    from ..io.cloud_fits import load_cube

    return cloud_from_loaded_fits(
        load_cube(config.cloud_file), source_distance_kpc=config.source_distance_kpc
    )
