"""Collect a reusable physical halo response for a fixed cloud and material."""

import argparse
from pathlib import Path

from dsh.examples import build_synthetic_four_cloud_scene
from dsh.geometry.clouds import cloud_from_loaded_fits
from dsh.observer.response import write_spectral_response
from dsh.observer.response_transport import collect_spectral_response
from dsh.physics.wd01_ism import load_wd01_ism
from dsh.sources.histories import load_spectral_history
from dsh.sources.radial_launch import build_radial_cloud_launch_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-history", type=Path, required=True)
    parser.add_argument("--material-directory", type=Path, required=True)
    parser.add_argument("--cloud-fits", type=Path)
    parser.add_argument("--source-distance-kpc", type=float, default=10)
    parser.add_argument("--packets", type=int, default=262144)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--max-interactions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    definition, _, _, _, physics = load_wd01_ism(args.material_directory)
    proposal, _ = load_spectral_history(args.proposal_history)
    if args.cloud_fits is None:
        cloud = build_synthetic_four_cloud_scene(args.source_distance_kpc)
    else:
        from dsh.io.cloud_fits import load_cube

        cloud = cloud_from_loaded_fits(
            load_cube(args.cloud_fits), source_distance_kpc=args.source_distance_kpc
        )
    response = collect_spectral_response(
        proposal,
        cloud,
        physics,
        build_radial_cloud_launch_geometry(cloud, physics),
        n_packets=args.packets,
        chunk_size=args.chunk_size,
        max_interactions=args.max_interactions,
        seed=args.seed,
        progress=lambda n, total: print(
            f"Response histories: {n:,}/{total:,}", flush=True
        ),
    )
    response.provenance["material_definition_sha256"] = definition.sha256
    print(write_spectral_response(args.output, response).resolve())


if __name__ == "__main__":
    main()
