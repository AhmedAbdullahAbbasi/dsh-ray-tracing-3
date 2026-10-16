"""Fold an intrinsic spectral history through exact halo response samples."""

import argparse
import json
from pathlib import Path

import numpy as np

from dsh.io.atomic import atomic_savez_compressed
from dsh.observer.response import fold_spectral_response, load_spectral_response
from dsh.sources.histories import load_spectral_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--source-history", type=Path, required=True)
    parser.add_argument("--arrival-time-edges-s", type=float, nargs="+", required=True)
    parser.add_argument("--energy-edges-kev", type=float, nargs="+", required=True)
    parser.add_argument("--radial-edges-arcsec", type=float, nargs="+", required=True)
    parser.add_argument("--no-covariance", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        raise ValueError("output path must end in .npz")
    response = load_spectral_response(args.response)
    source, source_metadata = load_spectral_history(args.source_history)
    result = fold_spectral_response(
        response,
        source,
        args.arrival_time_edges_s,
        args.energy_edges_kev,
        args.radial_edges_arcsec,
        covariance=not args.no_covariance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "fluence_units": "ph cm^-2",
        "axes": ["arrival_time", "energy", "radius"],
        "uncertainty": "photon-grouped finite-response Monte Carlo covariance; C-order flattening",
        "n_histories": result.n_histories,
        "response_provenance": response.provenance,
        **source_metadata,
    }
    payload = {
        "fluence": result.fluence,
        "standard_error": result.standard_error,
        "arrival_time_edges_s": args.arrival_time_edges_s,
        "energy_edges_kev": args.energy_edges_kev,
        "radial_edges_arcsec": args.radial_edges_arcsec,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if result.covariance is not None:
        payload["covariance"] = result.covariance
    atomic_savez_compressed(args.output, **payload)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
