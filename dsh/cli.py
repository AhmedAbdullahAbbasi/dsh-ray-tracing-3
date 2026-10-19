"""Compatibility CLI: translate old flags to the shared configured runner."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cloud-fits",
        type=Path,
        help=(
            "optional physical delta_NH FITS cube; the source distance must "
            "lie beyond its outer radial cell edge"
        ),
    )
    parser.add_argument("--source-distance-kpc", type=float, default=10.0)
    parser.add_argument(
        "--launch-sampler",
        choices=("radial", "rectangular"),
        default="radial",
        help="smooth circular importance proposal (default) or the uniform rectangular baseline",
    )
    parser.add_argument(
        "--materials",
        choices=("v1", "2-10", "declared"),
        default="v1",
        help="frozen three-energy default, checked 2–10 keV, or an explicitly supplied complete audited material",
    )
    parser.add_argument(
        "--material-directory",
        type=Path,
        help="explicit complete audited material directory for --materials declared",
    )
    parser.add_argument(
        "--source-model",
        choices=("constant-flare", "exponential-decay"),
        default="constant-flare",
    )
    spectrum_group = parser.add_mutually_exclusive_group()
    spectrum_group.add_argument(
        "--source-spectrum",
        choices=("representative", "hard-state-powerlaw"),
        default="representative",
        help="three fixed energies (default) or continuous 2–10 keV hard-state power law",
    )
    spectrum_group.add_argument(
        "--source-history",
        type=Path,
        help="intrinsic spectral-history JSON with explicit epoch, units, and cell slopes",
    )
    parser.add_argument(
        "--photon-index",
        type=float,
        default=1.7,
        help="hard-state photon index Gamma (default: 1.7)",
    )
    parser.add_argument(
        "--total-2-10-photon-flux",
        type=float,
        default=0.038,
        help="unabsorbed 2–10 keV photon flux [ph cm^-2 s^-1] in hard-state mode",
    )
    parser.add_argument(
        "--peak-band-fluxes",
        type=float,
        nargs=3,
        metavar=("F3P3", "F4P9", "F6P9"),
        default=(2.0e-2, 1.2e-2, 6.0e-3),
        help="three representative-energy fluxes in ph cm^-2 s^-1 (representative mode only)",
    )
    parser.add_argument(
        "--baseline-band-fluxes",
        type=float,
        nargs=3,
        metavar=("F3P3", "F4P9", "F6P9"),
        default=(0.0, 0.0, 0.0),
        help="asymptotic band fluxes for exponential-decay mode",
    )
    parser.add_argument("--decay-time-days", type=float, default=25.0)
    parser.add_argument("--decay-start-days", type=float, default=0.0)
    parser.add_argument("--decay-duration-days", type=float, default=120.0)
    parser.add_argument("--source-time-bin-days", type=float, default=0.25)
    parser.add_argument("--packets", type=int, default=4_096)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-interactions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--arrival-days", type=float, default=60.0)
    parser.add_argument("--time-bin-days", type=float, default=1.0)
    parser.add_argument(
        "--arrival-time-edges-days",
        type=float,
        nargs="+",
        metavar="DAY",
        help=(
            "optional explicit observer arrival-bin edges in days, beginning "
            "at zero; overrides --arrival-days and --time-bin-days"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dsh_v1_ideal_observer.npz"),
    )
    parser.add_argument(
        "--fits-output",
        type=Path,
        help="FITS path; defaults to the NPZ output path with suffix .fits",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        help="backend; otherwise follows an explicit JAX environment or defaults to CPU",
    )
    parser.add_argument(
        "--check", action="store_true", help="validate without transport or outputs"
    )
    return parser.parse_args(argv)


def _arrival_edges(arrival_days, time_bin_days, explicit_edges_days=None):
    # Kept for callers of this historical helper; the implementation is shared.
    from .compat import legacy_arrival_edges

    return legacy_arrival_edges(arrival_days, time_bin_days, explicit_edges_days)


def main(argv=None):
    args = parse_arguments(argv)
    try:
        from .command import bootstrap
        from .compat import run_cli

        device = (
            args.device
            or os.environ.get("JAX_PLATFORM_NAME")
            or os.environ.get("JAX_PLATFORMS", "cpu").split(",")[0]
        )
        device = "gpu" if device == "cuda" else device
        if device not in {"cpu", "gpu"}:
            raise ValueError("--device: specify cpu or gpu for this JAX environment")
        bootstrap(device)
        return run_cli(args, device)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"dsh-v1: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
