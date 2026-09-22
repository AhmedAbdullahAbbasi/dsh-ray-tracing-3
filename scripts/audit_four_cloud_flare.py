"""Screen saved four-cloud flare products for numerical and spatial adequacy.

Use the NPZ alone for a diagnostic of an existing run. Include the full FITS
and snapshot directory to verify generated products before calling them ready.
This audit never launches photon transport.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from dsh.physics.materials import DEFAULT_2_10_ABSORPTION, DEFAULT_2_10_SCATTERING
from dsh.validation.flare_production import inspect_flare_arrays, snapshot_slice


def _expected_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args):
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _fits_agreement(full_fits, archive, rows, snapshot_dir):
    from astropy.io import fits

    checks = {}
    with fits.open(full_fits, checksum=True, memmap=True) as hdul:
        hdul.verify("exception")
        header = hdul[0].header
        checks["metadata"] = bool(
            header.get("NPACKETS") == int(archive["requested_packet_count"])
            and header.get("MATMODEL") == str(archive["material_tables"])
            and header.get("SRCSPEC") == str(archive["source_spectrum"])
            and header.get("SCATSHA") == str(archive["scattering_table_sha256"])
            and header.get("ABSSHA") == str(archive["absorption_table_sha256"])
            and header.get("EMINKEV") == 2.0
            and header.get("EMAXKEV") == 10.0
        )
        fits_status = np.zeros(7, dtype=np.int64)
        for code, count in zip(
            hdul["STATUS"].data["STATUS_CODE"],
            hdul["STATUS"].data["COUNT"],
            strict=True,
        ):
            fits_status[int(code)] = int(count)
        checks["status_counts"] = bool(
            np.array_equal(fits_status, archive["transport_status_count"])
        )
        fits_edges = np.r_[
            hdul["TIME_BINS"].data["LOW"], hdul["TIME_BINS"].data["HIGH"][-1]
        ]
        checks["arrival_edges"] = bool(
            np.allclose(fits_edges, archive["arrival_time_edges_s"], rtol=0, atol=0.05)
        )
        for fits_name, npz_name in (
            ("TOTAL4D", "total_fluence"),
            ("FIRST4D", "first_scatter_fluence"),
            ("MULTI4D", "multiple_scatter_fluence"),
            ("EVENT4D", "event_count"),
        ):
            actual = hdul[fits_name].data
            expected = archive[npz_name]
            checks[fits_name.lower()] = bool(
                np.array_equal(actual, expected)
                if npz_name == "event_count"
                else np.allclose(actual, expected, rtol=1e-6, atol=1e-9)
            )
    if snapshot_dir is not None:
        counts = archive["event_count"]
        total = archive["total_fluence"]
        time_edges = archive["arrival_time_edges_s"]
        for row in rows:
            day = row["start_day"]
            path = snapshot_dir / f"flare_day_{day:03d}_to_{row['stop_day']:03d}.fits"
            if not path.is_file():
                checks[f"snapshot_{day}_present"] = False
                continue
            selection = snapshot_slice(time_edges, day, row["stop_day"] - day)
            with fits.open(path, checksum=True) as hdul:
                hdul.verify("exception")
                image = total[selection].sum(axis=(0, 1), dtype=np.float64)
                event_image = counts[selection].sum(axis=(0, 1), dtype=np.int64)
                checks[f"snapshot_{day}_pixels"] = bool(
                    np.allclose(hdul[0].data, image, rtol=2e-6, atol=1e-8)
                    and np.array_equal(hdul["EVENTIMG"].data, event_image)
                )
                checks[f"snapshot_{day}_provenance"] = bool(
                    hdul[0].header.get("TSTART") == day * 86_400
                    and hdul[0].header.get("TSTOP") == row["stop_day"] * 86_400
                    and hdul[0].header.get("BUNIT") == "ph cm-2"
                    and hdul[0].header.get("RUNSTAT") == "CLEAN"
                )
    return checks


def audit_run(
    npz_path: Path,
    *,
    full_fits: Path | None = None,
    snapshot_dir: Path | None = None,
    expected_packets: int = 2_500_000,
    first_day: int = 3,
    separation_days: int = 3,
    exposure_days: int = 1,
    coarse_factor: int = 20,
    min_snapshot_events: int = 2_000,
    min_supported_cell_events: int = 10,
    min_supported_event_fraction: float = 0.8,
) -> dict:
    if snapshot_dir is not None and full_fits is None:
        raise ValueError("snapshot directory requires --input-fits")
    with np.load(npz_path, allow_pickle=False) as archive:
        model_checks = {
            "continuous_2_to_10_kev_source": bool(
                str(archive["source_spectrum"]) == "hard-state-powerlaw"
                and str(archive["source_model"]) == "constant-flare"
                and np.array_equal(archive["energy_edges_kev"], [2.0, 4.0, 6.0, 10.0])
                and np.isfinite(float(archive["source_photon_index"]))
            ),
            "matched_material_hashes": bool(
                str(archive["material_tables"]) == "2-10"
                and str(archive["scattering_table_sha256"])
                == _expected_sha(DEFAULT_2_10_SCATTERING)
                and str(archive["absorption_table_sha256"])
                == _expected_sha(DEFAULT_2_10_ABSORPTION)
            ),
            "packet_count": bool(
                int(archive["requested_packet_count"]) == expected_packets
                and int(archive["source_packet_count"]) == expected_packets
            ),
            "binned_event_closure": bool(
                int(archive["binned_event_count"])
                == int(np.sum(archive["event_count"], dtype=np.int64))
                and int(archive["valid_event_count"])
                == int(archive["binned_event_count"])
                + int(archive["unbinned_event_count"])
            ),
            "fluence_and_bin_range_closure": bool(
                np.isclose(
                    float(archive["scored_observer_fluence"]),
                    float(archive["binned_weight_observer_fluence"])
                    + float(archive["unbinned_weight_observer_fluence"]),
                    rtol=1e-5,
                )
                and int(archive["outside_sky_event_count"]) == 0
                and int(archive["outside_energy_event_count"]) == 0
                and float(archive["outside_arrival_time_weight_observer_fluence"])
                <= 0.01 * float(archive["scored_observer_fluence"])
            ),
        }
        result = inspect_flare_arrays(
            counts=archive["event_count"],
            total=archive["total_fluence"],
            first=archive["first_scatter_fluence"],
            multiple=archive["multiple_scatter_fluence"],
            arrival_edges_s=archive["arrival_time_edges_s"],
            status_counts=archive["transport_status_count"],
            expected_packets=expected_packets,
            first_day=first_day,
            separation_days=separation_days,
            exposure_days=exposure_days,
            coarse_factor=coarse_factor,
            min_snapshot_events=min_snapshot_events,
            min_supported_cell_events=min_supported_cell_events,
            min_supported_event_fraction=min_supported_event_fraction,
        )
        file_checks = (
            _fits_agreement(full_fits, archive, result["snapshot_rows"], snapshot_dir)
            if full_fits is not None
            else {}
        )
        result.update(
            {
                "stage": "9E_production_flare_readiness",
                "git_head": _git("rev-parse", "HEAD"),
                "git_status": _git("status", "--short"),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "input_npz": str(npz_path.resolve()),
                "input_fits": str(full_fits.resolve()) if full_fits else None,
                "snapshot_directory": (
                    str(snapshot_dir.resolve()) if snapshot_dir else None
                ),
                "cloud_description": str(archive["cloud_description"]),
                "source_fluence_ph_cm2": float(archive["source_total_fluence"]),
                "score_event_count": int(archive["scored_observer_event_count"]),
                "binned_event_count": int(archive["binned_event_count"]),
                "outside_arrival_event_count": int(
                    archive["outside_arrival_time_event_count"]
                ),
                "outside_arrival_fluence_ph_cm2": float(
                    archive["outside_arrival_time_weight_observer_fluence"]
                ),
                "configuration": {
                    "expected_packets": expected_packets,
                    "first_day": first_day,
                    "separation_days": separation_days,
                    "exposure_days": exposure_days,
                    "coarse_factor": coarse_factor,
                    "min_snapshot_events": min_snapshot_events,
                    "min_supported_cell_events": min_supported_cell_events,
                    "min_supported_event_fraction": min_supported_event_fraction,
                },
                "model_checks": model_checks,
                "fits_checks": file_checks,
            }
        )
        result["archive_screen_passed"] = bool(
            result["all_passed"] and all(model_checks.values())
        )
        result["full_product_checks_performed"] = bool(
            full_fits is not None and snapshot_dir is not None
        )
        result["all_passed"] = bool(
            result["archive_screen_passed"]
            and result["full_product_checks_performed"]
            and all(file_checks.values())
        )
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--input-fits", type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-packets", type=int, default=2_500_000)
    parser.add_argument("--first-day", type=int, default=3)
    parser.add_argument("--separation-days", type=int, default=3)
    parser.add_argument("--exposure-days", type=int, default=1)
    parser.add_argument("--coarse-factor", type=int, default=20)
    parser.add_argument("--min-snapshot-events", type=int, default=2_000)
    parser.add_argument("--min-supported-cell-events", type=int, default=10)
    parser.add_argument("--min-supported-event-fraction", type=float, default=0.8)
    args = parser.parse_args()
    report = audit_run(
        args.input_npz,
        full_fits=args.input_fits,
        snapshot_dir=args.snapshot_dir,
        expected_packets=args.expected_packets,
        first_day=args.first_day,
        separation_days=args.separation_days,
        exposure_days=args.exposure_days,
        coarse_factor=args.coarse_factor,
        min_snapshot_events=args.min_snapshot_events,
        min_supported_cell_events=args.min_supported_cell_events,
        min_supported_event_fraction=args.min_supported_event_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}; all_passed={report['all_passed']}")
    for row in report["snapshot_rows"]:
        print(
            f"  days [{row['start_day']},{row['stop_day']}): "
            f"{row['event_count']} events, "
            f"{row['supported_event_fraction']:.1%} in supported "
            f"{args.coarse_factor}-pixel cells; "
            f"passed={row['passed']}"
        )
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
