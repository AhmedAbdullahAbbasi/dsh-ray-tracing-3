"""Extract three energy-integrated flare images from a production DSH FITS cube.

Each output image integrates one observer-arrival interval. Times are relative
to the direct (unscattered) source arrival, not calendar dates. Run this only
after the production simulation has written its complete FITS product.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from astropy.io import fits

DAY_S = 86_400.0


def _snapshot_indices(time_bins, start_day, exposure_days):
    lower = float(start_day) * DAY_S
    upper = float(start_day + exposure_days) * DAY_S
    low_edges = np.asarray(time_bins["LOW"], dtype=np.float64)
    high_edges = np.asarray(time_bins["HIGH"], dtype=np.float64)
    begin = np.flatnonzero(np.isclose(low_edges, lower, rtol=0.0, atol=0.05))
    end = np.flatnonzero(np.isclose(high_edges, upper, rtol=0.0, atol=0.05))
    if begin.size != 1 or end.size != 1 or end[0] < begin[0]:
        raise ValueError(
            f"arrival interval [{start_day}, {start_day + exposure_days}) days "
            "does not align with TIME_BINS; choose whole-day edges"
        )
    indices = slice(int(begin[0]), int(end[0]) + 1)
    if not np.allclose(
        low_edges[indices][1:], high_edges[indices][:-1], rtol=0, atol=0.05
    ):
        raise ValueError("TIME_BINS contain a gap within a requested snapshot")
    return indices


def _coarse_image_hdu(image, image_header, name, bin_factor, unit):
    """Sum square sky cells and retain a WCS centered on their merged pixels."""
    height, width = image.shape
    coarsened = image.reshape(
        height // bin_factor, bin_factor, width // bin_factor, bin_factor
    ).sum(axis=(1, 3))
    extension = fits.ImageHDU(coarsened, name=name)
    extension.header["BUNIT"] = unit
    extension.header["BINFACT"] = (bin_factor, "native pixels per coarse image axis")
    for axis in (1, 2):
        for prefix in ("CTYPE", "CUNIT"):
            key = f"{prefix}{axis}"
            if key in image_header:
                extension.header[key] = image_header[key]
        step = image_header[f"CDELT{axis}"]
        extension.header[f"CRPIX{axis}"] = 1.0
        extension.header[f"CRVAL{axis}"] = (
            image_header[f"CRVAL{axis}"]
            + ((bin_factor + 1) / 2 - image_header[f"CRPIX{axis}"]) * step
        )
        extension.header[f"CDELT{axis}"] = bin_factor * step
    return extension


def extract_snapshots(
    input_fits: Path,
    output_dir: Path,
    *,
    first_day: int = 3,
    separation_days: int = 3,
    exposure_days: int = 1,
    expected_packets: int = 2_500_000,
    allow_invalid_energy_count: int = 0,
) -> dict:
    """Check transport/product closure, then write three dated 2D FITS images."""
    if first_day < 0 or separation_days <= 0 or exposure_days <= 0:
        raise ValueError(
            "snapshot start must be nonnegative; widths and spacing positive"
        )
    if allow_invalid_energy_count < 0:
        raise ValueError("allow_invalid_energy_count must be nonnegative")
    start_days = [first_day + i * separation_days for i in range(3)]
    with fits.open(input_fits, checksum=True, memmap=True) as hdul:
        hdul.verify("exception")
        header = hdul[0].header
        if header.get("NPACKETS") != expected_packets:
            raise ValueError(f"expected {expected_packets:,} simulated photons")
        if header.get("MATMODEL") != "2-10":
            raise ValueError("expected the integrated 2–10 keV material tables")
        if header.get("SRCSPEC") != "hard-state-powerlaw":
            raise ValueError(
                "expected continuously sampled hard-state power-law source"
            )
        if (
            not np.isfinite(header.get("PHINDEX", np.nan))
            or header.get("EMINKEV") != 2.0
            or header.get("EMAXKEV") != 10.0
        ):
            raise ValueError("missing 2–10 keV power-law source provenance")
        for keyword in ("SCATSHA", "ABSSHA"):
            if len(header.get(keyword, "")) != 64:
                raise ValueError(f"missing material provenance {keyword}")
        statuses = {
            int(code): int(count)
            for code, count in zip(
                hdul["STATUS"].data["STATUS_CODE"],
                hdul["STATUS"].data["COUNT"],
                strict=True,
            )
        }
        if sum(statuses.values()) != expected_packets:
            raise ValueError("transport terminal counts do not sum to packet count")
        if (
            any(statuses.get(code, 0) for code in (0, 4, 6))
            or statuses.get(5, 0) > allow_invalid_energy_count
        ):
            raise ValueError(
                f"transport has numerical/invalid terminal states: {statuses}"
            )

        total = hdul["TOTAL4D"].data
        first = hdul["FIRST4D"].data
        multiple = hdul["MULTI4D"].data
        counts = hdul["EVENT4D"].data
        squared = hdul["HISTQIMG"].data if "HISTQIMG" in hdul else None
        histories = int(hdul["HISTQIMG"].header["NHIST"]) if squared is not None else 0
        time_bins = hdul["TIME_BINS"].data
        if total.ndim != 4 or any(
            array.shape != total.shape for array in (first, multiple, counts)
        ):
            raise ValueError("observer cubes must share shape (time, energy, y, x)")
        if len(time_bins) != total.shape[0]:
            raise ValueError("TIME_BINS length differs from the observer cubes")
        if not np.all(np.isfinite(total)) or np.any(total < 0):
            raise ValueError("nonfinite or negative observer fluence")
        if not np.allclose(total, first + multiple, rtol=1e-6, atol=0):
            raise ValueError("first and multiple scattering do not close to total")

        selections = [
            _snapshot_indices(time_bins, start, exposure_days) for start in start_days
        ]
        snapshots = []
        for start, selection in zip(start_days, selections, strict=True):
            image = np.sum(total[selection], axis=(0, 1), dtype=np.float64)
            first_image = np.sum(first[selection], axis=(0, 1), dtype=np.float64)
            multiple_image = np.sum(multiple[selection], axis=(0, 1), dtype=np.float64)
            count_image = np.sum(counts[selection], axis=(0, 1), dtype=np.int64)
            if not np.allclose(image, first_image + multiple_image, rtol=1e-6, atol=0):
                raise ValueError(f"image fluence closure failed at day {start}")
            if count_image.sum() <= 0 or image.sum() <= 0:
                raise ValueError(
                    f"no scored halo events during days [{start}, {start + exposure_days}); "
                    "check the cloud field of view and choose different days"
                )
            uncertainty = None
            if (
                squared is not None
                and histories > 1
                and selection.stop - selection.start == 1
            ):
                q = np.asarray(squared[selection.start], dtype=np.float64)
                uncertainty = np.sqrt(
                    histories
                    / (histories - 1)
                    * np.maximum(q - image * image / histories, 0)
                )
            snapshots.append(
                (start, image, first_image, multiple_image, count_image, uncertainty)
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "source_fits": str(input_fits.resolve()),
            "packets": expected_packets,
            "cloud": header.get("CLOUD", "unspecified"),
            "source_model": header.get("SRCMODEL", "unspecified"),
            "source_spectrum": header["SRCSPEC"],
            "photon_index": header["PHINDEX"],
            "energy_range_kev": [header["EMINKEV"], header["EMAXKEV"]],
            "material_tables": header["MATMODEL"],
            "scattering_table_sha256": header["SCATSHA"],
            "absorption_table_sha256": header["ABSSHA"],
            "status_counts": statuses,
            "accepted_invalid_energy_packets": statuses.get(5, 0),
            "diagnostics": {
                name.lower(): np.asarray(hdul["DIAGNOSTICS"].data[name][0]).item()
                for name in hdul["DIAGNOSTICS"].data.names
            },
            "time_reference": "days after direct source arrival",
            "products": [],
        }
        for (
            start,
            image,
            first_image,
            multiple_image,
            count_image,
            uncertainty,
        ) in snapshots:
            end = start + exposure_days
            output = output_dir / f"flare_day_{start:03d}_to_{end:03d}.fits"
            image_header = header.copy()
            image_header["TSTART"] = (start * DAY_S, "arrival interval start [s]")
            image_header["TSTOP"] = (end * DAY_S, "arrival interval stop [s]")
            image_header["TIMEREF"] = "direct source arrival"
            image_header["SRCFITS"] = input_fits.name
            image_header["INVENER"] = (
                statuses.get(5, 0),
                "out-of-range energy packets in original run",
            )
            image_header["RUNSTAT"] = (
                "DIAGNOSTIC" if statuses.get(5, 0) else "CLEAN",
                "recovered older run if DIAGNOSTIC",
            )
            image_header["BUNIT"] = "ph cm-2"
            image_header["BTYPE"] = "energy-integrated ideal-observer fluence"
            image_header["NHIST"] = histories
            image_header["UNCERT"] = (
                "HISTORY" if uncertainty is not None else "UNAVAILABLE"
            )
            image_header.add_history(
                "Sum over all energy bands and requested observer arrival bins"
            )
            extensions = [
                fits.PrimaryHDU(image.astype(np.float32), header=image_header),
                fits.ImageHDU(first_image.astype(np.float32), name="FIRSTIMG"),
                fits.ImageHDU(multiple_image.astype(np.float32), name="MULTIIMG"),
                fits.ImageHDU(count_image.astype(np.int32), name="EVENTIMG"),
                hdul["ENERGY_BINS"].copy(),
            ]
            if uncertainty is not None:
                extension = fits.ImageHDU(uncertainty.astype(np.float32), name="STDIMG")
                extension.header["BUNIT"] = "ph cm-2"
                extension.header["ERRTYPE"] = "photon-history standard error"
                extensions.append(extension)
            for extension in extensions[1:]:
                extension.header["BUNIT"] = (
                    "count" if extension.name == "EVENTIMG" else "ph cm-2"
                )
                for key in (
                    "CTYPE1",
                    "CUNIT1",
                    "CRPIX1",
                    "CRVAL1",
                    "CDELT1",
                    "CTYPE2",
                    "CUNIT2",
                    "CRPIX2",
                    "CRVAL2",
                    "CDELT2",
                ):
                    if key in image_header:
                        extension.header[key] = image_header[key]
            bin_factor = math.gcd(20, image.shape[0], image.shape[1])
            if bin_factor > 1:
                extensions += [
                    _coarse_image_hdu(
                        image.astype(np.float32),
                        image_header,
                        "COARSEFL",
                        bin_factor,
                        "ph cm-2",
                    ),
                    _coarse_image_hdu(
                        count_image.astype(np.int32),
                        image_header,
                        "COARSEEV",
                        bin_factor,
                        "count",
                    ),
                ]
            fits.HDUList(extensions).writeto(output, overwrite=True, checksum=True)
            manifest["products"].append(
                {
                    "file": output.name,
                    "arrival_start_day": start,
                    "arrival_stop_day": end,
                    "observer_fluence_ph_cm2": float(image.sum()),
                    "scored_event_count": int(count_image.sum()),
                    "first_scatter_fluence_ph_cm2": float(first_image.sum()),
                    "multiple_scatter_fluence_ph_cm2": float(multiple_image.sum()),
                    "standard_error_image_available": uncertainty is not None,
                }
            )

    manifest_path = output_dir / "flare_snapshots_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-day", type=int, default=3)
    parser.add_argument("--separation-days", type=int, default=3)
    parser.add_argument("--exposure-days", type=int, default=1)
    parser.add_argument("--expected-packets", type=int, default=2_500_000)
    parser.add_argument(
        "--allow-invalid-energy-count",
        type=int,
        default=0,
        help="explicitly allow this many rejected energy packets from an older run",
    )
    args = parser.parse_args()
    report = extract_snapshots(
        args.input_fits,
        args.output_dir,
        first_day=args.first_day,
        separation_days=args.separation_days,
        exposure_days=args.exposure_days,
        expected_packets=args.expected_packets,
        allow_invalid_energy_count=args.allow_invalid_energy_count,
    )
    for product in report["products"]:
        print(
            f"{product['file']}: {product['scored_event_count']:,} events, "
            f"{product['observer_fluence_ph_cm2']:.7g} ph cm^-2"
        )
    if report["accepted_invalid_energy_packets"]:
        print(
            "Diagnostic recovery only: "
            f"{report['accepted_invalid_energy_packets']} old-run energy packets "
            "were excluded; snapshot headers record INVENER."
        )
    print("Wrote flare_snapshots_manifest.json")


if __name__ == "__main__":
    main()
