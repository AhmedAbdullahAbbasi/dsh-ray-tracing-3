"""Extract dated monochromatic snapshots from an audited schema-7 run.

Images contain ideal-observer fluence, not detector counts. Per-pixel errors
come from photon-history second moments for one aligned time and energy bin.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..config.load import load_run_config
from .audit import audit_configured_run

DAY_S = 86_400.0
WCS_KEYS = (
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
)
PROVENANCE_KEYS = (
    "NPACKETS",
    "SRCFSHA",
    "CLDFSHA",
    "SCATSHA",
    "ABSSHA",
    "CFGSHA",
)


def _time_index(table, day: int, exposure_days: int) -> int:
    low = np.asarray(table["LOW"], dtype=np.float64)
    high = np.asarray(table["HIGH"], dtype=np.float64)
    matched = np.flatnonzero(
        np.isclose(low, day * DAY_S, rtol=0, atol=0.05)
        & np.isclose(high, (day + exposure_days) * DAY_S, rtol=0, atol=0.05)
    )
    if matched.size != 1:
        raise ValueError(
            f"[{day}, {day + exposure_days}) days must match one arrival bin "
            "to compute its exact per-pixel history uncertainty"
        )
    return int(matched[0])


def _coarse_hdu(image, primary_header, name, factor, unit):
    height, width = image.shape
    coarsened = image.reshape(height // factor, factor, width // factor, factor).sum(
        axis=(1, 3)
    )
    hdu = fits.ImageHDU(coarsened, name=name)
    hdu.header["BUNIT"] = unit
    hdu.header["BINFACT"] = factor
    for axis in (1, 2):
        for key in (f"CTYPE{axis}", f"CUNIT{axis}"):
            hdu.header[key] = primary_header[key]
        step = primary_header[f"CDELT{axis}"]
        hdu.header[f"CRPIX{axis}"] = 1.0
        hdu.header[f"CRVAL{axis}"] = (
            primary_header[f"CRVAL{axis}"]
            + ((factor + 1) / 2 - primary_header[f"CRPIX{axis}"]) * step
        )
        hdu.header[f"CDELT{axis}"] = factor * step
    return hdu


def extract_snapshots(
    config_path: Path,
    output_dir: Path | None = None,
    *,
    days: tuple[int, ...] = (3, 6, 9),
    exposure_days: int = 1,
) -> dict:
    """Audit full inputs/product, then write one FITS image per selected bin."""
    if (
        not days
        or any(
            isinstance(day, bool) or not isinstance(day, int) or day < 0 for day in days
        )
        or len(set(days)) != len(days)
    ):
        raise ValueError("days must contain distinct nonnegative integers")
    if (
        isinstance(exposure_days, bool)
        or not isinstance(exposure_days, int)
        or exposure_days < 1
    ):
        raise ValueError("exposure_days must be a positive integer")
    config = load_run_config(config_path)
    if config.components != "lines" or config.scene_kind != "fits":
        raise ValueError("snapshot input must select lines and a FITS cloud")
    audit = audit_configured_run(config)
    if not audit["all_passed"]:
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise ValueError(f"configured run failed product-integrity audit: {failed}")
    destination = Path(output_dir or config.output_npz.parent / "snapshots").resolve()
    entries = []
    with fits.open(config.output_fits, memmap=True) as hdus:
        header = hdus[0].header
        if header.get("OUTSCHEM") != 7 or header.get("SRCSPEC") != "file-cells":
            raise ValueError("snapshots require a schema-7 file-input product")
        source = hdus["SOURCE"].data
        if (
            len(source) != 1
            or int(source["KIND"][0]) != 0
            or source["ENERGY_LOW"][0] != source["ENERGY_HIGH"][0]
        ):
            raise ValueError("snapshots require one monochromatic source cell")
        energy_kev = float(source["ENERGY_LOW"][0])
        total = hdus["TOTAL4D"].data
        first = hdus["FIRST4D"].data
        multiple = hdus["MULTI4D"].data
        counts = hdus["EVENT4D"].data
        squared = hdus["HISTQ4D"].data
        if (
            total.ndim != 4
            or total.shape[1] != 1
            or any(
                array.shape != total.shape
                for array in (first, multiple, counts, squared)
            )
            or len(hdus["TIME_BINS"].data) != total.shape[0]
            or len(hdus["ENERGY_BINS"].data) != 1
        ):
            raise ValueError("monochromatic observer cubes have incompatible shapes")
        histories = int(hdus["HISTQ4D"].header["NHIST"])
        if histories != config.packets or histories <= 1:
            raise ValueError("photon-history count differs from requested packets")

        for day in days:
            index = _time_index(hdus["TIME_BINS"].data, day, exposure_days)
            image = np.asarray(total[index, 0], dtype=np.float64)
            first_image = np.asarray(first[index, 0], dtype=np.float64)
            multi_image = np.asarray(multiple[index, 0], dtype=np.float64)
            event_image = np.asarray(counts[index, 0], dtype=np.int64)
            q = np.asarray(squared[index, 0], dtype=np.float64)
            if not np.allclose(image, first_image + multi_image, rtol=2e-6, atol=1e-12):
                raise ValueError(f"first/multiple fluence does not close at day {day}")
            variance = (
                histories
                / (histories - 1)
                * np.maximum(q - image * image / histories, 0.0)
            )
            uncertainty = np.sqrt(variance)
            snapshot_header = fits.Header()
            for key in (*WCS_KEYS, *PROVENANCE_KEYS):
                if key in header:
                    snapshot_header[key] = header[key]
            snapshot_header["SNAPSCHE"] = 1
            snapshot_header["BUNIT"] = "ph cm-2"
            snapshot_header["BTYPE"] = "ideal-observer fluence"
            snapshot_header["TSTART"] = day * DAY_S
            snapshot_header["TSTOP"] = (day + exposure_days) * DAY_S
            snapshot_header["TIMEREF"] = "direct source arrival"
            snapshot_header["ELINE"] = energy_kev
            snapshot_header["NHIST"] = histories
            hdulist = [
                fits.PrimaryHDU(image.astype(np.float32), header=snapshot_header)
            ]
            for name, values, unit in (
                ("FIRSTIMG", first_image, "ph cm-2"),
                ("MULTIIMG", multi_image, "ph cm-2"),
                ("EVENTIMG", event_image, "count"),
                ("STDIMG", uncertainty, "ph cm-2"),
            ):
                extension = fits.ImageHDU(
                    values.astype(np.int32 if name == "EVENTIMG" else np.float32),
                    name=name,
                )
                extension.header["BUNIT"] = unit
                for key in WCS_KEYS:
                    if key in snapshot_header:
                        extension.header[key] = snapshot_header[key]
                hdulist.append(extension)
            factor = math.gcd(20, *image.shape)
            if factor > 1:
                hdulist.extend(
                    (
                        _coarse_hdu(
                            image.astype(np.float32),
                            snapshot_header,
                            "COARSEFL",
                            factor,
                            "ph cm-2",
                        ),
                        _coarse_hdu(
                            event_image.astype(np.int32),
                            snapshot_header,
                            "COARSEEV",
                            factor,
                            "count",
                        ),
                    )
                )
            destination.mkdir(parents=True, exist_ok=True)
            filename = f"line_day_{day:03d}_to_{day + exposure_days:03d}.fits"
            fits.HDUList(hdulist).writeto(
                destination / filename, overwrite=True, checksum=True
            )
            entries.append(
                {
                    "file": filename,
                    "arrival_start_day": day,
                    "arrival_stop_day": day + exposure_days,
                    "total_fluence_ph_cm2": float(image.sum(dtype=np.float64)),
                    "first_fluence_ph_cm2": float(first_image.sum(dtype=np.float64)),
                    "multiple_fluence_ph_cm2": float(multi_image.sum(dtype=np.float64)),
                    "scored_event_count": int(event_image.sum(dtype=np.int64)),
                    "pixels_with_events": int(np.count_nonzero(event_image)),
                }
            )
    manifest = {
        "input_config": str(config.config_path),
        "input_fits": str(config.output_fits),
        "source_energy_kev": energy_kev,
        "packets": config.packets,
        "product_integrity_passed": True,
        "validated_science_product": False,
        "snapshots": entries,
    }
    (destination / "line_snapshots_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--days", nargs="+", type=int, default=[3, 6, 9])
    parser.add_argument("--exposure-days", type=int, default=1)
    args = parser.parse_args()
    report = extract_snapshots(
        args.config,
        args.output_dir,
        days=tuple(args.days),
        exposure_days=args.exposure_days,
    )
    for item in report["snapshots"]:
        print(
            f"{item['file']}: {item['scored_event_count']:,} scored events, "
            f"{item['total_fluence_ph_cm2']:.7g} ph cm^-2"
        )


if __name__ == "__main__":
    main()
