"""Numerical and provenance checks for file-input ideal-observer products.

This is a product-integrity audit, not a Monte Carlo convergence test or an
independent astrophysical reference calculation.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .run import ResolvedRun, _file_sha256


def _close(left, right, *, rtol=2e-6, atol=1e-12) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=rtol, atol=atol))


def _input_hashes(config: ResolvedRun, archive) -> bool:
    try:
        expected = {
            "source_fits_sha256": _file_sha256(config.source_fits),
            "scattering_table_sha256": _file_sha256(config.scattering),
            "absorption_table_sha256": _file_sha256(config.absorption),
            "cloud_fits_sha256": _file_sha256(config.cloud_fits)
            if config.cloud_fits
            else "",
        }
    except FileNotFoundError:
        return False
    return all(str(archive[key]) == value for key, value in expected.items())


def _source_cells_close(archive) -> bool:
    fluence = np.asarray(archive["source_cell_fluence"], dtype=np.float64)
    cdf = np.asarray(archive["source_flat_cdf"], dtype=np.float64)
    start = np.asarray(archive["source_cell_time_start_s"])
    stop = np.asarray(archive["source_cell_time_stop_s"])
    low = np.asarray(archive["source_cell_energy_low_kev"])
    high = np.asarray(archive["source_cell_energy_high_kev"])
    kind = np.asarray(archive["source_cell_kind"])
    n = fluence.size
    if (
        n == 0
        or any(array.shape != (n,) for array in (cdf, start, stop, low, high, kind))
        or not all(
            np.all(np.isfinite(array))
            for array in (fluence, cdf, start, stop, low, high)
        )
        or not np.all(fluence > 0)
        or not np.all(stop > start)
        or not np.all(low > 0)
        or not np.all(np.isin(kind, [0, 1, 2]))
        or not np.all(np.where(kind == 0, low == high, low < high))
    ):
        return False
    return bool(
        _close(cdf, np.cumsum(fluence) / fluence.sum(), rtol=2e-6)
        and _close(archive["source_total_fluence"], fluence.sum())
        and _close(archive["source_fluence"], fluence.sum(), rtol=1e-5)
    )


def _archive_checks(config: ResolvedRun, archive, report: dict) -> dict[str, bool]:
    status = np.asarray(archive["transport_status_count"])
    packets = int(archive["requested_packet_count"])
    total = archive["total_fluence"]
    first = archive["first_scatter_fluence"]
    multiple = archive["multiple_scatter_fluence"]
    counts = archive["event_count"]
    resolved = str(archive["resolved_config"])
    return {
        "schema7": int(archive["output_schema_version"]) == 7,
        "configuration_digest": bool(
            hashlib.sha256(resolved.encode()).hexdigest()
            == str(archive["resolved_config_sha256"])
            and hashlib.sha256(str(archive["original_config"]).encode()).hexdigest()
            == str(archive["original_config_sha256"])
            == config.config_sha256
        ),
        "input_file_hashes": _input_hashes(config, archive),
        "source_cells": _source_cells_close(archive),
        "cloud_column_closure": _close(
            archive["cloud_n_h_cm3"]
            * archive["cloud_radial_bin_width_cm"][:, None, None],
            archive["cloud_delta_nh_cm2"],
            rtol=2e-6,
            atol=0,
        ),
        "clean_terminal_status": bool(
            status.shape == (7,)
            and status.sum() == packets
            and status[0] == 0
            and np.all(status[4:] == 0)
            and int(archive["source_packet_count"]) == packets
            and int(archive["history_count"]) == packets
        ),
        "event_accounting": bool(
            int(np.sum(counts, dtype=np.int64)) == int(archive["binned_event_count"])
            and int(archive["binned_event_count"])
            + int(archive["unbinned_event_count"])
            == int(archive["valid_event_count"])
            == int(archive["scored_observer_event_count"])
        ),
        "fluence_accounting": bool(
            _close(total, first + multiple, rtol=2e-6)
            and _close(
                np.sum(total, dtype=np.float64),
                archive["binned_weight_observer_fluence"],
                rtol=2e-5,
            )
            and _close(
                archive["scored_observer_fluence"],
                archive["binned_weight_observer_fluence"]
                + archive["unbinned_weight_observer_fluence"],
                rtol=2e-5,
            )
        ),
        "history_moments": bool(
            archive["total_fluence_squared"].shape == total.shape
            and archive["time_image_fluence_squared"].shape
            == (total.shape[0], *total.shape[2:])
            and archive["time_order_fluence_cross"].shape
            == (total.shape[0], 3, total.shape[0], 3)
            and _close(
                archive["time_order_fluence_sum"].sum(axis=1),
                total.sum(axis=(1, 2, 3), dtype=np.float64),
                rtol=2e-5,
            )
        ),
        "run_report": bool(
            report["status_counts"] == status.tolist()
            and report["numerical_passed"]
            and report["cap_passed"] == (status[4] == 0)
            and report["validated_science_product"] is False
            and report["config_sha256"] == str(archive["resolved_config_sha256"])
            and report["original_config_sha256"]
            == str(archive["original_config_sha256"])
            and report["source_sha256"] == str(archive["source_fits_sha256"])
            and report["scattering_sha256"] == str(archive["scattering_table_sha256"])
            and report["absorption_sha256"] == str(archive["absorption_table_sha256"])
            and report["cloud_sha256"] == (str(archive["cloud_fits_sha256"]) or None)
        ),
    }


def audit_configured_run(config: ResolvedRun) -> dict:
    """Cross-check schema-7 NPZ/FITS/report and currently supplied input bytes."""

    from astropy.io import fits

    report_path = config.output_npz.with_suffix(".run_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with np.load(config.output_npz, allow_pickle=False) as archive:
        checks = _archive_checks(config, archive, report)
        with fits.open(config.output_fits, checksum=True, memmap=True) as hdus:
            checks["fits_checksums"] = all(
                hdu.verify_checksum() == 1 and hdu.verify_datasum() == 1 for hdu in hdus
            )
            header = hdus[0].header
            checks["fits_provenance"] = bool(
                header.get("OUTSCHEM") == 7
                and header.get("NPACKETS") == int(archive["requested_packet_count"])
                and header.get("SRCFSHA") == str(archive["source_fits_sha256"])
                and header.get("SCATSHA") == str(archive["scattering_table_sha256"])
                and header.get("ABSSHA") == str(archive["absorption_table_sha256"])
                and header.get("CFGSHA") == str(archive["resolved_config_sha256"])
                and (header.get("CLDFSHA") or "") == str(archive["cloud_fits_sha256"])
            )
            checks["fits_observer_cubes"] = all(
                _close(hdus[hdu].data, archive[key], rtol=1e-6, atol=1e-9)
                for hdu, key in (
                    ("TOTAL4D", "total_fluence"),
                    ("FIRST4D", "first_scatter_fluence"),
                    ("MULTI4D", "multiple_scatter_fluence"),
                    ("EVENT4D", "event_count"),
                )
            )
            checks["fits_axes"] = all(
                np.array_equal(hdus[hdu].data["LOW"], archive[key][:-1])
                and np.array_equal(hdus[hdu].data["HIGH"], archive[key][1:])
                for hdu, key in (
                    ("TIME_BINS", "arrival_time_edges_s"),
                    ("ENERGY_BINS", "energy_edges_kev"),
                    ("X_BINS", "sky_x_edges_arcsec"),
                    ("Y_BINS", "sky_y_edges_arcsec"),
                    ("Z_BINS", "cloud_z_edges_kpc"),
                )
            )
            checks["fits_images"] = all(
                np.array_equal(
                    hdus[hdu].data,
                    np.sum(archive[key], axis=(0, 1), dtype=np.float64).astype(
                        np.float32
                    ),
                )
                for hdu, key in (
                    (0, "total_fluence"),
                    ("FIRSTIMG", "first_scatter_fluence"),
                    ("MULTIIMG", "multiple_scatter_fluence"),
                )
            )
            checks["fits_cloud"] = bool(
                _close(
                    hdus["CLOUDNH"].data,
                    archive["cloud_delta_nh_cm2"],
                    rtol=0,
                    atol=0,
                )
                and _close(
                    hdus["CLOUDDEN"].data, archive["cloud_n_h_cm3"], rtol=0, atol=0
                )
            )
            source = hdus["SOURCE"].data
            checks["fits_source_cells"] = bool(
                np.array_equal(source["KIND"], archive["source_cell_kind"])
                and np.array_equal(
                    source["TIME_INDEX"], archive["source_cell_time_index"]
                )
                and np.array_equal(
                    source["ENERGY_INDEX"], archive["source_cell_spectral_bin_index"]
                )
                and _close(source["TIME_LOW"], archive["source_cell_time_start_s"])
                and _close(source["TIME_HIGH"], archive["source_cell_time_stop_s"])
                and _close(source["ENERGY_LOW"], archive["source_cell_energy_low_kev"])
                and _close(
                    source["ENERGY_HIGH"], archive["source_cell_energy_high_kev"]
                )
                and _close(source["PHOTON_INDEX"], archive["source_cell_photon_index"])
                and _close(source["FLUENCE"], archive["source_cell_fluence"])
                and _close(source["SAMPLING_CDF"], archive["source_flat_cdf"])
                and _close(
                    source["PHOTON_FLUX"],
                    archive["source_cell_fluence"]
                    / (
                        archive["source_cell_time_stop_s"]
                        - archive["source_cell_time_start_s"]
                    ),
                )
            )
            physics = hdus["PHYSICS"].data
            checks["fits_materials"] = bool(
                _close(physics["ENERGY"], archive["physics_energy_kev"], atol=0)
                and _close(
                    physics["SIGMA_SCA"],
                    archive["physics_scattering_cross_section_cm2_per_h"],
                    atol=0,
                )
                and _close(
                    physics["SIGMA_ABS"],
                    archive["physics_absorption_cross_section_cm2_per_h"],
                    atol=0,
                )
                and _close(
                    hdus["SCATCDF"].data,
                    archive["physics_scattering_angle_cdf"],
                    atol=0,
                )
                and _close(
                    hdus["DSIGMA"].data,
                    archive["physics_differential_cross_section_cm2_per_sr_per_h"],
                    atol=0,
                )
            )
            counts = np.zeros(7, dtype=np.int64)
            counts[hdus["STATUS"].data["STATUS_CODE"]] = hdus["STATUS"].data["COUNT"]
            checks["fits_status"] = bool(
                np.array_equal(counts, archive["transport_status_count"])
            )
    return {
        "name": config.name,
        "checks": checks,
        "all_passed": all(checks.values()),
        "validated_science_product": False,
    }
