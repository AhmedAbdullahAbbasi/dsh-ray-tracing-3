"""Run portable configured smoke cases and independently inspect their products.

Use the checkout's Python environment. The report records the actual operating
system; a Linux run is never reported as Windows validation. This is an
orchestration/export gate, not a Monte Carlo convergence experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def matching(actual, expected, label, *, rtol=1e-6):
    require(np.shape(actual) == np.shape(expected), f"{label}: shape mismatch")
    require(
        np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)),
        f"{label}: nonfinite values",
    )
    require(np.allclose(actual, expected, rtol=rtol, atol=0), f"{label}: values differ")


def inspect_complete(directory):
    """Check archive contracts without importing production builders or JAX."""
    complete = read_json(directory / "complete.json")
    report = read_json(directory / "report.json")
    resolved = read_json(directory / "resolved.json")
    config = resolved["configuration"]
    name = config["name"]
    require(
        report["status"] == "complete" and report["numerical_complete"] is True,
        "run is not numerically complete",
    )
    require(
        report["diagnostic_only"] is False,
        "diagnostic product cannot pass production validation",
    )
    config_hash = digest(directory / "original.toml")
    require(
        config_hash
        == complete["config_sha256"]
        == report["config_sha256"]
        == config["config_sha256"],
        "configuration hashes differ",
    )
    inventory = complete["output_sha256"]
    require(
        set(inventory) == set(report["output_inventory"]),
        "completion/report inventories differ",
    )
    for filename, expected in inventory.items():
        require(
            Path(filename).name == filename and filename not in {".", ".."},
            "invalid inventory filename",
        )
        require(
            digest(directory / filename) == expected,
            f"output hash mismatch: {filename}",
        )
    require(
        {
            name + ".npz",
            name + ".fits",
            "original.toml",
            "resolved.json",
            "report.json",
        }.issubset(inventory),
        "missing required physical products or provenance",
    )
    with (
        np.load(directory / (name + ".npz"), allow_pickle=False) as data,
        fits.open(directory / (name + ".fits")) as hdus,
    ):
        require(
            all(
                hdu.verify_checksum() == 1 and hdu.verify_datasum() == 1 for hdu in hdus
            ),
            "FITS checksum or datasum failed",
        )
        require(
            str(data["config_sha256"]) == config_hash == hdus[0].header["CFGSHA"],
            "physical product configuration identity differs",
        )
        require(
            bool(data["numerical_complete"]) and hdus[0].header["NUMCOMP"],
            "physical product completion flags differ",
        )
        require(
            not bool(data["diagnostic_only"]) and not hdus[0].header["DIAGONLY"],
            "diagnostic flag set",
        )
        packets = config["execution"]["packets"]
        counts = np.asarray(data["transport_status_count"])
        require(
            counts.shape == (7,)
            and np.issubdtype(counts.dtype, np.integer)
            and np.all(counts >= 0),
            "invalid terminal counts",
        )
        require(
            counts.sum()
            == packets
            == int(data["source_packet_count"])
            == int(data["history_count"]),
            "photon accounting mismatch",
        )
        require(
            counts[0] == 0 and np.all(counts[4:] == 0),
            "incomplete or invalid transport",
        )
        matching(hdus["STATUS"].data["COUNT"], counts, "FITS terminal counts", rtol=0)
        axes = (
            ("sky_x_edges_arcsec", "X_BINS"),
            ("sky_y_edges_arcsec", "Y_BINS"),
            ("energy_edges_kev", "ENERGY_BINS"),
            ("arrival_time_edges_s", "TIME_BINS"),
        )
        for key, hdu_name in axes:
            edges = np.asarray(data[key])
            require(
                np.all(np.isfinite(edges)) and np.all(np.diff(edges) > 0),
                f"invalid {key}",
            )
            matching(hdus[hdu_name].data["LOW"], edges[:-1], key + " low", rtol=0)
            matching(hdus[hdu_name].data["HIGH"], edges[1:], key + " high", rtol=0)
        expected_shape = tuple(
            len(data[key]) - 1
            for key in (
                "arrival_time_edges_s",
                "energy_edges_kev",
                "sky_y_edges_arcsec",
                "sky_x_edges_arcsec",
            )
        )
        total = np.asarray(data["total_fluence"])
        require(total.shape == expected_shape, "fluence cube axis order/shape differs")
        pairs = (
            ("total_fluence", "TOTAL4D"),
            ("first_scatter_fluence", "FIRST4D"),
            ("multiple_scatter_fluence", "MULTI4D"),
            ("event_count", "EVENT4D"),
            ("total_fluence_squared", "HISTQ4D"),
            ("time_image_fluence_squared", "HISTQIMG"),
            ("time_order_fluence_sum", "ORDSUM"),
            ("time_order_fluence_cross", "ORDCROSS"),
        )
        for key, hdu_name in pairs:
            require(np.all(data[key] >= 0), f"negative {key}")
            matching(hdus[hdu_name].data, data[key], key, rtol=0)
        matching(
            total,
            data["first_scatter_fluence"] + data["multiple_scatter_fluence"],
            "first + multiple closure",
        )
        matching(hdus[0].data, total.sum(axis=(0, 1)), "integrated image")
        require(
            hdus[0].header["BUNIT"] == hdus["TOTAL4D"].header["BUNIT"] == "ph cm-2",
            "physical fluence units differ",
        )
        require(
            hdus["EVENT4D"].header["BTYPE"] == "Monte Carlo event count per 4D bin",
            "scored events mislabeled",
        )
        source_fluence = (
            data["source_band_flux"] * np.diff(data["source_time_edges_s"])[:, None]
        ).sum(dtype=float)
        matching(
            data["source_total_fluence"], source_fluence, "intrinsic source fluence"
        )
        matching(
            data["cloud_n_h_cm3"] * data["cloud_radial_bin_width_cm"][:, None, None],
            data["cloud_delta_nh_cm2"],
            "cloud column closure",
            rtol=3e-6,
        )
        matching(
            total.sum(),
            data["binned_weight_observer_fluence"],
            "binned fluence closure",
        )
        require(
            str(data["material_tables"]) == config["materials"]["kind"],
            "material selection differs",
        )
        if config["materials"]["kind"] == "declared_bundle":
            require(
                str(data["material_definition_sha256"]) != "unspecified",
                "declared material identity missing",
            )
        plots = config["plots"]
        if plots["enabled"]:
            for scale in plots["scales"]:
                png = directory / ("intensity_" + scale + ".png")
                require(
                    png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
                    and png.stat().st_size > 64,
                    f"invalid {scale} plot",
                )
        if plots["save_scored_event_map"]:
            start, stop = 0, expected_shape[0]
            if plots["epoch_days"]:
                declared = config["observer"]["arrival"]["edges_days"]
                start, stop = [declared.index(day) for day in plots["epoch_days"]]
            matching(
                np.load(directory / "scored_mc_events.npy", allow_pickle=False),
                data["event_count"][start:stop].sum(axis=(0, 1)),
                "scored event map",
                rtol=0,
            )
        return {
            "status": "passed",
            "packets": packets,
            "shape_time_energy_y_x": list(expected_shape),
            "terminal_counts": counts.tolist(),
            "source_fluence_ph_cm2": float(data["source_total_fluence"]),
            "binned_fluence_ph_cm2": float(data["binned_weight_observer_fluence"]),
            "window_loss_fluence_ph_cm2": float(
                data["unbinned_weight_observer_fluence"]
            ),
            "verified_inventory_files": len(inventory),
            "schema": int(data["output_schema_version"]),
            "backend": resolved["runtime"]["backend"],
            "runtime": {
                key: resolved["runtime"][key]
                for key in (
                    "simulation_python",
                    "simulation_numpy",
                    "simulation_jax",
                    "simulation_git_head",
                    "simulation_git_dirty",
                )
            },
        }


def replace_once(text, old, new):
    require(
        text.count(old) == 1, f"smoke template changed; expected exactly one {old!r}"
    )
    return text.replace(old, new, 1)


def prepare_case(repo, session, name, *, capped=False):
    template = "smoke_legacy" if capped else name
    text = (repo / "configs" / (template + ".toml")).read_text(encoding="utf-8")
    text = text.replace('"../dsh/data/', '"' + (repo / "dsh/data").as_posix() + "/")
    product_dir = session / name / "products"
    text = replace_once(
        text,
        'directory = "../outputs/' + template + '"',
        "directory = " + json.dumps(product_dir.as_posix()),
    )
    text = replace_once(
        text, "enabled = false", "enabled = true\nsave_scored_event_map = true"
    )
    if capped:
        text = replace_once(text, 'name = "smoke_legacy"', 'name = "capped_legacy"')
        text = replace_once(text, "packets = 64", "packets = 128")
        text = replace_once(text, "max_interactions = 32", "max_interactions = 1")
    path = session / name / "simulation.toml"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return path, product_dir


def invoke(config, command, cwd, log):
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    args = [sys.executable, "-m", "dsh.command", command, str(config)]
    print(f"{command}: {config}", flush=True)
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(
            args,
            cwd=cwd,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(f"  exit {result.returncode}; log: {log}", flush=True)
    return result.returncode


def run_smoke(repo, output_root):
    label = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    session = output_root / label
    cwd = session / "working directory with spaces"
    cwd.mkdir(parents=True)
    summary = {
        "validation_version": 1,
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "windows_execution": platform.system() == "Windows",
        "scientific_convergence_claimed": False,
        "session": str(session),
        "cases": {},
        "status": "failed",
    }
    summary["dependencies"] = {
        name: importlib.metadata.version(name)
        for name in ("jax", "numpy", "astropy", "matplotlib")
    }
    try:
        for name in ("smoke_legacy", "smoke_wd01", "capped_legacy"):
            capped = name == "capped_legacy"
            case = {"status": "running"}
            summary["cases"][name] = case
            config, products = prepare_case(repo, session, name, capped=capped)
            case["check_exit"] = invoke(
                config, "check", cwd, config.parent / "check.log"
            )
            require(case["check_exit"] == 0, f"{name}: configuration check failed")
            require(not products.exists(), f"{name}: check created an output directory")
            case["run_exit"] = invoke(config, "run", cwd, config.parent / "run.log")
            if capped:
                report = read_json(products / "report.json")
                require(
                    case["run_exit"] == 3
                    and report["status"] == "incomplete"
                    and report["numerical_complete"] is False,
                    "capped run did not fail clearly",
                )
                require(
                    report["numerical_failure_histories"] > 0,
                    "capped run did not exercise the failure gate",
                )
                require(
                    not (products / "complete.json").exists()
                    and not list(products.glob("*.npz"))
                    and not list(products.glob("*.fits")),
                    "capped run published production products",
                )
                case.update(
                    status="passed",
                    numerical_failure_histories=report["numerical_failure_histories"],
                    production_products_published=False,
                )
            else:
                require(case["run_exit"] == 0, f"{name}: simulation failed")
                case.update(inspect_complete(products))
            print(f"PASS: {name}", flush=True)
        summary["status"] = "passed"
    except (ValidationError, OSError, ValueError, KeyError, TypeError) as error:
        summary["error"] = str(error)
        print(f"FAIL: {error}", file=sys.stderr, flush=True)
    finally:
        summary_path = session / "smoke_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        archive = session / "smoke_report.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as report_zip:
            for path in sorted(session.rglob("*")):
                if path.is_file() and path.suffix in {".json", ".toml", ".log", ".png"}:
                    report_zip.write(path, path.relative_to(session))
        print(f"Report: {archive}", flush=True)
    return 0 if summary["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    repo = args.repo.expanduser().resolve()
    return run_smoke(
        repo,
        (args.output_root or repo / "outputs/windows-smoke").expanduser().resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
