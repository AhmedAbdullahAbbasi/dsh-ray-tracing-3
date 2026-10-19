"""Portable Stage 3 acceptance: configured, CLI and Python-settings runners."""

from __future__ import annotations

import argparse
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

if __package__:
    from .validate_configured_smoke import (
        inspect_complete,
        read_json,
        replace_once,
        require,
    )
else:
    from validate_configured_smoke import (
        inspect_complete,
        read_json,
        replace_once,
        require,
    )


def invoke(arguments, cwd, log):
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    print("Python: " + " ".join(map(str, arguments)), flush=True)
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.run(
            [sys.executable, *map(str, arguments)],
            cwd=cwd,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(f"  exit {process.returncode}; log: {log}", flush=True)
    return process.returncode


def configured_case(repo, session, filename):
    name = "configured_" + filename
    directory = session / name
    directory.mkdir()
    text = (repo / "configs/stage3" / (filename + ".toml")).read_text(encoding="utf-8")
    text = text.replace('"../../dsh/data/', '"' + (repo / "dsh/data").as_posix() + "/")
    text = text.replace('"../data/', '"' + (repo / "configs/data").as_posix() + "/")
    products = directory / "products"
    declared = "../../outputs/stage3_" + filename
    text = replace_once(
        text,
        'directory = "' + declared + '"',
        "directory = " + json.dumps(products.as_posix()),
    )
    config = directory / "simulation.toml"
    config.write_text(text, encoding="utf-8")
    return (
        ["-m", "dsh.command", "run", config],
        ["-m", "dsh.command", "check", config],
        products,
    )


def cli_case(repo, session, *, capped=False):
    name = "capped_cli" if capped else "legacy_cli"
    directory = session / name
    directory.mkdir()
    output = directory / "npz folder/result.npz"
    arguments = [
        "-m",
        "scripts.run_dsh_v1",
        "--cloud-fits",
        repo / "configs/data/example_cloud.fits",
        "--packets",
        "64",
        "--chunk-size",
        "32",
        "--max-interactions",
        "32",
        "--output",
        output,
        "--fits-output",
        directory / "fits folder/cube.fits",
        "--device",
        "cpu",
    ]
    if capped:
        # The built-in four-cloud scene is the established nontrivial cap control.
        arguments = [
            "-m",
            "dsh.cli",
            "--packets",
            "128",
            "--chunk-size",
            "32",
            "--max-interactions",
            "1",
            "--output",
            output,
            "--device",
            "cpu",
        ]
    else:
        arguments += ["--materials", "2-10", "--source-spectrum", "hard-state-powerlaw"]
    # scripts/ is a checkout utility, so invoke that shim by its absolute path.
    if not capped:
        arguments[:2] = [str(repo / "scripts/run_dsh_v1.py")]
    return arguments, [*arguments, "--check"], output.parent / "result.run"


def root_case(repo, session, *, capped=False):
    name = "capped_root" if capped else "legacy_root"
    directory = session / name
    directory.mkdir()
    values = {
        "SOURCE_DISTANCE_KPC": 10.0,
        "CLOUD_FITS_PATH": None,
        "GRID_X_ARCSEC": (-100.0, 110.0, 20.0),
        "GRID_Y_ARCSEC": (-100.0, 110.0, 20.0),
        "GRID_Z_KPC": (2.0, 6.0, 1.0),
        "DISC_CENTER_X_ARCSEC": 0.0,
        "DISC_CENTER_Y_ARCSEC": 0.0,
        "DISC_CENTER_Z_KPC": 3.5,
        "DISC_RADIUS_ARCSEC": 80.0,
        "DISC_THICKNESS_KPC": 2.0,
        "DISC_N_H_CM3": 1.0,
        "SOURCE_MODEL": "custom",
        "ENERGY_EDGES_KEV": None,
        "CUSTOM_TIME_EDGES_DAYS": (0.0, 1 / 24),
        "CUSTOM_BAND_FLUXES": ((0.02, 0.012, 0.006),),
        "DEVICE": "cpu",
        "PACKETS": 128 if capped else 64,
        "CHUNK_SIZE": 32,
        "MAX_INTERACTIONS": 1 if capped else 32,
        "SEED": 2026,
        "LAUNCH_SAMPLER": "radial",
        "ARRIVAL_DAYS": 2.5,
        "TIME_BIN_DAYS": 1.0,
        "MATERIAL_MODEL": "default",
        "IMAGE_EPOCH_DAYS": (0.2, 2.2),
        "IMAGE_ENERGY_KEV": 4.9,
        "IMAGE_FOV_ARCSEC": 120.0,
        "IMAGE_SCALES": ("log", "linear"),
        "IMAGE_DPI": 100,
        "PROGRESS_EVERY_CHUNKS": 5,
        "OUTPUT_PNG": str(directory / "images/image.png"),
        "OUTPUT_COUNT_PNG": str(directory / "counts/events.png"),
        "OUTPUT_COUNT_NPY": str(directory / "maps/events.npy"),
        "PROGRESS_LOG": str(directory / "logs/progress.log"),
    }
    settings = directory / "legacy_settings.py"
    settings.write_text(
        "\n".join(key + " = " + repr(value) for key, value in values.items()) + "\n",
        encoding="utf-8",
    )
    arguments = [str(repo / "run_simulation.py"), "--legacy-config", settings]
    return arguments, [*arguments, "--check"], directory / "images/image.run"


def run_validation(repo, output_root):
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
        "stage": 3,
        "platform": platform.system(),
        "python": platform.python_version(),
        "windows_execution": platform.system() == "Windows",
        "scientific_convergence_claimed": False,
        "status": "failed",
        "session": str(session),
        "cases": {},
    }
    try:
        summary["dependencies"] = {
            name: importlib.metadata.version(name)
            for name in ("jax", "numpy", "astropy", "matplotlib")
        }
        modules = [
            "tests.test_run_config",
            "tests.test_configuration_usability",
            "tests.test_configured_inputs",
            "tests.test_configuration_examples",
            "tests.test_configured_run",
            "tests.test_smoke_inspection",
            "tests.test_flare_snapshots",
            "tests.test_legacy_runners",
        ]
        summary["tests_exit"] = invoke(
            ["-m", "unittest", *modules, "-v"], repo, session / "tests.log"
        )
        require(summary["tests_exit"] == 0, "focused tests failed; inspect tests.log")
        cases = [
            ("configured_disc", lambda: configured_case(repo, session, "disc"), False),
            (
                "configured_representative_flux",
                lambda: configured_case(repo, session, "representative_flux"),
                False,
            ),
            ("legacy_cli", lambda: cli_case(repo, session), False),
            ("legacy_root", lambda: root_case(repo, session), False),
            ("capped_cli", lambda: cli_case(repo, session, capped=True), True),
            ("capped_root", lambda: root_case(repo, session, capped=True), True),
        ]
        for name, prepare, capped in cases:
            arguments, check, products = prepare()
            case = summary["cases"][name] = {"status": "running"}
            logs = session / name
            case["check_exit"] = invoke(check, cwd, logs / "check.log")
            require(case["check_exit"] == 0, name + ": check failed")
            require(not products.exists(), name + ": check created output directory")
            case["run_exit"] = invoke(arguments, cwd, logs / "run.log")
            if capped:
                report = read_json(products / "report.json")
                require(
                    case["run_exit"] == 3
                    and report["numerical_complete"] is False
                    and report["numerical_failure_histories"] > 0,
                    name + ": numerical cap gate did not reject run",
                )
                require(
                    not (products / "complete.json").exists(),
                    name + ": incomplete run has a completion marker",
                )
                require(
                    not list(logs.rglob("*.npz"))
                    and not list(logs.rglob("*.fits"))
                    and not list(logs.rglob("*.png"))
                    and not list(logs.rglob("*.npy")),
                    name + ": incomplete run published physical products or plots",
                )
                case.update(
                    status="passed",
                    numerical_failure_histories=report["numerical_failure_histories"],
                    production_products_published=False,
                )
            else:
                require(case["run_exit"] == 0, name + ": run failed")
                case.update(inspect_complete(products))
            print("PASS: " + name, flush=True)
        summary["status"] = "passed"
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        summary["error"] = str(error)
        for case in summary["cases"].values():
            if case["status"] == "running":
                case["status"] = "failed"
        print("FAIL: " + str(error), file=sys.stderr, flush=True)
    finally:
        (session / "stage3_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        archive = session / "stage3_report.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as report_zip:
            for path in sorted(session.rglob("*")):
                if path.is_file() and path.suffix in {
                    ".json",
                    ".toml",
                    ".log",
                    ".png",
                    ".py",
                }:
                    report_zip.write(path, path.relative_to(session))
        print("Report: " + str(archive), flush=True)
    return 0 if summary["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    repo = args.repo.expanduser().resolve()
    return run_validation(
        repo,
        (args.output_root or repo / "outputs/runner-stage3").expanduser().resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
