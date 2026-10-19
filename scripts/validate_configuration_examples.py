"""Exercise Stage 2 configuration contracts and all seven runnable examples."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .validate_configured_smoke import (
        ValidationError,
        inspect_complete,
        invoke,
        replace_once,
        require,
    )
else:
    from validate_configured_smoke import (
        ValidationError,
        inspect_complete,
        invoke,
        replace_once,
        require,
    )

EXAMPLES = (
    "smoke_legacy",
    "smoke_wd01",
    "example_band_powerlaw",
    "example_exponential",
    "example_history",
    "example_flux_table",
    "example_cloud_fits",
)


def prepare_example(repo, session, name):
    """Copy a committed example, retaining its physics and selecting fresh outputs."""
    text = (repo / "configs" / (name + ".toml")).read_text(encoding="utf-8")
    text = text.replace('"../dsh/data/', '"' + (repo / "dsh/data").as_posix() + "/")
    text = text.replace('"data/', '"' + (repo / "configs/data").as_posix() + "/")
    products = session / name / "products"
    text = replace_once(
        text,
        'directory = "../outputs/' + name + '"',
        "directory = " + json.dumps(products.as_posix()),
    )
    if "enabled = false" in text:
        text = replace_once(
            text, "enabled = false", "enabled = true\nsave_scored_event_map = true"
        )
    path = session / name / "simulation.toml"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return path, products


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
        "stage": 2,
        "platform": platform.system(),
        "python": platform.python_version(),
        "windows_execution": platform.system() == "Windows",
        "scientific_convergence_claimed": False,
        "session": str(session),
        "status": "failed",
        "cases": {},
    }
    try:
        summary["dependencies"] = {
            name: importlib.metadata.version(name)
            for name in ("jax", "numpy", "astropy", "matplotlib")
        }
        tests = [
            "tests.test_run_config",
            "tests.test_configuration_usability",
            "tests.test_configured_inputs",
            "tests.test_configuration_examples",
            "tests.test_configured_run",
            "tests.test_smoke_inspection",
        ]
        print("Running focused configuration and product tests...", flush=True)
        with (session / "tests.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-m", "unittest", *tests, "-v"],
                cwd=repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        summary["tests_exit"] = result.returncode
        require(result.returncode == 0, "focused tests failed; see tests.log")
        for name in EXAMPLES:
            case = {"status": "running"}
            summary["cases"][name] = case
            config, products = prepare_example(repo, session, name)
            case["check_exit"] = invoke(
                config, "check", cwd, config.parent / "check.log"
            )
            require(case["check_exit"] == 0, f"{name}: check failed")
            require(not products.exists(), f"{name}: check created outputs")
            case["run_exit"] = invoke(config, "run", cwd, config.parent / "run.log")
            require(case["run_exit"] == 0, f"{name}: run failed")
            case.update(inspect_complete(products))
            print(f"PASS: {name}", flush=True)
        summary["status"] = "passed"
    except (
        ValidationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        importlib.metadata.PackageNotFoundError,
    ) as error:
        summary["error"] = str(error)
        print(f"FAIL: {error}", file=sys.stderr, flush=True)
    finally:
        (session / "stage2_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        archive = session / "stage2_report.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as report:
            for path in sorted(session.rglob("*")):
                if path.is_file() and path.suffix in {".json", ".toml", ".log", ".png"}:
                    report.write(path, path.relative_to(session))
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
    return run_validation(
        repo,
        (args.output_root or repo / "outputs/configuration-stage2")
        .expanduser()
        .resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
