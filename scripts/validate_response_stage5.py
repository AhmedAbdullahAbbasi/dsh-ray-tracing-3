"""Portable collect/save/load/fold acceptance without cluster submission."""

from __future__ import annotations

import argparse
import hashlib
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect(directory):
    marker = json.loads((directory / "complete.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    resolved = json.loads((directory / "resolved.json").read_text())
    require(
        report["status"] == "complete" and report["numerical_complete"] is True,
        "incomplete response product",
    )
    require(
        hashlib.sha256((directory / "original.toml").read_bytes()).hexdigest()
        == marker["config_sha256"]
        == report["config_sha256"]
        == resolved["configuration"]["config_sha256"],
        "configuration hash mismatch",
    )
    require(
        set(marker["output_sha256"]) == set(report["output_inventory"]),
        "inventory mismatch",
    )
    for name, digest in marker["output_sha256"].items():
        require(Path(name).name == name, "unsafe inventory name")
        require(
            hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest,
            "product hash mismatch: " + name,
        )
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    session = (
        repo
        / "outputs/response-stage5"
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
    )
    session.mkdir(parents=True)
    summary = {
        "stage": 5,
        "platform": platform.system(),
        "python": platform.python_version(),
        "status": "failed",
        "cases": {},
        "scientific_convergence_claimed": False,
    }
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    def invoke(argv, log):
        with log.open("w", encoding="utf-8") as file:
            result = subprocess.run(
                [sys.executable, *map(str, argv)],
                cwd=repo,
                env=env,
                stdout=file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return result.returncode

    try:
        identity = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        summary["git_head"] = identity.stdout.strip()
        summary["tests_exit"] = invoke(
            [
                "-m",
                "unittest",
                "tests.test_response_workflows",
                "tests.test_run_config",
                "tests.test_configuration_usability",
                "tests.test_spectral_histories",
                "tests.test_slurm",
                "-v",
            ],
            session / "tests.log",
        )
        require(summary["tests_exit"] == 0, "focused tests failed")
        archive = None
        for mode in ("collect", "fold", "capped"):
            directory = session / mode
            directory.mkdir()
            template = (
                repo
                / "configs/response"
                / ("fold.toml" if mode == "fold" else "collect.toml")
            )
            text = template.read_text(encoding="utf-8")
            text = text.replace(
                '"../../dsh/data/', '"' + (repo / "dsh/data").as_posix() + "/"
            )
            text = text.replace(
                '"../data/', '"' + (repo / "configs/data").as_posix() + "/"
            )
            output = directory / "products"
            lines = []
            for line in text.splitlines():
                if line.startswith('directory = "../../outputs/'):
                    line = "directory = " + json.dumps(output.as_posix())
                elif mode == "fold" and line.startswith('file = "../../outputs/'):
                    line = "file = " + json.dumps(archive.as_posix())
                if mode == "capped":
                    line = line.replace(
                        "max_interactions = 32", "max_interactions = 1"
                    ).replace("packets = 64", "packets = 128")
                lines.append(line)
            config = directory / "simulation.toml"
            config.write_text("\n".join(lines) + "\n", encoding="utf-8")
            case = summary["cases"][mode] = {}
            case["check_exit"] = invoke(
                ["-m", "dsh.command", "check", config], directory / "check.log"
            )
            require(
                case["check_exit"] == 0 and not output.exists(),
                mode + ": check failed or created products",
            )
            case["run_exit"] = invoke(
                ["-m", "dsh.command", "run", config], directory / "run.log"
            )
            if mode == "capped":
                require(
                    case["run_exit"] == 3
                    and not (output / "complete.json").exists()
                    and not list(output.glob("*.npz")),
                    "capped run published response",
                )
                require(
                    json.loads((output / "report.json").read_text())[
                        "numerical_complete"
                    ]
                    is False,
                    "false numerical completion",
                )
            else:
                require(case["run_exit"] == 0, mode + ": run failed")
                resolved = inspect(output)
                if mode == "collect":
                    from dsh.observer.response import load_spectral_response

                    archive = output / "response_collect.npz"
                    response = load_spectral_response(archive)
                    require(response.n_histories == 64, "incorrect total history count")
                    counts = response.provenance["transport_status_counts"]
                    require(
                        sum(counts) == 64 and sum(counts[4:]) == 0 and counts[0] == 0,
                        "invalid transport accounting",
                    )
                    case.update(
                        n_histories=response.n_histories,
                        samples=len(response.energy_kev),
                        transport_status_counts=counts,
                    )
                else:
                    from dsh.observer.response import (
                        fold_spectral_response,
                        load_spectral_response,
                    )
                    from dsh.sources.histories import load_spectral_history

                    source, _ = load_spectral_history(
                        repo / "configs/data/example_history.json"
                    )
                    with np.load(
                        output / "response_fold.npz", allow_pickle=False
                    ) as data:
                        direct = fold_spectral_response(
                            load_spectral_response(archive),
                            source,
                            data["arrival_time_edges_s"],
                            data["energy_edges_kev"],
                            data["radial_edges_arcsec"],
                        )
                        for name in ("fluence", "standard_error", "covariance"):
                            np.testing.assert_array_equal(
                                data[name], getattr(direct, name)
                            )
                        case.update(
                            fluence_ph_cm2=float(data["fluence"].sum()),
                            shape=list(data["fluence"].shape),
                            photon_grouped_parity=True,
                        )
                case["runtime"] = resolved["runtime"]
            case["status"] = "passed"
            print("PASS: " + mode, flush=True)
        summary["status"] = "passed"
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        ImportError,
        AssertionError,
    ) as error:
        summary["error"] = str(error)
    finally:
        (session / "stage5_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        archive_path = session / "stage5_report.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(session.rglob("*")):
                if path.is_file() and path.suffix in {".json", ".log", ".toml"}:
                    z.write(path, path.relative_to(session))
        print(
            "Stage 5 " + summary["status"] + "; report: " + str(archive_path),
            flush=True,
        )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
