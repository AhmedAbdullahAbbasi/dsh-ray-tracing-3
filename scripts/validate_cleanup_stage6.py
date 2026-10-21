"""Run migrated workflow acceptance and synchronized execution profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_evidence(repo):
    results = []
    for relative in (
        "reflex_adapter/SHA256SUMS.json",
        "benchmarks/materials/wd01_reflex_2026-10-16/SHA256SUMS.json",
        "benchmarks/materials/model_comparison_2026-10-16/SHA256SUMS.json",
    ):
        path = repo / relative
        base = path.parent if relative.startswith("reflex_adapter/") else repo
        entries = json.loads(path.read_text(encoding="utf-8"))
        normalized = []
        for name, expected in entries.items():
            file = base / name
            raw = file.read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
            if actual != expected:
                require(
                    file.suffix
                    in {".py", ".md", ".json", ".txt", ".sh", ".ps1", ".log", ".csv"}
                    and hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
                    == expected,
                    "protected evidence changed: " + name,
                )
                normalized.append(name)
        results.append(
            {
                "manifest": relative,
                "entries": len(entries),
                "newline_only_matches": normalized,
            }
        )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    session = (
        repo
        / "outputs/cleanup-stage6"
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
    )
    session.mkdir(parents=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    summary = {
        "stage": 6,
        "platform": platform.system(),
        "python": platform.python_version(),
        "status": "failed",
        "cluster_tested": False,
        "performance_improvement_claimed": False,
        "scientific_convergence_claimed": False,
        "profiles": {},
    }

    def invoke(arguments, log):
        with log.open("w", encoding="utf-8") as file:
            result = subprocess.run(
                [sys.executable, *map(str, arguments)],
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
        summary["frozen_evidence"] = frozen_evidence(repo)
        summary["tests_exit"] = invoke(
            ["-m", "unittest", "tests.test_publication", "tests.test_profiling", "-v"],
            session / "tests.log",
        )
        require(summary["tests_exit"] == 0, "publication/profiling tests failed")
        response_session = None
        for label, script, member in (
            ("runners", "validate_runner_migration.py", "stage3_summary.json"),
            ("response", "validate_response_stage5.py", "stage5_summary.json"),
        ):
            log = session / (label + ".log")
            code = invoke([repo / "scripts" / script, "--repo", repo], log)
            require(code == 0, label + " acceptance failed; inspect " + str(log))
            matches = re.findall(
                r"(?:Report: |report: )(.+\.zip)", log.read_text(encoding="utf-8")
            )
            require(bool(matches), label + " report path missing")
            archive = Path(matches[-1].strip())
            with zipfile.ZipFile(archive) as z:
                child = json.loads(z.read(member))
            require(child["status"] == "passed", label + " report did not pass")
            shutil.copyfile(archive, session / (label + "_report.zip"))
            summary[label] = {
                "status": "passed",
                "git_head": child.get("git_head"),
                "cases": {
                    key: {
                        k: v
                        for k, v in value.items()
                        if k in {"status", "run_exit", "numerical_failure_histories"}
                    }
                    for key, value in child["cases"].items()
                },
            }
            if label == "response":
                response_session = archive.parent
        cases = (
            ("simulation", repo / "configs/smoke_legacy.toml"),
            ("collect", repo / "configs/response/collect.toml"),
            ("fold", repo / "configs/response/fold.toml"),
        )
        for label, template in cases:
            text = template.read_text(encoding="utf-8")
            text = (
                text.replace(
                    '"../../dsh/data/', '"' + (repo / "dsh/data").as_posix() + "/"
                )
                .replace('"../dsh/data/', '"' + (repo / "dsh/data").as_posix() + "/")
                .replace('"../data/', '"' + (repo / "configs/data").as_posix() + "/")
            )
            lines = []
            for line in text.splitlines():
                if line.startswith('directory = "') and "/outputs/" in line:
                    line = "directory = " + json.dumps(
                        (session / ("unused-production-" + label)).as_posix()
                    )
                if label == "fold" and line.startswith('file = "../../outputs/'):
                    line = "file = " + json.dumps(
                        (
                            response_session / "collect/products/response_collect.npz"
                        ).as_posix()
                    )
                if label == "simulation":
                    line = line.replace("packets = 64", "packets = 2048").replace(
                        "chunk_size = 32", "chunk_size = 256"
                    )
                elif label == "collect":
                    line = line.replace("packets = 64", "packets = 512").replace(
                        "chunk_size = 32", "chunk_size = 256"
                    )
                lines.append(line)
            config = session / (label + ".toml")
            config.write_text("\n".join(lines) + "\n", encoding="utf-8")
            directory = session / ("profile-" + label)
            code = invoke(
                [
                    "-m",
                    "dsh.profiling",
                    config,
                    "--output-directory",
                    directory,
                    "--repeats",
                    "2",
                ],
                session / (label + "-profile.log"),
            )
            require(code == 0, label + " profiling failed")
            profile = json.loads(
                (directory / "profile.json").read_text(encoding="utf-8")
            )
            require(
                profile["status"] == "passed"
                and profile["seeded_repeats_exact"] is True,
                label + " profile failed exact replay",
            )
            require(
                not (session / ("unused-production-" + label)).exists(),
                "profiling created a production directory",
            )
            summary["profiles"][label] = {
                key: profile[key]
                for key in (
                    "first_execution_seconds",
                    "repeat_median_seconds",
                    "seeded_repeats_exact",
                    "scientific_result_sha256",
                    "production_products_published",
                )
            }
            print("PASS: " + label + " profile", flush=True)
        summary["status"] = "passed"
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        summary["error"] = str(error)
    finally:
        (session / "stage6_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        archive = session / "stage6_report.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(session.rglob("*")):
                if (
                    path != archive
                    and path.is_file()
                    and path.suffix
                    in {".json", ".log", ".toml", ".txt", ".pstats", ".zip"}
                ):
                    z.write(path, path.relative_to(session))
        print("Stage 6 " + summary["status"] + "; report: " + str(archive), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
