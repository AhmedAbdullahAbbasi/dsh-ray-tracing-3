"""Run the three public Compton TOML workflows and inspect their saved products."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.output_directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    report = {
        "status": "failed",
        "workflows": {},
        "jax_enable_x64": os.environ.get("JAX_ENABLE_X64", "0"),
    }
    try:
        for name in ("cold_smoke", "collect", "fold"):
            text = (repo / "configs/compton" / (name + ".toml")).read_text(
                encoding="utf-8"
            )
            text = text.replace("../../dsh/data/", (repo / "dsh/data").as_posix() + "/")
            for stem in (
                "cold_compton_smoke",
                "cold_compton_response",
                "cold_compton_fold",
            ):
                text = text.replace("../../outputs/" + stem, (root / stem).as_posix())
            path = root / (name + ".toml")
            path.write_text(text, encoding="utf-8")
            output = (
                root
                / {
                    "cold_smoke": "cold_compton_smoke",
                    "collect": "cold_compton_response",
                    "fold": "cold_compton_fold",
                }[name]
            )
            for command in ("check", "run"):
                log = root / (name + "-" + command + ".log")
                with log.open("w", encoding="utf-8") as stream:
                    result = subprocess.run(
                        [sys.executable, "-m", "dsh.command", command, str(path)],
                        cwd=repo,
                        env=dict(
                            os.environ, PYTHONIOENCODING="utf-8", JAX_PLATFORMS="cpu"
                        ),
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if result.returncode:
                    raise ValueError(name + " " + command + " failed; see " + str(log))
                if command == "check" and output.exists():
                    raise ValueError("check created a product directory")
            marker = json.loads((output / "complete.json").read_text(encoding="utf-8"))
            for filename, digest in marker["output_sha256"].items():
                if (
                    hashlib.sha256((output / filename).read_bytes()).hexdigest()
                    != digest
                ):
                    raise ValueError("completion checksum mismatch: " + filename)
            quality = json.loads((output / "report.json").read_text(encoding="utf-8"))
            if not quality["numerical_complete"]:
                raise ValueError(name + " is incomplete")
            report["workflows"][name] = {"status": "passed", "report": quality}
        import numpy as np
        from astropy.io import fits

        with (
            np.load(
                root / "cold_compton_smoke/cold_compton_smoke.npz", allow_pickle=False
            ) as data,
            fits.open(
                root / "cold_compton_smoke/cold_compton_smoke.fits", checksum=True
            ) as hdus,
        ):
            if (
                int(data["output_schema_version"]) != 9
                or hdus[0].header["OUTSCHEM"] != 9
            ):
                raise ValueError("incorrect simulation schema")
            parts = [
                data[name + "__total_fluence"]
                for name in ("dust_only", "compton_only", "mixed")
            ]
            if any(
                not np.isfinite(value).all() or value.sum() <= 0 for value in parts
            ) or not np.allclose(
                sum(parts), data["total_fluence"], rtol=5e-6, atol=1e-20
            ):
                raise ValueError("component fluence closure failed")
            if float(data["unbinned_weight_observer_fluence"]) != 0:
                raise ValueError(
                    "smoke example unexpectedly lost observer window fluence"
                )
            np.testing.assert_array_equal(
                data["compton_only__total_fluence"], hdus["COMPT4D"].data
            )
            report["component_fluence_ph_cm2"] = {
                name: float(data[name + "__total_fluence"].sum())
                for name in ("dust_only", "compton_only", "mixed")
            }
        with np.load(
            root / "cold_compton_response/cold_compton_response.npz", allow_pickle=False
        ) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if metadata["schema_version"] != 2 or np.any(
                data["energy_kev"] > data["source_energy_kev"] * (1 + 2e-6)
            ):
                raise ValueError("incorrect response energy/schema contract")
            report["response_events"] = int(data["energy_kev"].size)
        with np.load(
            root / "cold_compton_fold/cold_compton_fold.npz", allow_pickle=False
        ) as data:
            parts = [
                data[name + "__fluence"]
                for name in ("dust_only", "compton_only", "mixed")
            ]
            if (
                not np.allclose(sum(parts), data["fluence"], rtol=5e-6, atol=1e-20)
                or not np.isfinite(data["covariance"]).all()
            ):
                raise ValueError("folded component/covariance closure failed")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    (root / "checks.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("Configured Compton workflows: " + report["status"], flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
