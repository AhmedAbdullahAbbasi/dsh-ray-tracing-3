"""Host-side input and runtime identity for configured simulations."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path


def hash_files(paths):
    hashes = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[str(path)] = digest.hexdigest()
    return hashes


def verify_inputs(expected):
    current = hash_files(tuple(Path(name) for name in expected))
    changed = [name for name in expected if current[name] != expected[name]]
    if changed:
        raise ValueError(f"inputs changed during the configured run: {changed}")


def runtime_identity():
    import jax
    import numpy as np

    root = Path(__file__).resolve().parents[2]

    def git(*args):
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except OSError:
            return None

    return {
        "simulation_git_head": git("rev-parse", "HEAD") or "unavailable",
        "simulation_git_dirty": git("status", "--porcelain") != "",
        "simulation_python": platform.python_version(),
        "simulation_numpy": np.__version__,
        "simulation_jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "runtime_source_sha256": hash_files(
            tuple(sorted((root / "dsh").rglob("*.py")))
        ),
    }
