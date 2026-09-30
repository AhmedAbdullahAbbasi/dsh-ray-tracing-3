"""Host-side file and checkout provenance helpers."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def _git_value(*args) -> str | None:
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
