"""Publish complete NumPy archives without exposing a partial replacement."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


def atomic_savez_compressed(path, **arrays):
    """Write beside the destination and replace it only after ZIP finalization.

    A failed or interrupted save leaves an existing completed archive intact.
    Concurrent readers see either the prior archive or the completed new one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
