"""A failed archive replacement must preserve the completed prior result."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dsh.io.atomic import atomic_savez_compressed


class TestAtomicOutput(unittest.TestCase):
    def test_interrupted_replacement_preserves_prior_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.npz"
            atomic_savez_compressed(path, value=np.array([1, 2, 3]))
            original = path.read_bytes()

            def interrupted(temporary, **_arrays):
                Path(temporary).write_bytes(b"incomplete archive")
                raise OSError("injected interrupted write")

            with (
                patch("dsh.io.atomic.np.savez_compressed", interrupted),
                self.assertRaisesRegex(OSError, "interrupted"),
            ):
                atomic_savez_compressed(path, value=np.array([4, 5, 6]))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_completed_replacement_has_all_arrays_and_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.npz"
            atomic_savez_compressed(
                path, value=np.arange(5), metadata=np.asarray("units")
            )
            atomic_savez_compressed(
                path, value=np.arange(10), metadata=np.asarray("new")
            )
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["value"], np.arange(10))
                self.assertEqual(str(data["metadata"]), "new")
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
