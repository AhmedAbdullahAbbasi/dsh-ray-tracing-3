"""Download pinned author sources and compile the optional gas-only evaluator.

Upstream TBnew code is not redistributed in this repository. Requires a C and
Fortran compiler; portable generated tables need neither compiler at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import urllib.request
from pathlib import Path

SOURCES = {
    "tbnew_2.3.2.tar.gz": (
        "https://pulsar.sternwarte.uni-erlangen.de/wilms/research/tbabs/tbnew_2.3.2.tar.gz",
        "31c0fbd957f6d003c6841fe52cd1fd47cdc12d1d4983dc68d1ee69109ddadb13",
    ),
    "phfit2.f": (
        "https://www.pa.uky.edu/~verner/dima/photo/phfit2.f",
        "8df982535a4a2b952e33ca2ffb6d81600e156d7373fbf547a0c0170c1cecc296",
    ),
}


def build(directory, *, cc="gcc", fc="gfortran", fortran_prefix=None):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in SOURCES.items():
        path = directory / name
        raw = (
            path.read_bytes()
            if path.exists()
            else urllib.request.urlopen(url, timeout=60).read()
        )
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"upstream source checksum mismatch: {name}")
        path.write_bytes(raw)
    with tarfile.open(
        fileobj=io.BytesIO((directory / "tbnew_2.3.2.tar.gz").read_bytes())
    ) as archive:
        for name in ("tbvabs.c", "tbvabs.h", "ophoto.h", "nephoto.h", "fephoto.h"):
            with archive.extractfile("./" + name) as stream:
                (directory / name).write_bytes(stream.read())
    bridge = Path(__file__).with_name("tbnew_gas_bridge.c").resolve()
    flags = ["-B" + str(Path(fortran_prefix).resolve()) + "/"] if fortran_prefix else []
    subprocess.run(
        [
            fc,
            *flags,
            "-O2",
            "-fPIC",
            "-std=legacy",
            "-c",
            str(directory / "phfit2.f"),
            "-o",
            str(directory / "phfit2.o"),
        ],
        check=True,
    )
    library = directory / "libdsh_tbnew_gas.so"
    subprocess.run(
        [
            cc,
            "-O2",
            "-fPIC",
            "-shared",
            str(directory / "tbvabs.c"),
            str(bridge),
            str(directory / "phfit2.o"),
            "-lm",
            "-Wl,--no-undefined",
            "-o",
            str(library),
        ],
        check=True,
    )
    metadata = {
        "schema_version": 1,
        "backend": "author TBnew 2.3.2 gas routines + author Verner PHFIT2 v2",
        "energy_range_kev": [2, 10],
        "sources": {
            name: {"url": url, "sha256": sha} for name, (url, sha) in SOURCES.items()
        },
        "bridge_sha256": hashlib.sha256(
            bridge.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "cc_version": subprocess.check_output(
            [cc, "--version"], text=True
        ).splitlines()[0],
        "fc_version": subprocess.check_output(
            [fc, "--version"], text=True
        ).splitlines()[0],
        "scope": "TBnew H/He/O/Ne/Fe and H2 prescriptions; Verner neutral atoms through Zn for the remaining elements. No XSPEC process or grain shielding in this evaluator.",
    }
    library.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return library


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--cc", default="gcc")
    p.add_argument("--fc", default="gfortran")
    p.add_argument("--fortran-prefix", type=Path)
    a = p.parse_args()
    print(build(a.directory, cc=a.cc, fc=a.fc, fortran_prefix=a.fortran_prefix))


if __name__ == "__main__":
    main()
