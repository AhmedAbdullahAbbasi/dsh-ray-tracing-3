"""Compatibility entry point; implementation lives in :mod:`dsh.products.legacy_snapshots`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.products.legacy_snapshots")

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
