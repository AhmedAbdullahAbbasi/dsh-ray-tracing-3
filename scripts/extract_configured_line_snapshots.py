"""Compatibility entry point; implementation lives in :mod:`dsh.products.snapshots`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.products.snapshots")

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
