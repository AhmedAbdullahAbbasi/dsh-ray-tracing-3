"""Compatibility import for :mod:`dsh.materials.grids`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.materials.grids")
sys.modules[__name__] = _implementation
