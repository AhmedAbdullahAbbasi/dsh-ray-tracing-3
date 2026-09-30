"""Compatibility import for :mod:`dsh.core.geometry.coordinates`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.core.geometry.coordinates")
sys.modules[__name__] = _implementation
