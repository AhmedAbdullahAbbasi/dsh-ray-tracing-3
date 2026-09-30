"""Compatibility import for :mod:`dsh.materials.absorption`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.materials.absorption")
sys.modules[__name__] = _implementation
