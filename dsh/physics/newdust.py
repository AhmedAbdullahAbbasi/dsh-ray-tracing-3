"""Compatibility import for :mod:`dsh.materials.scattering`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.materials.scattering")
sys.modules[__name__] = _implementation
