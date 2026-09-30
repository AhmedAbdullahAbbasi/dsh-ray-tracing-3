"""Compatibility import for :mod:`dsh.materials.recipes.rg_drude`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.materials.recipes.rg_drude")
sys.modules[__name__] = _implementation
