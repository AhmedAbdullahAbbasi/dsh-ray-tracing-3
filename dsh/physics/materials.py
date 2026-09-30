"""Compatibility import for :mod:`dsh.materials.registry`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.materials.registry")
sys.modules[__name__] = _implementation
