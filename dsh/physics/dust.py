"""Compatibility import for :mod:`dsh.materials.table`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.materials.table")
sys.modules[__name__] = _implementation
