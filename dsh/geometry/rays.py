"""Compatibility import for :mod:`dsh.core.geometry.rays`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.core.geometry.rays")
sys.modules[__name__] = _implementation
