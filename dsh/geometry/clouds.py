"""Compatibility import for :mod:`dsh.core.geometry.clouds`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.core.geometry.clouds")
sys.modules[__name__] = _implementation
