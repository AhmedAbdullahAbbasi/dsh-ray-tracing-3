"""Compatibility import for :mod:`dsh.sources.format`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.sources.format")
sys.modules[__name__] = _implementation
