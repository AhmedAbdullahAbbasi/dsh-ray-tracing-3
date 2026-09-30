"""Compatibility import for :mod:`dsh.core.transport.directions`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.core.transport.directions")
sys.modules[__name__] = _implementation
