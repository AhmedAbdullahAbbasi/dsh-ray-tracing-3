"""Compatibility import for :mod:`dsh.core.observer.binning`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.core.observer.binning")
sys.modules[__name__] = _implementation
