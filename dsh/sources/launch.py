"""Compatibility import for :mod:`dsh.core.launch`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.core.launch")
sys.modules[__name__] = _implementation
