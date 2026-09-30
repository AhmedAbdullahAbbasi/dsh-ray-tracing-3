"""Compatibility import for :mod:`dsh.core.pipeline`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.core.pipeline")
sys.modules[__name__] = _implementation
