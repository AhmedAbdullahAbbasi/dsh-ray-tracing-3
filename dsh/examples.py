"""Compatibility import for :mod:`dsh.scenes.examples`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.scenes.examples")
sys.modules[__name__] = _implementation
