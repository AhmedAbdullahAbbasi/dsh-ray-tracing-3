"""Compatibility import for :mod:`dsh.core.observer.scoring`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.core.observer.scoring")
sys.modules[__name__] = _implementation
