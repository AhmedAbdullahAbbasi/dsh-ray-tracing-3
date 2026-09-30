"""Compatibility import for :mod:`dsh.core.transport.kernel`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.core.transport.kernel")
sys.modules[__name__] = _implementation
