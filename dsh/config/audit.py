"""Compatibility import for :mod:`dsh.products.audit`."""

import sys
from importlib import import_module

_implementation = import_module("dsh.products.audit")
sys.modules[__name__] = _implementation
