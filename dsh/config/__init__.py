"""Configuration parsing; execution and products are loaded only on request."""

from importlib import import_module

from .load import load_run_config
from .schema import ResolvedRun

__all__ = [
    "ResolvedRun",
    "load_run_config",
    "build_run",
    "run_configured_simulation",
    "audit_configured_run",
]


def __getattr__(name):
    modules = {
        "build_run": "dsh.build",
        "run_configured_simulation": "dsh.run",
        "audit_configured_run": "dsh.products.audit",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(modules[name]), name)
