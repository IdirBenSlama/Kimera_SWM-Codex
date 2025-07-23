"""backend.utils package
========================
Utility sub-packages for Kimera.
This file ensures Python recognises `backend.utils` as a proper package so
`backend.utils.threading_utils` and others can be imported reliably.
"""

from importlib import import_module as _import_module
from types import ModuleType as _ModuleType
from typing import Dict as _Dict

_submodules = ("threading_utils", "config", "debug_utils")

_globals: _Dict[str, _ModuleType] = globals()
for _name in _submodules:
    try:
        _module = _import_module(f"{__name__}.{_name}")
        _globals[_name] = _module
    except ModuleNotFoundError:
        # Submodule might be generated later; ignore for now.
        pass

__all__ = list(_submodules) 