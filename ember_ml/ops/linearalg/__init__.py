"""
Linear Algebra operations module.

This module dynamically aliases functions from the active backend
(NumPy, PyTorch, MLX) upon import to provide a consistent `ops.linearalg.*` interface.
"""

import importlib
import sys
import os
from typing import List, Optional, Callable, Any

# Import backend control functions
from ember_ml.backend import get_backend, get_backend_module

# Master list of linear algebra functions expected to be aliased
_LINEARALG_OPS_LIST = [
    'solve', 'inv', 'svd', 'eig', 'eigh', 'eigvals', 'det', 'norm', 'qr',
    'cholesky', 'lstsq', 'diag', 'diagonal', 'orthogonal',
    # HPC operations
    'HPC16x8'
]
def get_linearalg_module():
    """Imports the activation functions from the active backend module."""
    # This function is not used in this module but can be used for testing purposes
    # or to ensure that the backend module is imported correctly.
    # Reload the backend module to ensure the latest version is use
    backend_name = get_backend()
    module_name = get_backend_module().__name__ + '.linearalg'
    module = importlib.import_module(module_name)
    return module

# Placeholder initialization
for _op_name in _LINEARALG_OPS_LIST:
    if _op_name not in globals():
        globals()[_op_name] = None

# Keep track if aliases have been set for the current backend
_aliased_backend_linearalg: Optional[str] = None

def _update_linearalg_aliases():
    """Dynamically updates this module's namespace with backend linearalg functions."""
    global _aliased_backend_linearalg
    backend_name = get_backend()

    # Avoid re-aliasing if backend hasn't changed since last update for this module
    if backend_name == _aliased_backend_linearalg:
        return

    backend_module = get_linearalg_module()
    current_module = sys.modules[__name__]
    missing_ops = []

    for func_name in _LINEARALG_OPS_LIST:
        try:
            backend_function = getattr(backend_module, func_name)
            setattr(current_module, func_name, backend_function)
            globals()[func_name] = backend_function
        except AttributeError:
            setattr(current_module, func_name, None)
            globals()[func_name] = None
            missing_ops.append(func_name)

    if missing_ops:
        # Suppress warning here as ops/__init__ might also warn
        # print(f"Warning: Backend '{backend_name}' does not implement the following linearalg ops: {', '.join(missing_ops)}")
        pass
    _aliased_backend_linearalg = backend_name

# --- Initial alias setup ---
# Populate aliases when this module is first imported.
# Relies on the backend having been determined by prior imports.
_update_linearalg_aliases()

# --- Define __all__ ---
__all__ = _LINEARALG_OPS_LIST # type: ignore
