"""
Statistical operations module.

This module dynamically aliases functions from the active backend
(NumPy, PyTorch, MLX) to provide a consistent `stats.*` interface.
It handles backend switching by updating these aliases.
"""

import importlib
import sys
from typing import Optional

# Import backend control functions
from ember_ml.backend import (
    get_backend,
    set_backend as original_set_backend, # Import original set_backend
    get_backend_module
)

# Master list of statistical functions expected to be aliased
_STATS_OPS_LIST = [
    'mean',
    'var',
    'median',
    'std',
    'percentile',
    'max',
    'min',
    'sum',
    'cumsum',
    'argmax',
    'sort',
    'argsort',
    'gaussian',
]

# Placeholder for functions that will be dynamically loaded
for _op_name in _STATS_OPS_LIST:
    if _op_name not in globals(): # Avoid overwriting built-ins initially
        globals()[_op_name] = None
    # Built-ins like max, min, sum, sort will be overwritten
def get_stats_module():
    """Imports the stats functions from the active backend module."""
    # Get the backend base module name (e.g., 'ember_ml.backend.numpy')
    backend_base_module_name = get_backend_module().__name__
    print(f"DEBUG: get_stats_module - Backend base module name: {backend_base_module_name}")
    
    # Construct the stats module name (e.g., 'ember_ml.backend.numpy.stats')
    module_name = backend_base_module_name + '.stats'
    print(f"DEBUG: get_stats_module - Constructed module name: {module_name}")
    
    try:
        # Import the stats module from the backend
        module = importlib.import_module(module_name)
        print(f"DEBUG: get_stats_module - Successfully imported module: {module.__name__}")
        return module
    except ImportError as e:
        print(f"DEBUG: get_stats_module - Failed to import module {module_name}: {e}")
        return None
    except AttributeError as e:
        # This catches the "module has no attribute 'ops'" error
        print(f"DEBUG: get_stats_module - AttributeError: {e}")
        return None

# Keep track of the currently aliased backend for stats
_aliased_backend_stats: Optional[str] = None

def _update_stats_aliases():
    """Dynamically updates this module's namespace with backend stats functions."""
    global _aliased_backend_stats
    backend_name = get_backend()

    if backend_name == _aliased_backend_stats:
        return

    backend_module = get_stats_module()
    # Attempt to force reload the specific backend stats module
    try:
        backend_module = importlib.reload(backend_module)
    except Exception as e:
        print(f"Warning: Failed to reload backend stats module {getattr(backend_module, '__name__', 'N/A')}: {e}")
    current_module = sys.modules[__name__]
    missing_ops = []

    for func_name in _STATS_OPS_LIST:
        try:
            backend_function = getattr(backend_module, func_name)
            setattr(current_module, func_name, backend_function)
            globals()[func_name] = backend_function # Update globals too
        except AttributeError:
            setattr(current_module, func_name, None)
            globals()[func_name] = None
            missing_ops.append(func_name)

    if missing_ops:
        print(f"Warning: Backend '{backend_name}' does not implement the following stats ops: {', '.join(missing_ops)}")
    _aliased_backend_stats = backend_name

# --- Define set_backend for this module to trigger alias updates ---
# This ensures that if the backend is changed elsewhere, this module's aliases
# are updated *when this module is next imported or used*.
# However, to ensure immediate update upon external set_backend call,
# the main ops.__init__.py set_backend needs to trigger this.
# Reverting to the simpler model where ops.set_backend handles everything.
# We will remove this custom set_backend and rely on ops.__init__.py

# --- Initial alias setup ---
_init_backend_name_stats = get_backend() # Ensure backend is determined
_update_stats_aliases() # Populate aliases on first import

# --- Define __all__ ---
__all__ = _STATS_OPS_LIST # type: ignore