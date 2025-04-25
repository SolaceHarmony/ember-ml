"""
Bitwise operations module for the Ember ML frontend.

This module dynamically aliases functions from the active backend's
bitwise implementation (e.g., ember_ml.backend.mlx.bitwise)
to provide a consistent `ops.bitwise.*` interface.
Follows the pattern of ember_ml.ops.linearalg.
"""

import importlib
import sys
import os
from typing import List, Optional, Callable, Any

# Import backend control functions
from ember_ml.backend import get_backend, get_backend_module

# Master list of bitwise operations expected to be aliased
# This should match the __all__ list in the backend bitwise modules
_BITWISE_OPS_LIST = [
    # Basic Ops
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    # Shift Ops
    "left_shift",
    "right_shift",
    "rotate_left",
    "rotate_right",
    # Bit Ops
    "count_ones",
    "count_zeros",
    "get_bit",
    "set_bit",
    "toggle_bit",
    # Wave Ops
    "binary_wave_interference",
    "binary_wave_propagate",
    "create_duty_cycle",
    "generate_blocky_sin",
]

def get_bitwise_module():
    """Imports the bitwise operations from the active backend module."""
    # Construct the path to the backend's bitwise submodule
    backend_name = get_backend()
    module_name = get_backend_module().__name__ + '.bitwise'
    try:
        module = importlib.import_module(module_name)
        # Optionally reload for development? Match linearalg if it does.
        # module = importlib.reload(module)
        return module
    except ModuleNotFoundError:
        # Handle cases where the bitwise module might not exist for a backend
        print(f"Warning: Bitwise operations module not found for backend '{backend_name}'.")
        return None


# Placeholder initialization
for _op_name in _BITWISE_OPS_LIST:
    if _op_name not in globals():
        globals()[_op_name] = None

# Keep track if aliases have been set for the current backend
_aliased_backend_bitwise: Optional[str] = None

def _update_bitwise_aliases():
    """Dynamically updates this module's namespace with backend bitwise functions."""
    global _aliased_backend_bitwise
    backend_name = get_backend()

    # Avoid re-aliasing if backend hasn't changed
    if backend_name == _aliased_backend_bitwise:
        return

    backend_module = get_bitwise_module()
    current_module = sys.modules[__name__]
    missing_ops = []

    if backend_module is None:
         # If the backend module doesn't exist, set all ops to None
         for func_name in _BITWISE_OPS_LIST:
             setattr(current_module, func_name, None)
             globals()[func_name] = None
         print(f"Warning: No bitwise operations available for backend '{backend_name}'.")
    else:
        for func_name in _BITWISE_OPS_LIST:
            try:
                backend_function = getattr(backend_module, func_name)
                setattr(current_module, func_name, backend_function)
                globals()[func_name] = backend_function
            except AttributeError:
                setattr(current_module, func_name, None)
                globals()[func_name] = None
                missing_ops.append(func_name)

        if missing_ops:
            # Suppress warning here as ops/__init__ might also warn,
            # or provide a specific warning about missing bitwise ops.
            # print(f"Warning: Missing bitwise operations for backend '{backend_name}': {missing_ops}")
            pass

    _aliased_backend_bitwise = backend_name

# --- Initial alias setup ---
# Populate aliases when this module is first imported.
# Relies on the backend having been determined by prior imports (e.g., in ops/__init__).
_update_bitwise_aliases()

# --- Define __all__ ---
# Expose only the functions listed, not internal helpers
__all__ = _BITWISE_OPS_LIST