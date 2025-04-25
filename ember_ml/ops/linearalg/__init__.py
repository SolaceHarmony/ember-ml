"""
Linear Algebra operations module.

This module dynamically aliases functions from the active backend
(NumPy, PyTorch, MLX) upon import to provide a consistent `ops.linearalg.*` interface.
"""

import importlib
import sys
import os
import numpy as np
from typing import List, Optional, Callable, Any, Tuple, Union, Dict

# Import backend control functions
from ember_ml.backend import get_backend, get_backend_module

# Master list of linear algebra functions expected to be aliased
_LINEARALG_OPS_LIST = [
    'solve', 'inv', 'svd', 'eig', 'eigh', 'eigvals', 'det', 'norm', 'qr',
    'cholesky', 'lstsq', 'diag', 'diagonal', 'orthogonal',
    # High-Precision Computing (HPC) operations
    'qr_128', 'orthogonalize_nonsquare', '_add_limb_precision', 'HPC16x8'
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

# --- Fallback implementations ---
def _fallback_orthogonal(shape, gain=1.0, dtype=None, device=None):
    """
    Fallback implementation of orthogonal matrix initialization.
    
    This is used when the orthogonal function is not available in the backend.
    It uses NumPy for the QR decomposition and then converts back to the appropriate tensor type.
    
    Args:
        shape: Shape of the tensor to initialize
        gain: Multiplicative factor to apply to the orthogonal matrix
        dtype: Data type of the tensor (ignored in fallback)
        device: Device to place the tensor on (ignored in fallback)
        
    Returns:
        A random orthogonal matrix of the specified shape
    """
    # Convert shape to tuple of integers if it's not already
    if not isinstance(shape, (list, tuple)):
        try:
            shape = tuple(int(dim) for dim in shape)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid shape: {shape}")
    
    if len(shape) < 2:
        raise ValueError("Shape must have at least 2 dimensions")
    
    # Flatten all dimensions after the first
    rows, cols = shape[0], np.prod(shape[1:])
    flat_shape = (rows, cols)
    
    # Generate a random matrix with NumPy
    np_rand = np.random.normal(0.0, 1.0, flat_shape).astype(np.float32)
    
    # Use NumPy's QR decomposition
    q, r = np.linalg.qr(np_rand)
    
    # Make Q uniform by multiplying by sign of diagonal of R
    d = np.diag(r)
    ph = np.sign(d)
    q = q * ph
    
    # Apply gain and reshape
    q = q * gain
    
    # Reshape to the desired shape
    if len(shape) > 2:
        q = q.reshape(shape)
    
    # Import tensor module to convert back to the appropriate tensor type
    from ember_ml.nn import tensor
    return tensor.convert_to_tensor(q)

# --- Initial alias setup ---
# Populate aliases when this module is first imported.
# Relies on the backend having been determined by prior imports.
_update_linearalg_aliases()

# Add fallback for orthogonal if it's not available
if orthogonal is None:
    orthogonal = _fallback_orthogonal

# --- Define __all__ ---
__all__ = _LINEARALG_OPS_LIST # type: ignore
