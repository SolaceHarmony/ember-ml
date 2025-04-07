"""
Operations module.

This module dynamically aliases functions from the active backend
(NumPy, PyTorch, MLX) to provide a consistent `ops.*` interface.
It handles backend switching by updating these aliases.
"""

import importlib
import sys
import os
from typing import List, Optional, Callable, Any

# Import backend control functions (including the original set_backend)
from ember_ml.backend import (
    get_backend,
    set_backend as original_set_backend,
    get_backend_module,
    auto_select_backend
)

# Master list of all functions expected to be aliased by ember_ml.ops
# NOTE: Ensure this list is consistent with functions exported by backend __init__.py files
_MASTER_OPS_LIST = [
    # Math
    'add', 'subtract', 'multiply', 'divide', 'matmul', 'dot', 'mean', 'sum', 'max', 'min',
    'exp', 'log', 'log10', 'log2', 'pow', 'sqrt', 'square', 'abs', 'sign', 'sin', 'cos', 'tan',
    'sinh', 'cosh', 'clip', 'var', 'negative', 'mod', 'floor_divide', 'sort', 'gradient', 'cumsum', 'eigh',
    'pi', 'power',
    # Comparison
    'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal', 'logical_and', 'logical_or',
    'logical_not', 'logical_xor', 'allclose', 'isclose', 'all', 'where', 'isnan',
    # Device
    'to_device', 'get_device', 'get_available_devices', 'memory_usage', 'memory_info', 'synchronize',
    'set_default_device', 'get_default_device', 'is_available',
    # IO
    'save', 'load',
    # Loss
    'mean_squared_error', 'mean_absolute_error', 'binary_crossentropy', 'categorical_crossentropy',
    'sparse_categorical_crossentropy', 'huber_loss', 'log_cosh_loss',
    # Vector
    'normalize_vector', 'compute_energy_stability', 'compute_interference_strength', 'compute_phase_coherence',
    'partial_interference', 'euclidean_distance', 'cosine_similarity', 'exponential_decay', 'fft', 'ifft',
    'fft2', 'ifft2', 'fftn', 'ifftn', 'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn', 'gaussian',
    # Linear Algebra
    # 'solve', 'inv', 'svd', 'eig', 'eigvals', 'det', 'norm', 'qr', 'cholesky', 'lstsq', 'diag', 'diagonal',
    # Feature Ops
    'pca', 'transform', 'inverse_transform', 'standardize', 'normalize',
    # Activations (These are typically handled by nn.modules.activations.ops, but backends might expose them)
    'relu', 'sigmoid', 'tanh', 'softmax', 'softplus',
    # Stats Ops (Add specific functions as needed and implemented in backends)
    'median', 'std', 'percentile',
]

# Placeholder for functions that will be dynamically loaded
# Initialize globals to None to indicate they aren't loaded yet
for _op_name in _MASTER_OPS_LIST:
    # Check if the name is already defined (like built-ins: max, min, sum, pow, abs, all)
    if _op_name not in globals():
        globals()[_op_name] = None
    # For built-ins, we will overwrite them later with the backend version, which is intended.

# Keep track of the currently aliased backend
_aliased_backend: Optional[str] = None

def _update_ops_aliases():
    """Dynamically updates the ops module's namespace with backend functions."""
    global _aliased_backend
    backend_name = get_backend()

    # Only update if the backend has changed
    if backend_name == _aliased_backend:
        return

    backend_module = get_backend_module() # Get the main backend module
    # Attempt to force reload the backend module to avoid stale cache issues
    try:
        backend_module = importlib.reload(backend_module)
    except Exception as e:
        print(f"Warning: Failed to reload backend module {backend_module.__name__}: {e}")
    current_ops_module = sys.modules[__name__] # Get the 'ember_ml.ops' module object

    missing_ops = []
    successful_aliases = 0
    for func_name in _MASTER_OPS_LIST:
        try:
            backend_function = getattr(backend_module, func_name)
            setattr(current_ops_module, func_name, backend_function)
            # Also update globals() so functions defined *within this file* can use the aliases
            # This might be less necessary now but kept for safety.
            globals()[func_name] = backend_function
            successful_aliases += 1
        except AttributeError:
            setattr(current_ops_module, func_name, None) # Set alias to None
            globals()[func_name] = None
            missing_ops.append(func_name)

    if missing_ops:
        print(f"Warning: Backend '{backend_name}' does not implement the following ops: {', '.join(missing_ops)}")
    _aliased_backend = backend_name # Mark this backend as aliased

# --- Define set_backend for this module to trigger alias updates ---
def set_backend(backend: str):
    """Sets the backend and updates ops aliases."""
    # Check if backend is actually changing to avoid redundant updates
    current_backend_before_change = get_backend()
    if backend == current_backend_before_change:
        return
    original_set_backend(backend) # Call the original function in backend/__init__.py
    _update_ops_aliases() # Update aliases in this module
    # Explicitly trigger updates in other aliasing modules
    try:
        from ember_ml.ops.stats import _update_stats_aliases
        _update_stats_aliases()
    except ImportError:
        print("Warning: Could not import or update stats aliases.")
    try:
        from ember_ml.nn.modules.activations import _update_activation_aliases
        _update_activation_aliases()
    except ImportError:
        print("Warning: Could not import or update activation aliases.")
    # Explicitly trigger update in activations module
    from ember_ml.nn.modules.activations import _update_activation_aliases as update_activations
    update_activations()

# --- Initial alias setup ---
# Ensure backend is determined and aliases populated on first import
_init_backend_name = get_backend() # This call triggers auto-selection if needed
_update_ops_aliases() # Populate aliases based on the determined backend

# --- Define __all__ ---
# Includes backend controls aliased here and the master list of ops
__all__ = [
    'set_backend', 'get_backend', 'auto_select_backend', # Expose backend controls via ops
] + _MASTER_OPS_LIST # Note: _MASTER_OPS_LIST no longer contains linearalg functions