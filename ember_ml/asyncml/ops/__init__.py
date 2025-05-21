"""
Asynchronous operations module.

This module dynamically aliases asynchronous wrapper functions for operations
from the active backend (NumPy, PyTorch, MLX) to provide a consistent
`async_ops.*` interface. It handles backend switching by updating these aliases.
"""

import importlib
import sys
import os
import asyncio
import functools
from typing import Optional, Any, Union, TypeVar

# Import types for type annotations
# Assuming TensorLike, Tensor, DType are defined elsewhere and accessible
# For now, using Any as placeholders as in the provided stub
type TensorLike = Any
type Tensor = Any
type DType = Any

# Define type variables for function signatures
T = TypeVar('T', bound=TensorLike)
D = TypeVar('D', bound=Union[str, DType])

# Import backend control functions
from ember_ml.backend import (
    get_backend,
    set_backend as original_set_backend,
    get_backend_module,
    auto_select_backend
)

# Master list of all functions expected to be aliased by ember_ml.async.ops
# This list is copied from ember_ml.ops.__init__.py
_MASTER_OPS_LIST = [
    # Math (Core arithmetic, trig, exponential, etc.)
    'add', 'subtract', 'multiply', 'divide', 'matmul', 'dot',
    'exp', 'log', 'log10', 'log2', 'pow', 'sqrt', 'square', 'abs', 'sign', 'sin', 'cos', 'tan',
    'sinh', 'cosh', 'clip', 'negative', 'mod', 'floor_divide', 'floor', 'ceil', 'gradient',
    'power', # Note: 'power' is alias for 'pow'
    # Comparison
    'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal', 'logical_and', 'logical_or',
    'logical_not', 'logical_xor', 'allclose', 'isclose', 'all', 'where', 'isnan',
    # Device
    'to_device', 'get_device', 'get_available_devices', 'memory_usage', 'memory_info', 'synchronize',
    'set_default_device', 'get_default_device', 'is_available',
    # IO
    'save', 'load',
    # Loss
    'mse', 'mean_absolute_error', 'binary_crossentropy', 'categorical_crossentropy',
    'sparse_categorical_crossentropy', 'huber_loss', 'log_cosh_loss',
    # Vector (Includes FFT and distance/similarity metrics)
    'normalize_vector', 'compute_energy_stability', 'compute_interference_strength', 'compute_phase_coherence',
    'partial_interference', 'euclidean_distance', 'cosine_similarity', 'exponential_decay', 'fft', 'ifft',
    'fft2', 'ifft2', 'fftn', 'ifftn', 'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
    # Array manipulation
    'vstack', 'hstack',
]

# Helper to create an asynchronous wrapper for a synchronous function
def _make_async_wrapper(sync_func):
    """Wraps a synchronous function in an async function using asyncio.to_thread."""
    @functools.wraps(sync_func)
    async def async_wrapper(*args, **kwargs):
        # Execute the synchronous function in a separate thread
        return await asyncio.to_thread(sync_func, *args, **kwargs)
    return async_wrapper

# Placeholder for functions that will be dynamically loaded
# Initialize globals to None to indicate they aren't loaded yet
for _op_name in _MASTER_OPS_LIST:
    if _op_name not in globals():
        globals()[_op_name] = None

# Keep track of the currently aliased backend
_aliased_backend: Optional[str] = None

def _update_async_ops_aliases():
    """Dynamically updates the async_ops module's namespace with async wrapper functions."""
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

    # Get the backend's synchronous ops module
    try:
        backend_ops_module = getattr(backend_module, 'ops')
    except AttributeError:
        # This is expected for most backends, so make it a debug message instead of an error
        print(f"Debug: Backend '{backend_name}' does not have an 'ops' module. This is expected for most backends.")
        # Set all aliases to None if the ops module is missing
        current_async_ops_module = sys.modules[__name__]
        for func_name in _MASTER_OPS_LIST:
             setattr(current_async_ops_module, func_name, None)
             globals()[func_name] = None
        _aliased_backend = backend_name
        return

    current_async_ops_module = sys.modules[__name__] # Get the 'ember_ml.async.ops' module object

    missing_ops = []
    successful_aliases = 0
    for func_name in _MASTER_OPS_LIST:
        try:
            # Get the synchronous function from the backend's ops module
            sync_func = getattr(backend_ops_module, func_name)
            # Create an asynchronous wrapper
            async_wrapper = _make_async_wrapper(sync_func)
            # Alias the asynchronous wrapper in the current module
            setattr(current_async_ops_module, func_name, async_wrapper)
            globals()[func_name] = async_wrapper
            successful_aliases += 1
        except AttributeError:
            setattr(current_async_ops_module, func_name, None) # Set alias to None
            globals()[func_name] = None
            missing_ops.append(func_name)

    # Handle pi separately as it's a constant
    try:
        setattr(current_async_ops_module, 'pi', getattr(backend_ops_module, 'pi'))
        globals()['pi'] = getattr(backend_ops_module, 'pi')
    except AttributeError:
         setattr(current_async_ops_module, 'pi', None)
         globals()['pi'] = None
         missing_ops.append('pi')


    if missing_ops:
        print(f"Warning: Backend '{backend_name}' ops module does not implement the following functions for async aliasing: {', '.join(missing_ops)}")
    _aliased_backend = backend_name # Mark this backend as aliased

# --- Define set_backend for this module to trigger alias updates ---
# This function will likely be called from ember_ml.backend.__init__.py
# We define it here for clarity on what needs to happen when backend is set.
# The actual set_backend logic that triggers this should be in ember_ml.backend.__init__.py
# def set_backend(backend: str):
#     """Sets the backend and updates async ops aliases."""
#     # This function should ideally be called by the main set_backend in ember_ml.backend
#     # For now, we'll just define the update logic.
#     _update_async_ops_aliases()

# --- Initial alias setup ---
# Ensure backend is determined and aliases populated on first import
# This assumes get_backend() in ember_ml.backend handles initial auto-selection
_init_backend_name = get_backend() # This call triggers auto-selection if needed
_update_async_ops_aliases() # Populate aliases based on the determined backend

# --- Define __all__ ---
# Includes backend controls aliased here and the master list of ops
__all__ = [
    # Expose backend controls via async_ops if desired, or keep them only in backend
    # 'set_backend', 'get_backend', 'auto_select_backend',
    'pi', # Add pi explicitly
] + _MASTER_OPS_LIST # type: ignore