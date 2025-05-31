"""
Operations module.

This module dynamically aliases functions from the active backend
(NumPy, PyTorch, MLX) to provide a consistent `ops.*` interface.
It handles backend switching by updating these aliases.
"""

import importlib
import sys
import logging
from typing import Optional, Any, Union, TypeVar, Protocol, runtime_checkable

# Setup basic logging configuration if not already configured
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Import types for type annotations
@runtime_checkable
class TensorLike(Protocol):
    """Protocol for tensor-like objects"""
    def __array__(self) -> Any: ...

type Tensor = Any  # Placeholder for actual tensor types
type DType = Any  # Placeholder for actual data types

# Define type variables for function signatures
T = TypeVar('T', bound=TensorLike)
D = TypeVar('D', bound=Union[str, DType])

# Import backend control functions (including the original set_backend)
from ember_ml.backend import (
    get_backend,
    set_backend as original_set_backend,
    get_backend_module,
    auto_select_backend
)

# Master list of all functions expected to be aliased by ember_ml.ops
_MASTER_OPS_LIST = [
    # Math (Core arithmetic, trig, exponential, etc.)
    'add', 'subtract', 'multiply', 'divide', 'matmul', 'dot',
    'exp', 'log', 'log10', 'log2', 'pow', 'sqrt', 'square', 'abs', 'sign', 'sin', 'cos', 'tan',
    'sinh', 'cosh', 'clip', 'negative', 'mod', 'floor_divide', 'floor', 'ceil', 'gradient',
    'power', # Note: 'power' is alias for 'pow'
    # Comparison
    'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal', 'logical_and', 'logical_or',
    'logical_not', 'logical_xor', 'allclose', 'isclose', 'all', 'any', 'where', 'isnan',
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

# Placeholder for functions that will be dynamically loaded
# Initialize globals to None to indicate they aren't loaded yet
for _op_name in _MASTER_OPS_LIST:
    # Check if the name is already defined (like built-ins: max, min, sum, pow, abs, all)
    if _op_name not in globals():
        globals()[_op_name] = None

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
        import logging
        logging.warning(f"Failed to reload backend module {backend_module.__name__}: {e}")
    current_ops_module = sys.modules[__name__] # Get the 'ember_ml.ops' module object

    missing_ops = []
    successful_aliases = 0
    for func_name in _MASTER_OPS_LIST:
        try:
            backend_function = getattr(backend_module, func_name)
            setattr(current_ops_module, func_name, backend_function)
            # Also update globals() so functions defined *within this file* can use the aliases
            globals()[func_name] = backend_function
            successful_aliases += 1
        except AttributeError:
            setattr(current_ops_module, func_name, None) # Set alias to None
            globals()[func_name] = None
            missing_ops.append(func_name)
    
    # Import pi directly from math to avoid backend issues
    try:
        # Try to get pi from the backend's math_ops module
        setattr(current_ops_module, 'pi', backend_module.math_ops.pi)
        globals()['pi'] = backend_module.math_ops.pi
    except AttributeError:
        # If math_ops is not available, try to get pi directly from the backend
        try:
            setattr(current_ops_module, 'pi', backend_module.pi)
            globals()['pi'] = backend_module.pi
        except AttributeError:
            # If pi is not available in the backend, use the value from math
            import math
            setattr(current_ops_module, 'pi', math.pi)
            globals()['pi'] = math.pi
            import logging
            logging.warning(f"Using math.pi as fallback for backend '{backend_name}'")
    if missing_ops:
        import logging
        logging.warning(f"Backend '{backend_name}' does not implement the following ops: {', '.join(missing_ops)}")
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
    _update_submodule_aliases()

def _update_submodule_aliases():
    """Update all submodule aliases when the backend changes."""
    try:
        from ember_ml.ops.stats import _update_stats_aliases
        _update_stats_aliases()
    except ImportError:
        import logging
        logging.warning("Could not import or update stats aliases.")
    
    try:
        from ember_ml.ops.linearalg import _update_linearalg_aliases
        _update_linearalg_aliases()
    except ImportError:
        import logging
        logging.warning("Could not import or update linearalg aliases.")
    
    try:
        from ember_ml.ops.bitwise import _update_bitwise_aliases
        _update_bitwise_aliases()
    except ImportError:
        import logging
        logging.warning("Could not import or update bitwise aliases.")
    
    try:
        from ember_ml.nn.modules.activations import _update_activation_aliases
        _update_activation_aliases()
    except ImportError:
        import logging
        logging.warning("Could not import or update activation aliases.")
    
    try:
        from ember_ml.nn.features import _update_features_aliases
        _update_features_aliases()
    except ImportError:
        import logging
        logging.warning("Could not import or update features aliases.")

# --- Initial alias setup ---
# Ensure backend is determined and aliases populated on first import
_init_backend_name = get_backend() # This call triggers auto-selection if needed
_update_ops_aliases() # Populate aliases based on the determined backend

# Import submodules AFTER initial setup to ensure they get proper backend context
# But defer alias updates until after imports to avoid circular imports
from ember_ml.ops import stats
from ember_ml.ops import linearalg
from ember_ml.ops import bitwise

# Now that all modules are imported, trigger alias updates
_update_submodule_aliases()

# --- Define __all__ ---
# Includes backend controls aliased here and the master list of ops
__all__ = [
    'set_backend', 'get_backend', 'auto_select_backend', # Expose backend controls via ops
    'pi', # Add pi explicitly
    'stats', 'linearalg', 'bitwise', # Expose submodules
] + _MASTER_OPS_LIST # type: ignore