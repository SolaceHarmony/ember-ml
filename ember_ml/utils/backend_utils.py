"""
Backend Utilities for ember_ml

This module provides utility functions for working with ember_ml's backend system,
making it easier to convert between NumPy arrays and backend tensors, and to perform
common operations in a backend-agnostic way.
"""

import logging
from typing import Any, List, Optional, Tuple, Dict

from ember_ml import tensor

# Set up logging
logger = logging.getLogger('ember_ml.utils.backend')

# Backend module paths
_BACKENDS = {
    'numpy': 'ember_ml.backend.numpy',
    'torch': 'ember_ml.backend.torch',
    'torch_optimized': 'ember_ml.backend.torch_backend_optimized',
    'mlx': 'ember_ml.backend.mlx'
}

# Import ember_ml backend
# Use relative imports to ensure availability within the package
try:
    # Try absolute import first (for installed package)
    from ember_ml.backend import get_backend, set_backend
    from ember_ml import ops
except ImportError:
    # Fall back to relative import (for development)
    try:
        from ..backend import get_backend, set_backend
        from .. import ops
    except ImportError:
        # Critical failure - ember_ml backend is required
        logger.error("CRITICAL ERROR: ember_ml backend not available. This is a required dependency.")
        raise ImportError("ember_ml backend is required but not available. Please ensure the package is properly installed.")

def get_current_backend() -> str:
    """
    Get the current backend name.
    
    Returns:
        str: Name of the current backend ('mlx', 'torch', 'numpy')
    """
    return get_backend()

def set_preferred_backend(backend_name: Optional[str] = None) -> str:
    """
    Set the preferred backend if available.
    
    Args:
        backend_name: Name of the preferred backend ('mlx', 'torch', 'numpy')
        
    Returns:
        str: Name of the actually set backend
    """
    if backend_name is None:
        # Let ember_ml choose the best available backend
        return get_backend()
    
    try:
        set_backend(backend_name)
        logger.info(f"Set backend to {backend_name}")
        return get_backend()
    except ValueError:
        logger.warning(f"Backend {backend_name} not available. Using default.")
        return get_backend()

def initialize_random_seed(seed: int = 42) -> None:
    """
    Initialize random seed for reproducibility across all backends.
    
    Args:
        seed: Random seed value
    """
    # Import the set_seed function from the current backend
    backend_module = __import__(_BACKENDS[get_current_backend()], fromlist=['set_seed'])
    if hasattr(backend_module, 'set_seed'):
        backend_module.set_seed(seed)
        logger.info(f"Set random seed to {seed} for {get_backend()} backend")
    else:
        logger.warning(f"Backend {get_backend()} does not support setting random seed")


def sin_cos_transform(values: Any, period: float = 1.0) -> Tuple[Any, Any]:
    """
    Apply sine and cosine transformations for cyclical features.
    
    Args:
        values: Input values to transform (TensorLike)
        period: Period for the transformation (float)
        
    Returns:
        Tuple of (sin_values, cos_values) as EmberTensors
    """
    values_tensor = tensor.convert_to_tensor(values)  # Ensure it's an EmberTensor

    # Determine dtype and device safely without direct attribute access
    try:
        device = ops.get_device(values_tensor)
    except Exception:
        device = getattr(values_tensor, "device", None)

    try:
        dtype = ops.dtype(values_tensor)
    except Exception:
        dtype = getattr(values_tensor, "dtype", None)

    # Ensure constants are tensors of the same dtype and device for ops
    # ops.pi is likely a float, ensure it's converted correctly
    # The ops functions (multiply, divide, sin, cos) should handle broadcasting of scalar constants
    # if values_tensor is an EmberTensor. However, explicit conversion is safer for backend purity.

    two_pi_val = 2 * ops.pi  # Python float

    # Let ops handle scalar broadcasting if possible, assuming period is float
    # arg = ops.divide(ops.multiply(two_pi_val, values_tensor), period)
    # For stricter backend purity, convert all scalars to tensors:
    two_pi_tensor = tensor.convert_to_tensor(two_pi_val, dtype=dtype, device=device)
    period_tensor = tensor.convert_to_tensor(period, dtype=dtype, device=device)

    term_mul = ops.multiply(two_pi_tensor, values_tensor)
    arg = ops.divide(term_mul, period_tensor)

    sin_values = ops.sin(arg)
    cos_values = ops.cos(arg)
    
    return sin_values, cos_values

def vstack_safe(arrays: List[Any]) -> Optional[Any]:
    """
    Safely stack arrays vertically using the current backend.

    Args:
        arrays: List of arrays to stack

    Returns:
        Stacked array in the current backend format. When input items are not
        already tensors, they are converted using :func:`tensor.convert_to_tensor`
        with :data:`tensor.float32` as the default ``dtype``.
    """
    if not arrays:
        return None # Or return an empty tensor: tensor.zeros((0,)) etc.

    # Infer common dtype/device from first element when possible
    first_item_device = ops.get_device()
    first_item_dtype = tensor.float32
    first = arrays[0]
    if hasattr(first, "dtype"):
        try:
            first_item_dtype = tensor.dtype(first)
        except Exception:
            pass
    if hasattr(first, "device"):
        try:
            first_item_device = first.device  # type: ignore
        except Exception:
            pass

    ember_tensors = [tensor.convert_to_tensor(arr, dtype=first_item_dtype, device=first_item_device) for arr in arrays]

    # Ensure all tensors are now on the same device (e.g., device of the first tensor)
    # This step might be important if convert_to_tensor doesn't unify devices.
    # For now, assume convert_to_tensor handles device placement or they are already compatible.
    # A more robust version would explicitly move tensors to a common device.

    # Check if all arrays have the same shape except for the first dimension
    if not ember_tensors: # Should have been caught by `if not arrays:`
        return None

    first_shape_rest = tensor.shape(ember_tensors[0])[1:]
    # shapes_match = True # Initialize to True
    # for arr_et in ember_tensors[1:]:
    #     if tensor.shape(arr_et)[1:] != first_shape_rest:
    #         shapes_match = False
    #         break
    # A more Pythonic way using all()
    shapes_match = all(tensor.shape(arr_et)[1:] == first_shape_rest for arr_et in ember_tensors[1:])

    if shapes_match:
        return tensor.concatenate(ember_tensors, axis=0)
    else:
        # Log the actual shapes for better debugging
        all_shapes = [tensor.shape(arr_et) for arr_et in ember_tensors]
        logger.warning(f"Arrays have incompatible trailing shapes: {all_shapes}. Cannot vstack.")
        # Returning None or raising an error might be better than returning the first element.
        # For now, returning None as per original behavior for empty arrays.
        return None

def get_backend_info() -> Dict[str, Any]:
    """
    Get information about the current backend.
    
    Returns:
        Dict with backend information
    """
    info = {
        'name': get_current_backend()
    }
    
    # Add more backend-specific information
    try:
        info['device'] = ops.get_device()
    except:
        info['device'] = 'unknown'
    
    return info

def print_backend_info() -> None:
    """Print information about the current backend."""
    info = get_backend_info()
    
    print(f"Current backend: {info['name']}")
    print(f"Device: {info.get('device', 'unknown')}")
    
    # Test a simple operation
    a = tensor.ones((2, 2))
    b = tensor.ones((2, 2))
    c = ops.matmul(a, b)

    print(f"Test operation: {a} @ {b} = {c}")
