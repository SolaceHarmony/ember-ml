"""
Tensor module for ember_ml.

This module provides a backend-agnostic tensor implementation that works with
any backend (NumPy, PyTorch, MLX) using the backend abstraction layer.
"""

# Import interfaces
from ember_ml.nn.tensor.interfaces import TensorInterface  # noqa
from ember_ml.nn.tensor.interfaces.dtype import DTypeInterface  # noqa

# Import directly from common implementation
from ember_ml.nn.tensor.common import EmberTensor  # noqa
from ember_ml.nn.tensor.common.dtypes import (  # noqa
    EmberDType, DType, dtype,
    get_dtype, to_dtype_str, from_dtype_str
)
# Import dtype objects directly from dtypes.py
from ember_ml.nn.tensor.common.dtypes import (  # noqa
    float32, float64, int32, int64, bool_,
    int8, int16, uint8, uint16, uint32, uint64, float16
)

# Import tensor operations from common
from ember_ml.nn.tensor.common import (  # noqa
    zeros, ones, eye, arange, linspace,
    zeros_like, ones_like, full, full_like,
    reshape, transpose, concatenate, stack, split,
    expand_dims, squeeze, tile, gather, scatter, tensor_scatter_nd_update,
    slice, slice_update, cast, copy, var, pad,
    sort, argsort, to_numpy, item, shape,
    random_uniform, random_normal, maximum,
    random_bernoulli, random_gamma, random_exponential, random_poisson,
    random_categorical, random_permutation, shuffle, set_seed, get_seed
)

# Import internal functions for backward compatibility
# These are marked with underscore to indicate they are internal
from ember_ml.nn.tensor.common import _convert_to_backend_tensor  # noqa

# Define array function as an alias for EmberTensor constructor
def array(data, dtype=None, device=None, requires_grad=False):
    """
    Create a tensor from data.
    
    Args:
        data: Input data (array, list, scalar)
        dtype: Optional data type
        device: Optional device to place the tensor on
        requires_grad: Whether the tensor requires gradients
        
    Returns:
        EmberTensor
    """
    return EmberTensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

from typing import Any

def convert_to_tensor(data: Any, dtype=None, device=None, requires_grad=False):
    """
    Create a tensor from data.
    
    Args:
        data: Input data (array, list, scalar, or tensor)
        dtype: Optional data type
        device: Optional device to place the tensor on
        requires_grad: Whether the tensor requires gradients
        
    Returns:
        EmberTensor
    """
    # If already an EmberTensor, return it directly (reference passing)
    if type(EmberTensor) == type(data):
        return data
    
    # Convert to backend tensor first using the internal function
    from ember_ml.nn.tensor.common import _convert_to_backend_tensor
    backend_tensor = _convert_to_backend_tensor(data, dtype=dtype)
    
    # Wrap in EmberTensor
    return EmberTensor(backend_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

# Export all classes and functions
__all__ = [
    # Interfaces
    'TensorInterface',
    'DTypeInterface',
    
    # Implementations
    'EmberTensor',
    'EmberDType',
    'DType',
    'dtype',
    
    # Tensor constructor
    'array',
    'convert_to_tensor',
    
    # Internal functions (for backward compatibility)
    '_convert_to_backend_tensor',
    
    # Tensor operations
    'zeros', 'ones', 'eye', 'arange', 'linspace',
    'zeros_like', 'ones_like', 'full', 'full_like',
    'reshape', 'transpose', 'concatenate', 'stack', 'split',
    'expand_dims', 'squeeze', 'tile', 'gather', 'scatter', 'tensor_scatter_nd_update',
    'slice', 'slice_update', 'cast', 'copy', 'var', 'pad',
    'sort', 'argsort', 'to_numpy', 'item', 'shape', 'dtype',
    'random_uniform', 'random_normal', 'maximum',
    'random_bernoulli', 'random_gamma', 'random_exponential', 'random_poisson',
    'random_categorical', 'random_permutation', 'shuffle', 'set_seed', 'get_seed',
    
    # Data types
    'float32', 'float64', 'int32', 'int64', 'bool_',
    'int8', 'int16', 'uint8', 'uint16', 'uint32', 'uint64', 'float16',
    
    # Data type operations
    'get_dtype', 'to_dtype_str', 'from_dtype_str',
]