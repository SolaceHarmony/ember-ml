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
    EmberDType, DType, dtype as dtype_instance, # Alias the instance import
    get_dtype, to_dtype_str, from_dtype_str
)
# Import the dtype *function* separately to ensure it's available
from ember_ml.nn.tensor.common import dtype # noqa
# Import dtype objects directly from dtypes.py
from ember_ml.nn.tensor.common.dtypes import (  # noqa
    float32, float64, int32, int64, bool_,
    int8, int16, uint8, uint16, uint32, uint64, float16
)

# Import tensor operations from common
from ember_ml.nn.tensor.common import (  # noqa
    zeros, ones, eye, arange, linspace,
    zeros_like, ones_like, full, full_like,
    reshape, transpose, concatenate, stack, split, split_tensor,
    expand_dims, squeeze, tile, gather, scatter, tensor_scatter_nd_update,
    slice_tensor, slice_update, index_update, cast, copy, pad,
    to_numpy, item, shape,
    random_uniform, random_normal, maximum,
    random_bernoulli, random_gamma, random_exponential, random_poisson,
    random_categorical, random_permutation, shuffle, random_shuffle, set_seed, get_seed,
    meshgrid, nonzero, index # Add nonzero here
)

# Import the internal conversion function
from ember_ml.nn.tensor.common import _convert_to_backend_tensor
from typing import Any

# Define array function to return a raw backend tensor
def array(data: Any, dtype: Any = None, device: Optional[str] = None) -> Any: # Removed requires_grad
    """
    Create a raw backend tensor from data. Alias for convert_to_tensor.
    
    Args:
        data: Input data (array, list, scalar, EmberTensor, or backend tensor)
        dtype: Optional data type for the resulting backend tensor.
        device: Optional device to place the backend tensor on.
        
    Returns:
        Raw backend tensor.
    """
    return convert_to_tensor(data, dtype=dtype, device=device)
 
def convert_to_tensor(data: Any, dtype: Any = None, device: Optional[str] = None) -> Any: # Removed requires_grad
    """
    Convert data to a raw backend tensor of the currently active backend.

    If the input is an EmberTensor, its underlying backend tensor will be
    extracted and potentially converted to the specified dtype and device.
    If the input is already a backend tensor of the active backend, it might
    be returned as is or converted if dtype/device are different.
    Other data types (lists, scalars, NumPy arrays) will be converted.
    
    Args:
        data: Input data (array, list, scalar, EmberTensor, or backend tensor).
        dtype: Optional target data type for the backend tensor.
        device: Optional target device for the backend tensor.
        
    Returns:
        A raw backend tensor.
    """
    if isinstance(data, EmberTensor):
        # If it's an EmberTensor, get its backend tensor.
        # Then, _convert_to_backend_tensor will handle dtype/device conversion if needed.
        # _convert_to_backend_tensor should be able to take a backend tensor as input.
        return _convert_to_backend_tensor(data.to_backend_tensor(), dtype=dtype, device=device)
    # For any other type of data (including raw backend tensors from a different backend,
    # or lists, numpy arrays, scalars), _convert_to_backend_tensor handles it.
    return _convert_to_backend_tensor(data, dtype=dtype, device=device)

# Export all classes and functions
__all__ = [
    # Interfaces
    'TensorInterface',
    'DTypeInterface',
    
    # Implementations
    'EmberTensor',
    'EmberDType',
    'DType',
    'dtype', # This should now correctly refer to the function
    # 'dtype_instance' is not typically part of the public API, so omit from __all__
    
    # Tensor constructor
    'array',
    'convert_to_tensor',
    
    # Tensor operations
    'zeros', 'ones', 'eye', 'arange', 'linspace',
    'zeros_like', 'ones_like', 'full', 'full_like',
    'reshape', 'transpose', 'concatenate', 'stack', 'split', 'split_tensor',
    'expand_dims', 'squeeze', 'tile', 'gather', 'scatter', 'tensor_scatter_nd_update',
    'slice_tensor', 'slice_update', 'index_update', 'cast', 'copy', 'pad',
    'to_numpy', 'item', 'index','shape',
    'random_uniform', 'random_normal', 'maximum',
    'random_bernoulli', 'random_gamma', 'random_exponential', 'random_poisson',
    'random_categorical', 'random_permutation', 'shuffle', 'random_shuffle', 'set_seed', 'get_seed', 'meshgrid', 'nonzero', # Add nonzero here
    
    # Data types
    'float32', 'float64', 'int32', 'int64', 'bool_',
    'int8', 'int16', 'uint8', 'uint16', 'uint32', 'uint64', 'float16',
    
    # Data type operations
    'get_dtype', 'to_dtype_str', 'from_dtype_str',
]