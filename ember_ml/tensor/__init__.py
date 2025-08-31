"""
Tensor module for ember_ml.

This module provides a backend-agnostic tensor implementation that works with
any backend (NumPy, PyTorch, MLX) using the backend abstraction layer.
"""

from typing import Any, Optional

from ..tensor_module import (
    EmberTensor,
    zeros, ones, eye, arange, linspace,
    zeros_like, ones_like, full, full_like,
    reshape, transpose, concatenate, stack, split, split_tensor,
    expand_dims, squeeze, tile, gather, scatter, tensor_scatter_nd_update,
    slice_tensor, slice_update, index_update, cast, copy, pad,
    to_numpy, item, shape,
    random_uniform, random_normal, maximum,
    random_bernoulli, random_gamma, random_exponential, random_poisson,
    random_categorical, random_permutation, shuffle, random_shuffle,
    set_seed, get_seed, meshgrid, nonzero, index,
    _convert_to_backend_tensor,
    dtype as dtype,
)

from ..dtypes import (
    EmberDType, DType,
    get_dtype, to_dtype_str, from_dtype_str,
    float32, float64, int32, int64, bool_,
    int8, int16, uint8, uint16, uint32, uint64, float16,
)


def array(data: Any, dtype: Any = None, device: Optional[str] = None) -> Any:
    """Create a raw backend tensor from data. Alias for convert_to_tensor."""
    return convert_to_tensor(data, dtype=dtype, device=device)


def convert_to_tensor(data: Any, dtype: Any = None, device: Optional[str] = None) -> Any:
    """Convert data to a raw backend tensor of the currently active backend."""
    if isinstance(data, EmberTensor):
        return _convert_to_backend_tensor(data.to_backend_tensor(), dtype=dtype, device=device)
    return _convert_to_backend_tensor(data, dtype=dtype, device=device)


__all__ = [
    'EmberTensor',
    'EmberDType',
    'DType',
    'dtype',
    'array',
    'convert_to_tensor',
    'zeros', 'ones', 'eye', 'arange', 'linspace',
    'zeros_like', 'ones_like', 'full', 'full_like',
    'reshape', 'transpose', 'concatenate', 'stack', 'split', 'split_tensor',
    'expand_dims', 'squeeze', 'tile', 'gather', 'scatter', 'tensor_scatter_nd_update',
    'slice_tensor', 'slice_update', 'index_update', 'cast', 'copy', 'pad',
    'to_numpy', 'item', 'index', 'shape',
    'random_uniform', 'random_normal', 'maximum',
    'random_bernoulli', 'random_gamma', 'random_exponential', 'random_poisson',
    'random_categorical', 'random_permutation', 'shuffle', 'random_shuffle', 'set_seed', 'get_seed', 'meshgrid', 'nonzero',
    'float32', 'float64', 'int32', 'int64', 'bool_',
    'int8', 'int16', 'uint8', 'uint16', 'uint32', 'uint64', 'float16',
    'get_dtype', 'to_dtype_str', 'from_dtype_str',
]
