"""
NumPy tensor operations.

This module provides standalone functions for tensor operations using the NumPy backend.
These functions can be called directly or through the NumpyTensor class methods.
"""

# Import operations from modules
from ember_ml.backend.numpy.tensor.ops.casting import cast
from ember_ml.backend.numpy.tensor.ops.creation import (
    zeros, ones, eye, zeros_like, ones_like, full, full_like, arange, linspace
)
from ember_ml.backend.numpy.tensor.ops.manipulation import (
    reshape, transpose, concatenate, stack, split, expand_dims, squeeze, tile, pad
)
from ember_ml.backend.numpy.tensor.ops.indexing import (
    slice_tensor, slice_update, gather, tensor_scatter_nd_update
)
from ember_ml.backend.numpy.tensor.ops.utility import (
    convert_to_tensor, to_numpy, item, shape, dtype, copy, var, sort, argsort, maximum
)
from ember_ml.backend.numpy.tensor.ops.random import (
    random_normal, random_uniform, random_binomial, random_gamma, random_exponential,
    random_poisson, random_categorical, random_permutation, shuffle, set_seed, get_seed
)

# Export all operations
__all__ = [
    # Casting operations
    'cast',
    
    # Creation operations
    'zeros',
    'ones',
    'eye',
    'zeros_like',
    'ones_like',
    'full',
    'full_like',
    'arange',
    'linspace',
    
    # Manipulation operations
    'reshape',
    'transpose',
    'concatenate',
    'stack',
    'split',
    'expand_dims',
    'squeeze',
    'tile',
    'pad',
    
    # Indexing operations
    'slice_tensor',
    'slice_update',
    'gather',
    'tensor_scatter_nd_update',
    
    # Utility operations
    'convert_to_tensor',
    'to_numpy',
    'item',
    'shape',
    'dtype',
    'copy',
    'var',
    'sort',
    'argsort',
    'maximum',
    
    # Random operations
    'random_normal',
    'random_uniform',
    'random_binomial',
    'random_gamma',
    'random_exponential',
    'random_poisson',
    'random_categorical',
    'random_permutation',
    'shuffle',
    'set_seed',
    'get_seed',
]