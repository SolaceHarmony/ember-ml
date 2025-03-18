"""
Common tensor implementations.

This module provides backend-agnostic implementations of tensor operations
using the backend abstraction layer.
"""

import importlib

from ember_ml.backend import get_backend, get_backend_module

# Cache for backend instances
_CURRENT_INSTANCES = {}

def _get_backend_module():
    """Get the current backend module."""
    try:
        return get_backend_module()
    except (ImportError, ModuleNotFoundError):
        # If backend-specific implementation not found, use common implementation
        return importlib.import_module('ember_ml.backend.numpy')

def _get_tensor_ops():
    """Get the tensor operations for the current backend."""
    backend = get_backend()
    if backend not in _CURRENT_INSTANCES:
        backend_module = _get_backend_module()
        if hasattr(backend_module, 'Tensor'):
            _CURRENT_INSTANCES[backend] = backend_module.Tensor()
        # Fallback to old implementation for backward compatibility
        elif hasattr(backend_module, 'TensorOps'):
            _CURRENT_INSTANCES[backend] = backend_module.TensorOps()
        else:
            raise ImportError(f"Could not find Tensor implementation for backend {backend}")
    return _CURRENT_INSTANCES[backend]

# Define tensor operations using lambda functions
zeros = lambda *args, **kwargs: _get_tensor_ops().zeros(*args, **kwargs)
ones = lambda *args, **kwargs: _get_tensor_ops().ones(*args, **kwargs)
zeros_like = lambda *args, **kwargs: _get_tensor_ops().zeros_like(*args, **kwargs)
ones_like = lambda *args, **kwargs: _get_tensor_ops().ones_like(*args, **kwargs)
eye = lambda *args, **kwargs: _get_tensor_ops().eye(*args, **kwargs)
arange = lambda *args, **kwargs: _get_tensor_ops().arange(*args, **kwargs)
linspace = lambda *args, **kwargs: _get_tensor_ops().linspace(*args, **kwargs)
full = lambda *args, **kwargs: _get_tensor_ops().full(*args, **kwargs)
full_like = lambda *args, **kwargs: _get_tensor_ops().full_like(*args, **kwargs)
reshape = lambda *args, **kwargs: _get_tensor_ops().reshape(*args, **kwargs)
transpose = lambda *args, **kwargs: _get_tensor_ops().transpose(*args, **kwargs)
concatenate = lambda *args, **kwargs: _get_tensor_ops().concatenate(*args, **kwargs)
stack = lambda *args, **kwargs: _get_tensor_ops().stack(*args, **kwargs)
split = lambda *args, **kwargs: _get_tensor_ops().split(*args, **kwargs)
expand_dims = lambda *args, **kwargs: _get_tensor_ops().expand_dims(*args, **kwargs)
squeeze = lambda *args, **kwargs: _get_tensor_ops().squeeze(*args, **kwargs)
tile = lambda *args, **kwargs: _get_tensor_ops().tile(*args, **kwargs)
gather = lambda *args, **kwargs: _get_tensor_ops().gather(*args, **kwargs)
scatter = lambda *args, **kwargs: _get_tensor_ops().scatter(*args, **kwargs)
tensor_scatter_nd_update = lambda *args, **kwargs: _get_tensor_ops().tensor_scatter_nd_update(*args, **kwargs)
slice = lambda *args, **kwargs: _get_tensor_ops().slice(*args, **kwargs)
slice_update = lambda *args, **kwargs: _get_tensor_ops().slice_update(*args, **kwargs)
# Rename the current function to indicate it's internal
_convert_to_backend_tensor = lambda *args, **kwargs: _get_tensor_ops().convert_to_tensor(*args, **kwargs)
shape = lambda *args, **kwargs: _get_tensor_ops().shape(*args, **kwargs)
dtype = lambda *args, **kwargs: _get_tensor_ops().dtype(*args, **kwargs)
cast = lambda *args, **kwargs: _get_tensor_ops().cast(*args, **kwargs)
copy = lambda *args, **kwargs: _get_tensor_ops().copy(*args, **kwargs)
var = lambda *args, **kwargs: _get_tensor_ops().var(*args, **kwargs)
pad = lambda *args, **kwargs: _get_tensor_ops().pad(*args, **kwargs)
item = lambda *args, **kwargs: _get_tensor_ops().item(*args, **kwargs)
sort = lambda *args, **kwargs: _get_tensor_ops().sort(*args, **kwargs)
argsort = lambda *args, **kwargs: _get_tensor_ops().argsort(*args, **kwargs)
to_numpy = lambda *args, **kwargs: _get_tensor_ops().to_numpy(*args, **kwargs)
tolist = lambda *args, **kwargs: _get_tensor_ops().tolist(*args, **kwargs)
random_uniform = lambda *args, **kwargs: _get_tensor_ops().random_uniform(*args, **kwargs)
random_normal = lambda *args, **kwargs: _get_tensor_ops().random_normal(*args, **kwargs)
maximum = lambda *args, **kwargs: _get_tensor_ops().maximum(*args, **kwargs)

# Add missing random operations
random_bernoulli = lambda *args, **kwargs: _get_tensor_ops().random_binomial(*args, **kwargs)
random_gamma = lambda *args, **kwargs: _get_tensor_ops().random_gamma(*args, **kwargs)
random_exponential = lambda *args, **kwargs: _get_tensor_ops().random_exponential(*args, **kwargs)
random_poisson = lambda *args, **kwargs: _get_tensor_ops().random_poisson(*args, **kwargs)
random_categorical = lambda *args, **kwargs: _get_tensor_ops().random_categorical(*args, **kwargs)
random_permutation = lambda *args, **kwargs: _get_tensor_ops().random_permutation(*args, **kwargs)
shuffle = lambda *args, **kwargs: _get_tensor_ops().shuffle(*args, **kwargs)
set_seed = lambda *args, **kwargs: _get_tensor_ops().set_seed(*args, **kwargs)
get_seed = lambda *args, **kwargs: _get_tensor_ops().get_seed(*args, **kwargs)

# Import EmberTensor class for use in __all__ but don't import it directly
# This avoids the unused import warning
from ember_ml.nn.tensor.common import ember_tensor
EmberTensor = ember_tensor.EmberTensor

__all__ = [
    # Implementations
    'EmberTensor',
    
    # Operations
    'zeros',
    'ones',
    'zeros_like',
    'ones_like',
    'eye',
    'arange',
    'linspace',
    'full',
    'full_like',
    'reshape',
    'transpose',
    'concatenate',
    'stack',
    'split',
    'expand_dims',
    'squeeze',
    'tile',
    'gather',
    'scatter',
    'tensor_scatter_nd_update',
    'slice',
    'slice_update',
    'shape',
    'dtype',
    'cast',
    'copy',
    'var',
    'pad',
    'item',
    'sort',
    'argsort',
    'to_numpy',
    'tolist',
    'random_uniform',
    'random_normal',
    'maximum',
    
    # Additional random operations
    'random_bernoulli',
    'random_gamma',
    'random_exponential',
    'random_poisson',
    'random_categorical',
    'random_permutation',
    'shuffle',
    'set_seed',
    'get_seed',
    # Note: _convert_to_backend_tensor is intentionally not exported
]