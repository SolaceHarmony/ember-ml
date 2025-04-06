"""
Statistical operations module.

This module provides operations for statistical analysis of tensors.
"""

from typing import Type
from ember_ml.ops.stats.stats_ops import StatsOps

_CURRENT_INSTANCES = {}

def _load_ops_module():
    """Load the current ops module."""
    from ember_ml.backend import get_backend_module
    return get_backend_module()

def _get_ops_instance(ops_class: Type):
    """Get an instance of the specified ops class."""
    global _CURRENT_INSTANCES

    if ops_class not in _CURRENT_INSTANCES:
        module = _load_ops_module()

        # Get the backend directly
        from ember_ml.backend import get_backend
        backend = get_backend()

        # Get the ops class name based on the current implementation
        if backend == 'numpy':
            class_name_prefix = 'Numpy'
        elif backend == 'torch':
            class_name_prefix = 'Torch'
        elif backend == 'mlx':
            class_name_prefix = 'MLX'
        else:
            raise ValueError(f"Unknown ops implementation: {backend}")

        # Get the class name
        if ops_class == StatsOps:
            class_name = f"{class_name_prefix}StatsOps"
        else:
            raise ValueError(f"Unknown ops class: {ops_class}")

        # Get the class and create an instance
        ops_class_impl = getattr(module, class_name)
        _CURRENT_INSTANCES[ops_class] = ops_class_impl()

    return _CURRENT_INSTANCES[ops_class]

def stats_ops():
    """Get the stats ops implementation for the current backend."""
    return _get_ops_instance(StatsOps)

# Expose all statistical operations through lambda functions
mean = lambda *args, **kwargs: stats_ops().mean(*args, **kwargs)
var = lambda *args, **kwargs: stats_ops().var(*args, **kwargs)
median = lambda *args, **kwargs: stats_ops().median(*args, **kwargs)
std = lambda *args, **kwargs: stats_ops().std(*args, **kwargs)
percentile = lambda *args, **kwargs: stats_ops().percentile(*args, **kwargs)
max = lambda *args, **kwargs: stats_ops().max(*args, **kwargs)
min = lambda *args, **kwargs: stats_ops().min(*args, **kwargs)
sum = lambda *args, **kwargs: stats_ops().sum(*args, **kwargs)
cumsum = lambda *args, **kwargs: stats_ops().cumsum(*args, **kwargs)
argmax = lambda *args, **kwargs: stats_ops().argmax(*args, **kwargs)
sort = lambda *args, **kwargs: stats_ops().sort(*args, **kwargs)
argsort = lambda *args, **kwargs: stats_ops().argsort(*args, **kwargs)

# Define exports
__all__ = [
    'mean',
    'var',
    'median',
    'std',
    'percentile',
    'max',
    'min',
    'sum',
    'cumsum',
    'argmax',
    'sort',
    'argsort',
]