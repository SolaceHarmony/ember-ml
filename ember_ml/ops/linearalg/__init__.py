"""
Operations module.

This module provides operations that abstract machine learning library
scalar operations. Tensor operations ONLY EXIST in ember_ml.nn.tensor and backend.*.tensor.*. The only exception is tensor compatibility with arithmetic.
"""

from typing import Type
_CURRENT_INSTANCES = {}


# Import specific operations from interfaces
from ember_ml.ops.linearalg.linearalg_ops import *

# Use backend directly
from ember_ml.backend import get_backend
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
  
        if ops_class == LinearAlgOps:
            class_name = f"{class_name_prefix}LinearAlgOps"
        else:
            raise ValueError(f"Unknown ops class: {ops_class}")
        
        # Get the class and create an instance
        ops_class_impl = getattr(module, class_name)
        _CURRENT_INSTANCES[ops_class] = ops_class_impl()
    
    return _CURRENT_INSTANCES[ops_class]

def linearalg_ops() -> LinearAlgOps:
    """Get solver operations."""
    return _get_ops_instance(LinearAlgOps)


# Export all functions and classes
__all__ = [
    # Classes
    'LinearAlgOps',
    
    # Functions
    'linearalg_ops',
    
    # Linear Algebra operations
    'solve',
    'inv',
    'det',
    'norm',
    'qr',
    'svd',
    'cholesky',
    'lstsq',
    'eig',
    'eigvals',
    'diag',
    'diagonal',
]

# Linear Algebra operations
solve = lambda *args, **kwargs: linearalg_ops().solve(*args, **kwargs)
inv = lambda *args, **kwargs: linearalg_ops().inv(*args, **kwargs)
det = lambda *args, **kwargs: linearalg_ops().det(*args, **kwargs)
norm = lambda *args, **kwargs: linearalg_ops().norm(*args, **kwargs)
qr = lambda *args, **kwargs: linearalg_ops().qr(*args, **kwargs)
svd = lambda *args, **kwargs: linearalg_ops().svd(*args, **kwargs)
cholesky = lambda *args, **kwargs: linearalg_ops().cholesky(*args, **kwargs)
lstsq = lambda *args, **kwargs: linearalg_ops().lstsq(*args, **kwargs)
eig = lambda *args, **kwargs: linearalg_ops().eig(*args, **kwargs)
eigvals = lambda *args, **kwargs: linearalg_ops().eigvals(*args, **kwargs)
diag = lambda *args, **kwargs: linearalg_ops().diag(*args, **kwargs)
diagonal = lambda *args, **kwargs: linearalg_ops().diagonal(*args, **kwargs)
