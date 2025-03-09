"""
Operations module.

This module provides operations that abstract machine learning library
tensor and scalar operations.
"""

import os
import importlib
from typing import Optional, Dict, Any, Type

# Import interfaces
from ember_ml.ops.interfaces import TensorOps, MathOps, DeviceOps, RandomOps, ComparisonOps, DTypeOps, SolverOps

# Import specific operations from interfaces
from ember_ml.ops.interfaces.tensor_ops import *
from ember_ml.ops.interfaces.math_ops import *
from ember_ml.ops.interfaces.device_ops import *
from ember_ml.ops.interfaces.random_ops import *
from ember_ml.ops.interfaces.comparison_ops import *

# Import data types
from ember_ml.ops.dtypes import *

# Import tensor wrapper
from ember_ml.ops.tensor import EmberTensor

# Import comparison operations
from ember_ml.ops.interfaces.comparison_ops import ComparisonOps

# Use backend directly
from ember_ml.backend import get_backend, set_backend, get_backend_module

_CURRENT_INSTANCES = {}

def get_ops():
    """Get the current ops implementation name."""
    return get_backend()

def set_ops(ops_name: str):
    """Set the current ops implementation."""
    global _CURRENT_INSTANCES
    
    # Set the backend
    set_backend(ops_name)
    
    # Clear instances
    _CURRENT_INSTANCES = {}

def _load_ops_module():
    """Load the current ops module."""
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
        if ops_class == TensorOps:
            class_name = f"{class_name_prefix}TensorOps"
        elif ops_class == MathOps:
            class_name = f"{class_name_prefix}MathOps"
        elif ops_class == DeviceOps:
            class_name = f"{class_name_prefix}DeviceOps"
        elif ops_class == RandomOps:
            class_name = f"{class_name_prefix}RandomOps"
        elif ops_class == ComparisonOps:
            class_name = f"{class_name_prefix}ComparisonOps"
        elif ops_class == DTypeOps:
            class_name = f"{class_name_prefix}DTypeOps"
        elif ops_class == SolverOps:
            class_name = f"{class_name_prefix}SolverOps"
        else:
            raise ValueError(f"Unknown ops class: {ops_class}")
        
        # Get the class and create an instance
        ops_class_impl = getattr(module, class_name)
        _CURRENT_INSTANCES[ops_class] = ops_class_impl()
    
    return _CURRENT_INSTANCES[ops_class]

# Convenience functions
def tensor_ops() -> TensorOps:
    """Get tensor operations."""
    return _get_ops_instance(TensorOps)

def math_ops() -> MathOps:
    """Get math operations."""
    return _get_ops_instance(MathOps)

def device_ops() -> DeviceOps:
    """Get device operations."""
    return _get_ops_instance(DeviceOps)

def random_ops() -> RandomOps:
    """Get random operations."""
    return _get_ops_instance(RandomOps)

def comparison_ops() -> ComparisonOps:
    """Get comparison operations."""
    return _get_ops_instance(ComparisonOps)

def dtype_ops() -> DTypeOps:
    """Get data type operations."""
    return _get_ops_instance(DTypeOps)

def solver_ops() -> SolverOps:
    """Get solver operations."""
    return _get_ops_instance(SolverOps)

# Direct access to operations
# Tensor operations
zeros = lambda *args, **kwargs: tensor_ops().zeros(*args, **kwargs)
ones = lambda *args, **kwargs: tensor_ops().ones(*args, **kwargs)
zeros_like = lambda *args, **kwargs: tensor_ops().zeros_like(*args, **kwargs)
ones_like = lambda *args, **kwargs: tensor_ops().ones_like(*args, **kwargs)
eye = lambda *args, **kwargs: tensor_ops().eye(*args, **kwargs)
arange = lambda *args, **kwargs: tensor_ops().arange(*args, **kwargs)
linspace = lambda *args, **kwargs: tensor_ops().linspace(*args, **kwargs)
full = lambda *args, **kwargs: tensor_ops().full(*args, **kwargs)
full_like = lambda *args, **kwargs: tensor_ops().full_like(*args, **kwargs)
reshape = lambda *args, **kwargs: tensor_ops().reshape(*args, **kwargs)
transpose = lambda *args, **kwargs: tensor_ops().transpose(*args, **kwargs)
concatenate = lambda *args, **kwargs: tensor_ops().concatenate(*args, **kwargs)
stack = lambda *args, **kwargs: tensor_ops().stack(*args, **kwargs)
split = lambda *args, **kwargs: tensor_ops().split(*args, **kwargs)
expand_dims = lambda *args, **kwargs: tensor_ops().expand_dims(*args, **kwargs)
squeeze = lambda *args, **kwargs: tensor_ops().squeeze(*args, **kwargs)
tile = lambda *args, **kwargs: tensor_ops().tile(*args, **kwargs)
gather = lambda *args, **kwargs: tensor_ops().gather(*args, **kwargs)
tensor_scatter_nd_update = lambda *args, **kwargs: tensor_ops().tensor_scatter_nd_update(*args, **kwargs)
convert_to_tensor = lambda *args, **kwargs: tensor_ops().convert_to_tensor(*args, **kwargs)
shape = lambda *args, **kwargs: tensor_ops().shape(*args, **kwargs)
dtype = lambda *args, **kwargs: tensor_ops().dtype(*args, **kwargs)
cast = lambda *args, **kwargs: tensor_ops().cast(*args, **kwargs)
copy = lambda *args, **kwargs: tensor_ops().copy(*args, **kwargs)
var = lambda *args, **kwargs: tensor_ops().var(*args, **kwargs)

# Import all data types from dtypes module
from ember_ml.ops.dtypes import *

get_dtype = lambda *args, **kwargs: dtype_ops().get_dtype(*args, **kwargs)
to_numpy_dtype = lambda *args, **kwargs: dtype_ops().to_numpy_dtype(*args, **kwargs)
from_numpy_dtype = lambda *args, **kwargs: dtype_ops().from_numpy_dtype(*args, **kwargs)

# Math operations
def _get_pi():
    return math_ops().pi

pi = _get_pi()
add = lambda *args, **kwargs: math_ops().add(*args, **kwargs)
subtract = lambda *args, **kwargs: math_ops().subtract(*args, **kwargs)
multiply = lambda *args, **kwargs: math_ops().multiply(*args, **kwargs)
divide = lambda *args, **kwargs: math_ops().divide(*args, **kwargs)
dot = lambda *args, **kwargs: math_ops().dot(*args, **kwargs)
matmul = lambda *args, **kwargs: math_ops().matmul(*args, **kwargs)
mean = lambda *args, **kwargs: math_ops().mean(*args, **kwargs)
sum = lambda *args, **kwargs: math_ops().sum(*args, **kwargs)
max = lambda *args, **kwargs: math_ops().max(*args, **kwargs)
min = lambda *args, **kwargs: math_ops().min(*args, **kwargs)
exp = lambda *args, **kwargs: math_ops().exp(*args, **kwargs)
log = lambda *args, **kwargs: math_ops().log(*args, **kwargs)
log10 = lambda *args, **kwargs: math_ops().log10(*args, **kwargs)
log2 = lambda *args, **kwargs: math_ops().log2(*args, **kwargs)
pow = lambda *args, **kwargs: math_ops().pow(*args, **kwargs)
sqrt = lambda *args, **kwargs: math_ops().sqrt(*args, **kwargs)
square = lambda *args, **kwargs: math_ops().square(*args, **kwargs)
abs = lambda *args, **kwargs: math_ops().abs(*args, **kwargs)
sign = lambda *args, **kwargs: math_ops().sign(*args, **kwargs)
clip = lambda *args, **kwargs: math_ops().clip(*args, **kwargs)
sin = lambda *args, **kwargs: math_ops().sin(*args, **kwargs)
cos = lambda *args, **kwargs: math_ops().cos(*args, **kwargs)
tan = lambda *args, **kwargs: math_ops().tan(*args, **kwargs)
sinh = lambda *args, **kwargs: math_ops().sinh(*args, **kwargs)
cosh = lambda *args, **kwargs: math_ops().cosh(*args, **kwargs)
tanh = lambda *args, **kwargs: math_ops().tanh(*args, **kwargs)
sigmoid = lambda *args, **kwargs: math_ops().sigmoid(*args, **kwargs)
relu = lambda *args, **kwargs: math_ops().relu(*args, **kwargs)
softmax = lambda *args, **kwargs: math_ops().softmax(*args, **kwargs)
sort = lambda *args, **kwargs: math_ops().sort(*args, **kwargs)

# Device operations
to_device = lambda *args, **kwargs: device_ops().to_device(*args, **kwargs)
get_device = lambda *args, **kwargs: device_ops().get_device(*args, **kwargs)
get_available_devices = lambda *args, **kwargs: device_ops().get_available_devices(*args, **kwargs)
memory_usage = lambda *args, **kwargs: device_ops().memory_usage(*args, **kwargs)

# Random operations
random_normal = lambda *args, **kwargs: random_ops().random_normal(*args, **kwargs)
random_uniform = lambda *args, **kwargs: random_ops().random_uniform(*args, **kwargs)
random_binomial = lambda *args, **kwargs: random_ops().random_binomial(*args, **kwargs)
random_permutation = lambda *args, **kwargs: random_ops().random_permutation(*args, **kwargs)
set_random_seed = lambda *args, **kwargs: random_ops().set_random_seed(*args, **kwargs)

# Comparison operations
equal = lambda *args, **kwargs: comparison_ops().equal(*args, **kwargs)
not_equal = lambda *args, **kwargs: comparison_ops().not_equal(*args, **kwargs)
less = lambda *args, **kwargs: comparison_ops().less(*args, **kwargs)
less_equal = lambda *args, **kwargs: comparison_ops().less_equal(*args, **kwargs)
greater = lambda *args, **kwargs: comparison_ops().greater(*args, **kwargs)
greater_equal = lambda *args, **kwargs: comparison_ops().greater_equal(*args, **kwargs)
logical_and = lambda *args, **kwargs: comparison_ops().logical_and(*args, **kwargs)
logical_or = lambda *args, **kwargs: comparison_ops().logical_or(*args, **kwargs)
logical_not = lambda *args, **kwargs: comparison_ops().logical_not(*args, **kwargs)
logical_xor = lambda *args, **kwargs: comparison_ops().logical_xor(*args, **kwargs)
allclose = lambda *args, **kwargs: comparison_ops().allclose(*args, **kwargs)
isclose = lambda *args, **kwargs: comparison_ops().isclose(*args, **kwargs)
all = lambda *args, **kwargs: comparison_ops().all(*args, **kwargs)

# Conversion functions
to_numpy = lambda x: tensor_ops().to_numpy(x)

# Activation functions
def get_activation(activation: str):
    """Get activation function by name."""
    if activation == 'relu':
        return relu
    elif activation == 'sigmoid':
        return sigmoid
    elif activation == 'tanh':
        return tanh
    elif activation == 'softmax':
        return softmax
    else:
        raise ValueError(f"Unknown activation function: {activation}")

# Gradient operations
def gradients(y, xs):
    """Compute gradients of y with respect to xs."""
    # This is a placeholder. The actual implementation will depend on the backend.
    # For now, we'll just use the backend's gradients function.
    from ember_ml import backend as K
    return K.gradients(y, xs)

# Export all functions and classes
__all__ = [
    # Classes
    'TensorOps',
    'MathOps',
    'DeviceOps',
    'RandomOps',
    'ComparisonOps',
    'DTypeOps',
    'SolverOps',
    'EmberTensor',
    
    # Functions
    'get_ops',
    'set_ops',
    'tensor_ops',
    'math_ops',
    'device_ops',
    'random_ops',
    'comparison_ops',
    'dtype_ops',
    'solver_ops',
    'get_activation',
    'gradients',
    'to_numpy',
    
    # Tensor operations
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
    'tensor_scatter_nd_update',
    'convert_to_tensor',
    'shape',
    'dtype',
    'cast',
    'copy',
    'var',
    
    # Math operations
    'add',
    'subtract',
    'multiply',
    'divide',
    'dot',
    'matmul',
    'mean',
    'sum',
    'max',
    'min',
    'exp',
    'log',
    'log10',
    'log2',
    'pow',
    'sqrt',
    'square',
    'abs',
    'sign',
    'clip',
    'sin',
    'cos',
    'tan',
    'sinh',
    'cosh',
    'tanh',
    'sigmoid',
    'relu',
    'softmax',
    'sort',
    
    # Device operations
    'to_device',
    'get_device',
    'get_available_devices',
    'memory_usage',
    
    # Random operations
    'random_normal',
    'random_uniform',
    'random_binomial',
    'random_permutation',
    'set_random_seed',
    
    # Comparison operations
    'equal',
    'not_equal',
    'less',
    'less_equal',
    'greater',
    'greater_equal',
    'logical_and',
    'logical_or',
    'logical_not',
    'logical_xor',
    'allclose',
    'isclose',
    'all',
    
    # Solver operations
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
    
    # Data types
    'int8',
    'int16',
    'int32',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'float16',
    'float32',
    'float64',
    'bool_',
    'get_dtype',
    'to_numpy_dtype',
    'from_numpy_dtype',
]

solve = lambda *args, **kwargs: solver_ops().solve(*args, **kwargs)
inv = lambda *args, **kwargs: solver_ops().inv(*args, **kwargs)
det = lambda *args, **kwargs: solver_ops().det(*args, **kwargs)
norm = lambda *args, **kwargs: solver_ops().norm(*args, **kwargs)
qr = lambda *args, **kwargs: solver_ops().qr(*args, **kwargs)
svd = lambda *args, **kwargs: solver_ops().svd(*args, **kwargs)
cholesky = lambda *args, **kwargs: solver_ops().cholesky(*args, **kwargs)
lstsq = lambda *args, **kwargs: solver_ops().lstsq(*args, **kwargs)
eig = lambda *args, **kwargs: solver_ops().eig(*args, **kwargs)
eigvals = lambda *args, **kwargs: solver_ops().eigvals(*args, **kwargs)
