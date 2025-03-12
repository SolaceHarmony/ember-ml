"""
Operations module.

This module provides operations that abstract machine learning library
tensor and scalar operations.
"""

import os
import importlib
from typing import Optional, Dict, Any, Type

# Import interfaces
from ember_ml.ops.interfaces import TensorOps, MathOps, DeviceOps, RandomOps, ComparisonOps, DTypeOps, SolverOps, IOOps, LossOps, VectorOps

# Import specific operations from interfaces
from ember_ml.ops.interfaces.tensor_ops import *
from ember_ml.ops.interfaces.math_ops import *
from ember_ml.ops.interfaces.device_ops import *
from ember_ml.ops.interfaces.random_ops import *
from ember_ml.ops.interfaces.comparison_ops import *
from ember_ml.ops.interfaces.dtype_ops import *
from ember_ml.ops.interfaces.solver_ops import *
from ember_ml.ops.interfaces.io_ops import *
from ember_ml.ops.interfaces.loss_ops import *
from ember_ml.ops.interfaces.vector_ops import *

# Import data types
from ember_ml.ops.dtypes import *

# Import tensor wrapper
from ember_ml.ops.tensor import EmberTensor

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
        elif backend == 'ember':
            class_name_prefix = 'EmberBackend'
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
        elif ops_class == IOOps:
            class_name = f"{class_name_prefix}IOOps"
        elif ops_class == LossOps:
            class_name = f"{class_name_prefix}LossOps"
        elif ops_class == VectorOps:
            class_name = f"{class_name_prefix}VectorOps"
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

def io_ops() -> IOOps:
    """Get I/O operations."""
    return _get_ops_instance(IOOps)

def loss_ops() -> LossOps:
    """Get loss operations."""
    return _get_ops_instance(LossOps)

def vector_ops() -> VectorOps:
    """Get vector operations."""
    return _get_ops_instance(VectorOps)

# Feature operations are in ember_ml.features, not in ops

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
slice = lambda *args, **kwargs: tensor_ops().slice(*args, **kwargs)
slice_update = lambda *args, **kwargs: tensor_ops().slice_update(*args, **kwargs)
convert_to_tensor = lambda *args, **kwargs: tensor_ops().convert_to_tensor(*args, **kwargs)
shape = lambda *args, **kwargs: tensor_ops().shape(*args, **kwargs)
dtype = lambda *args, **kwargs: tensor_ops().dtype(*args, **kwargs)
cast = lambda *args, **kwargs: tensor_ops().cast(*args, **kwargs)
copy = lambda *args, **kwargs: tensor_ops().copy(*args, **kwargs)
var = lambda *args, **kwargs: tensor_ops().var(*args, **kwargs)
pad = lambda *args, **kwargs: tensor_ops().pad(*args, **kwargs)
item = lambda *args, **kwargs: tensor_ops().item(*args, **kwargs)

# Import all data types from dtypes module
from ember_ml.ops.dtypes import *

get_dtype = lambda *args, **kwargs: dtype_ops().get_dtype(*args, **kwargs)
to_dtype_str = lambda *args, **kwargs: dtype_ops().to_dtype_str(*args, **kwargs)
from_dtype_str = lambda *args, **kwargs: dtype_ops().from_dtype_str(*args, **kwargs)

# Math operations
def _get_pi():
    return math_ops().pi

pi = _get_pi()
add = lambda *args, **kwargs: math_ops().add(*args, **kwargs)
subtract = lambda *args, **kwargs: math_ops().subtract(*args, **kwargs)
multiply = lambda *args, **kwargs: math_ops().multiply(*args, **kwargs)
divide = lambda *args, **kwargs: math_ops().divide(*args, **kwargs)
floor_divide = lambda *args, **kwargs: math_ops().floor_divide(*args, **kwargs)
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
negative = lambda *args, **kwargs: math_ops().negative(*args, **kwargs)
sign = lambda *args, **kwargs: math_ops().sign(*args, **kwargs)
clip = lambda *args, **kwargs: math_ops().clip(*args, **kwargs)
sin = lambda *args, **kwargs: math_ops().sin(*args, **kwargs)
cos = lambda *args, **kwargs: math_ops().cos(*args, **kwargs)
tan = lambda *args, **kwargs: math_ops().tan(*args, **kwargs)
sinh = lambda *args, **kwargs: math_ops().sinh(*args, **kwargs)
cosh = lambda *args, **kwargs: math_ops().cosh(*args, **kwargs)
tanh = lambda *args, **kwargs: math_ops().tanh(*args, **kwargs)
sigmoid = lambda *args, **kwargs: math_ops().sigmoid(*args, **kwargs)
softplus = lambda *args, **kwargs: math_ops().softplus(*args, **kwargs)
relu = lambda *args, **kwargs: math_ops().relu(*args, **kwargs)
softmax = lambda *args, **kwargs: math_ops().softmax(*args, **kwargs)
gradient = lambda *args, **kwargs: math_ops().gradient(*args, **kwargs)

# Tensor sort operation
sort = lambda *args, **kwargs: tensor_ops().sort(*args, **kwargs)

# Device operations
to_device = lambda *args, **kwargs: device_ops().to_device(*args, **kwargs)
get_device = lambda *args, **kwargs: device_ops().get_device(*args, **kwargs)
get_available_devices = lambda *args, **kwargs: device_ops().get_available_devices(*args, **kwargs)
memory_usage = lambda *args, **kwargs: device_ops().memory_usage(*args, **kwargs)
memory_info = lambda *args, **kwargs: device_ops().memory_info(*args, **kwargs)

# Random operations
random_normal = lambda *args, **kwargs: random_ops().random_normal(*args, **kwargs)
random_uniform = lambda *args, **kwargs: random_ops().random_uniform(*args, **kwargs)
random_binomial = lambda *args, **kwargs: random_ops().random_binomial(*args, **kwargs)
random_gamma = lambda *args, **kwargs: random_ops().random_gamma(*args, **kwargs)
random_poisson = lambda *args, **kwargs: random_ops().random_poisson(*args, **kwargs)
random_exponential = lambda *args, **kwargs: random_ops().random_exponential(*args, **kwargs)
random_categorical = lambda *args, **kwargs: random_ops().random_categorical(*args, **kwargs)
random_permutation = lambda *args, **kwargs: random_ops().random_permutation(*args, **kwargs)
shuffle = lambda *args, **kwargs: random_ops().shuffle(*args, **kwargs)
set_seed = lambda *args, **kwargs: random_ops().set_seed(*args, **kwargs)
get_seed = lambda *args, **kwargs: random_ops().get_seed(*args, **kwargs)

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
where = lambda *args, **kwargs: comparison_ops().where(*args, **kwargs)

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

# I/O operations
save = lambda *args, **kwargs: io_ops().save(*args, **kwargs)
load = lambda *args, **kwargs: io_ops().load(*args, **kwargs)

# Loss operations
mean_squared_error = lambda *args, **kwargs: loss_ops().mean_squared_error(*args, **kwargs)
mean_absolute_error = lambda *args, **kwargs: loss_ops().mean_absolute_error(*args, **kwargs)
binary_crossentropy = lambda *args, **kwargs: loss_ops().binary_crossentropy(*args, **kwargs)
categorical_crossentropy = lambda *args, **kwargs: loss_ops().categorical_crossentropy(*args, **kwargs)
sparse_categorical_crossentropy = lambda *args, **kwargs: loss_ops().sparse_categorical_crossentropy(*args, **kwargs)
huber_loss = lambda *args, **kwargs: loss_ops().huber_loss(*args, **kwargs)
log_cosh_loss = lambda *args, **kwargs: loss_ops().log_cosh_loss(*args, **kwargs)

# Vector operations
normalize_vector = lambda *args, **kwargs: vector_ops().normalize_vector(*args, **kwargs)
compute_energy_stability = lambda *args, **kwargs: vector_ops().compute_energy_stability(*args, **kwargs)
compute_interference_strength = lambda *args, **kwargs: vector_ops().compute_interference_strength(*args, **kwargs)
compute_phase_coherence = lambda *args, **kwargs: vector_ops().compute_phase_coherence(*args, **kwargs)
partial_interference = lambda *args, **kwargs: vector_ops().partial_interference(*args, **kwargs)
euclidean_distance = lambda *args, **kwargs: vector_ops().euclidean_distance(*args, **kwargs)
cosine_similarity = lambda *args, **kwargs: vector_ops().cosine_similarity(*args, **kwargs)
exponential_decay = lambda *args, **kwargs: vector_ops().exponential_decay(*args, **kwargs)
gaussian = lambda *args, **kwargs: vector_ops().gaussian(*args, **kwargs)

# Feature operations are in ember_ml.features, not in ops

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
    'VectorOps',
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
    'io_ops',
    'loss_ops',
    'vector_ops',
    'get_activation',
    'gradient',
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
    'slice',
    'slice_update',
    'convert_to_tensor',
    'shape',
    'dtype',
    'cast',
    'copy',
    'var',
    'pad',
    'sort',
    'item',
    
    # Math operations
    'add',
    'subtract',
    'multiply',
    'divide',
    'floor_divide',
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
    'negative',
    'sign',
    'clip',
    'sin',
    'cos',
    'tan',
    'sinh',
    'cosh',
    'tanh',
    'sigmoid',
    'softplus',
    'relu',
    'softmax',
    'gradient',
    
    # Device operations
    'to_device',
    'get_device',
    'get_available_devices',
    'memory_usage',
    'memory_info',
    
    # Random operations
    'random_normal',
    'random_uniform',
    'random_binomial',
    'random_gamma',
    'random_poisson',
    'random_exponential',
    'random_categorical',
    'random_permutation',
    'shuffle',
    'set_seed',
    'get_seed',
    
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
    'where',
    
    # I/O operations
    'save',
    'load',
    
    # Loss operations
    'mean_squared_error',
    'mean_absolute_error',
    'binary_crossentropy',
    'categorical_crossentropy',
    'sparse_categorical_crossentropy',
    'huber_loss',
    'log_cosh_loss',
    
    # Vector operations
    'normalize_vector',
    'compute_energy_stability',
    'compute_interference_strength',
    'compute_phase_coherence',
    'partial_interference',
    'euclidean_distance',
    'cosine_similarity',
    'exponential_decay',
    'gaussian',
    
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
    'to_dtype_str',
    'from_dtype_str',
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
