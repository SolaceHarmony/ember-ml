"""
Operations module.

This module provides operations that abstract machine learning library
scalar operations. Tensor operations ONLY EXIST in ember_ml.nn.tensor and backend.*.tensor.*. The only exception is tensor compatibility with arithmetic.
"""

from typing import Type, Dict
from abc import ABC

# Import interfaces
from ember_ml.ops.math_ops import MathOps
from ember_ml.ops.device_ops import DeviceOps
from ember_ml.ops.comparison_ops import ComparisonOps
from ember_ml.ops.io_ops import IOOps
from ember_ml.ops.loss_ops import LossOps
from ember_ml.ops.vector_ops import VectorOps
#from ember_ml.ops.linearalg import LinearAlgOps

# Import specific operations from interfaces
from ember_ml.ops.math_ops import *
from ember_ml.ops.device_ops import *
from ember_ml.ops.comparison_ops import *
from ember_ml.ops.io_ops import *
from ember_ml.ops.loss_ops import *
from ember_ml.ops.vector_ops import *
# Stats is imported directly by users, not through ops, similar to linearalg

# Use backend directly
from ember_ml.backend import get_backend, set_backend, get_backend_module

_CURRENT_INSTANCES: Dict[Type[ABC], ABC] = {}

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
        if ops_class == MathOps:
            class_name = f"{class_name_prefix}MathOps"
        elif ops_class == DeviceOps:
            class_name = f"{class_name_prefix}DeviceOps"
        elif ops_class == ComparisonOps:
            class_name = f"{class_name_prefix}ComparisonOps"
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
def math_ops() -> MathOps:
    """Get math operations."""
    return _get_ops_instance(MathOps)

def device_ops() -> DeviceOps:
    """Get device operations."""
    return _get_ops_instance(DeviceOps)

def comparison_ops() -> ComparisonOps:
    """Get comparison operations."""
    return _get_ops_instance(ComparisonOps)

def io_ops() -> IOOps:
    """Get I/O operations."""
    return _get_ops_instance(IOOps)

def loss_ops() -> LossOps:
    """Get loss operations."""
    return _get_ops_instance(LossOps)

def vector_ops() -> VectorOps:
    """Get vector operations."""
    return _get_ops_instance(VectorOps)

# Math operations
add = lambda *args, **kwargs: math_ops().add(*args, **kwargs)
subtract = lambda *args, **kwargs: math_ops().subtract(*args, **kwargs)
multiply = lambda *args, **kwargs: math_ops().multiply(*args, **kwargs)
divide = lambda *args, **kwargs: math_ops().divide(*args, **kwargs)
gather = lambda *args, **kwargs: math_ops().gather(*args, **kwargs)
floor_divide = lambda *args, **kwargs: math_ops().floor_divide(*args, **kwargs)
dot = lambda *args, **kwargs: math_ops().dot(*args, **kwargs)
matmul = lambda *args, **kwargs: math_ops().matmul(*args, **kwargs)
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
eigh = lambda *args, **kwargs: math_ops().eigh(*args, **kwargs)
pi = math_ops().pi

# Device operations
to_device = lambda *args, **kwargs: device_ops().to_device(*args, **kwargs)
get_device = lambda *args, **kwargs: device_ops().get_device(*args, **kwargs)
get_available_devices = lambda *args, **kwargs: device_ops().get_available_devices(*args, **kwargs)
memory_usage = lambda *args, **kwargs: device_ops().memory_usage(*args, **kwargs)
memory_info = lambda *args, **kwargs: device_ops().memory_info(*args, **kwargs)

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
isnan = lambda *args, **kwargs: comparison_ops().isnan(*args, **kwargs)

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
fft = lambda *args, **kwargs: vector_ops().fft(*args, **kwargs)
ifft = lambda *args, **kwargs: vector_ops().ifft(*args, **kwargs)
fft2 = lambda *args, **kwargs: vector_ops().fft2(*args, **kwargs)
ifft2 = lambda *args, **kwargs: vector_ops().ifft2(*args, **kwargs)
fftn = lambda *args, **kwargs: vector_ops().fftn(*args, **kwargs)
ifftn = lambda *args, **kwargs: vector_ops().ifftn(*args, **kwargs)
rfft = lambda *args, **kwargs: vector_ops().rfft(*args, **kwargs)
irfft = lambda *args, **kwargs: vector_ops().irfft(*args, **kwargs)
rfft2 = lambda *args, **kwargs: vector_ops().rfft2(*args, **kwargs)
irfft2 = lambda *args, **kwargs: vector_ops().irfft2(*args, **kwargs)
rfftn = lambda *args, **kwargs: vector_ops().rfftn(*args, **kwargs)
irfftn = lambda *args, **kwargs: vector_ops().irfftn(*args, **kwargs)
# Export all functions and classes
__all__ = [
    # Classes
    'MathOps',
    'DeviceOps',
    'ComparisonOps',
    'SolverOps',
    'VectorOps',
    
    # Functions
    'get_ops',
    'set_ops',
    'set_backend',
    'math_ops',
    'device_ops',
    'comparison_ops',
    'solver_ops',
    'io_ops',
    'loss_ops',
    'vector_ops',
    'linearalg_ops',
    'get_activation',
    'gradient',
    
    
    # Math operations
    'add',
    'subtract',
    'multiply',
    'divide',
    'floor_divide',
    'dot',
    'matmul',
    'gather',
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
    'eigh',
    'eigh',
    
    # Device operations
    'to_device',
    'get_device',
    'get_available_devices',
    'memory_usage',
    'memory_info',
    
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
    'isnan',
    
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
    'gaussian'
    'fft',
    'ifft',
    'fft2',
    'ifft2',
    'fftn',
    'ifftn',
    'rfft',
    'irfft',
    'rfft2',
    'irfft2',
    'rfftn',
    'irfftn',
    
]