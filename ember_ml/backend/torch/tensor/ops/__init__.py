"""
PyTorch tensor operations.

This module provides standalone functions for tensor operations using the PyTorch backend.
These functions can be called directly or through the TorchTensor class methods.
"""
# Import configuration variables
from ember_ml.backend.torch.config import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE
)

# Import tensor classes
from ember_ml.backend.torch.tensor import TorchDType, TorchTensor

# We'll use lazy imports for ops classes to avoid circular imports
def get_torch_math_ops():
    from ember_ml.backend.torch.math_ops import TorchMathOps
    return TorchMathOps

def get_torch_comparison_ops():
    from ember_ml.backend.torch.comparison_ops import TorchComparisonOps
    return TorchComparisonOps

def get_torch_device_ops():
    from ember_ml.backend.torch.device_ops import TorchDeviceOps
    return TorchDeviceOps

def get_torch_io_ops():
    from ember_ml.backend.torch.io_ops import TorchIOOps
    return TorchIOOps

def get_torch_linearalg_ops():
    from ember_ml.backend.torch.linearalg import TorchLinearAlgOps
    return TorchLinearAlgOps

# Import specific functions from math_ops
from ember_ml.backend.torch.math_ops import (
    add,
    subtract,
    multiply,
    divide,
    matmul,
    dot,
    mean,
    sum,
    max,
    min,
    exp,
    log,
    log10,
    log2,
    pow,
    sqrt,
    square,
    abs,
    sign,
    sin,
    cos,
    tan,
    sinh,
    cosh,
    tanh,
    sigmoid,
    relu,
    softmax,
    clip,
    var,
    negative,
    mod,
    floor_divide,
    sort,
    gradient,
    cumsum,
    eigh
)


# Import specific functions from comparison_ops
from ember_ml.backend.torch.comparison_ops import (
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
    logical_and,
    logical_or,
    logical_not,
    logical_xor,
    allclose,
    isclose,
    all,
    where
)

# Import specific functions from device_ops
from ember_ml.backend.torch.device_ops import (
    to_device,
    get_device,
    get_available_devices,
    memory_usage,
    memory_info,
    synchronize,
    set_default_device,
    get_default_device,
    is_available
)

# Import specific functions from io_ops
from ember_ml.backend.torch.io_ops import (
    save,
    load
)

# Set power function
power = pow

# Define the list of symbols to export
__all__ = [
    # Configuration variables
    'DEFAULT_DEVICE',
    'DEFAULT_DTYPE',

    # Ops classes getters
    'get_torch_math_ops',
    'get_torch_comparison_ops',
    'get_torch_device_ops',
    'get_torch_io_ops',
    'get_torch_linearalg_ops',

    # Tensor classes
    'TorchDType',
    'TorchTensor',

    # Math operations
    'add',
    'subtract',
    'multiply',
    'divide',
    'matmul',
    'dot',
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
    'sin',
    'cos',
    'tan',
    'sinh',
    'cosh',
    'tanh',
    'sigmoid',
    'relu',
    'softmax',
    'clip',
    'var',
    'negative',
    'mod',
    'floor_divide',
    'sort',
    'gradient',
    'cumsum',
    'eigh',
    
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
    
    # Device operations
    'to_device',
    'get_device',
    'get_available_devices',
    'memory_usage',
    'memory_info',
    'synchronize',
    'set_default_device',
    'get_default_device',
    'is_available',
    
    # IO operations
    'save',
    'load',
    
    # Additional operations
    'power'
]