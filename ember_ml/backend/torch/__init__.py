"""
PyTorch backend for ember_ml.

This module provides PyTorch implementations of tensor operations.
"""

# Define the list of symbols to export
__all__ = [
    # Configuration variables
    'DEFAULT_DEVICE',
    'DEFAULT_DTYPE',
    
    # Ops classes
    'TorchMathOps',
    'TorchComparisonOps',
    'TorchDeviceOps',
    'TorchIOOps',
    'TorchLinearAlgOps',
    
    # Math operations
    'add', 'subtract', 'multiply', 'divide', 'matmul', 'dot',
    'mean', 'sum', 'max', 'min', 'exp', 'log', 'log10', 'log2',
    'pow', 'sqrt', 'square', 'abs', 'sign', 'sin', 'cos', 'tan',
    'sinh', 'cosh', 'tanh', 'sigmoid', 'relu', 'softmax', 'clip',
    'var', 'pi', 'power',
    
    # Comparison operations
    'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal',
    'logical_and', 'logical_or', 'logical_not', 'logical_xor',
    
    # Device operations
    'to_device', 'get_device', 'get_available_devices', 'memory_usage',
    'memory_info', 'synchronize', 'set_default_device', 'get_default_device',
    'is_available',
    
    # Linear Algebra operations
    'solve', 'inv', 'svd', 'eig', 'eigvals', 'det', 'norm', 'qr',
    'cholesky', 'lstsq', 'diag', 'diagonal',
    
    # I/O operations
    'save', 'load'
]

# Import configuration variables
from ember_ml.backend.torch.config import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE
)

# Import tensor classes
from ember_ml.backend.torch.tensor import TorchDType, TorchTensor

# Import all ops classes
from ember_ml.backend.torch.math_ops import TorchMathOps
from ember_ml.backend.torch.comparison_ops import TorchComparisonOps
from ember_ml.backend.torch.device_ops import TorchDeviceOps
from ember_ml.backend.torch.io_ops import TorchIOOps
from ember_ml.backend.torch.linearalg import TorchLinearAlgOps

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
    pi
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
    logical_xor
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

# Import specific functions from linearalg
from ember_ml.backend.torch.linearalg import (
    solve,
    inv,
    svd,
    eig,
    eigvals,
    det,
    norm,
    qr,
    cholesky,
    lstsq,
    diag,
    diagonal
)

# Import specific functions from io_ops
from ember_ml.backend.torch.io_ops import (
    save,
    load
)

# Set power function
power = pow
