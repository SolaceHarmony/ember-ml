"""
MLX backend for ember_ml.

This module provides MLX implementations of tensor operations.
"""

# Define the list of symbols to export
__all__ = [
    # Configuration variables
    'DEFAULT_DEVICE',
    'DEFAULT_DTYPE',
    
    # Ops classes
    'MLXMathOps',
    'MLXComparisonOps',
    'MLXDeviceOps',
    'MLXSolverOps',
    'MLXIOOps',
    
    # Tensor classes
    'MLXDType',
    'MLXTensor',
    
    # Math operations
    'add', 'subtract', 'multiply', 'divide', 'matmul', 'dot',
    'mean', 'sum', 'max', 'min', 'exp', 'log', 'log10', 'log2',
    'pow', 'sqrt', 'square', 'abs', 'sign', 'sin', 'cos', 'tan',
    'sinh', 'cosh', 'tanh', 'sigmoid', 'relu', 'softmax', 'clip',
    'var', 'negative', 'mod', 'floor_divide', 'sort', 'gradient',
    'cumsum', 'eigh', 'power',
    
    # Comparison operations
    'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal',
    'logical_and', 'logical_or', 'logical_not', 'logical_xor',
    'allclose', 'isclose', 'all', 'where',
    
    # Device operations
    'to_device', 'get_device', 'get_available_devices', 'memory_usage',
    'memory_info', 'synchronize', 'set_default_device', 'get_default_device',
    'is_available',
    
    # Solver operations
    'solve', 'inv', 'svd', 'eig', 'eigvals', 'det', 'norm', 'qr',
    'cholesky', 'lstsq',
    
    # I/O operations
    'save', 'load'
]

# Import configuration variables
from ember_ml.backend.mlx.config import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE
)

# Import tensor classes
from ember_ml.backend.mlx.tensor import MLXDType, MLXTensor

# Import all ops classes
from ember_ml.backend.mlx.math_ops import MLXMathOps
from ember_ml.backend.mlx.comparison_ops import MLXComparisonOps
from ember_ml.backend.mlx.device_ops import MLXDeviceOps
from ember_ml.backend.mlx.solver_ops import MLXSolverOps
from ember_ml.backend.mlx.io_ops import MLXIOOps

# Import specific functions from math_ops
from ember_ml.backend.mlx.math_ops import (
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

# Random operations are now methods of the MLXTensor class

# Import specific functions from comparison_ops
from ember_ml.backend.mlx.comparison_ops import (
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
from ember_ml.backend.mlx.device_ops import (
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

# Import specific functions from solver_ops
from ember_ml.backend.mlx.solver_ops import (
    solve,
    inv,
    svd,
    eig,
    eigvals,
    det,
    norm,
    qr,
    cholesky,
    lstsq
)

# Import specific functions from io_ops
from ember_ml.backend.mlx.io_ops import (
    save,
    load
)

# Set power function
power = pow

# No need to add to __all__ as it's already included in the list