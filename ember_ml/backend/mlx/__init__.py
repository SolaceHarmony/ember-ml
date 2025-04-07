"""
MLX backend for ember_ml.

This module provides MLX implementations of tensor operations.
"""

# Import configuration variables
from ember_ml.backend.mlx.config import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE
)

# Import tensor classes
from ember_ml.backend.mlx.tensor import MLXDType, MLXTensor

# Import all ops classes
# Removed Ops class imports
# from ember_ml.backend.mlx.math_ops import MLXMathOps
# from ember_ml.backend.mlx.comparison_ops import MLXComparisonOps
# from ember_ml.backend.mlx.device_ops import MLXDeviceOps
# from ember_ml.backend.mlx.linearalg import MLXLinearAlgOps
# from ember_ml.backend.mlx.io_ops import MLXIOOps
# from ember_ml.backend.mlx.vector_ops import MLXVectorOps
# from ember_ml.backend.mlx.stats import MLXStatsOps
# from ember_ml.backend.mlx.loss_ops import MLXLossOps

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
# Import specific functions from io_ops
from ember_ml.backend.mlx.io_ops import (
    save,
    load
)

# Import specific functions from vector_ops
from ember_ml.backend.mlx.vector_ops import (
    normalize_vector,
    compute_energy_stability,
    compute_interference_strength,
    compute_phase_coherence,
    partial_interference,
    euclidean_distance,
    cosine_similarity,
    exponential_decay,
    fft,
    ifft,
    fft2,
    ifft2,
    fftn,
    ifftn,
    rfft,
    irfft,
    rfft2,
    irfft2,
    rfftn,
    irfftn
)

# Import activation functions (Added)
from ember_ml.backend.mlx.activations import relu, sigmoid, tanh, softmax, softplus
# Set power function
power = pow

# Define the list of symbols to export
__all__ = [
    # Configuration variables
    'DEFAULT_DEVICE',
    'DEFAULT_DTYPE',

    # Ops classes removed from export
    # 'MLXMathOps',
    # 'MLXComparisonOps',
    # 'MLXDeviceOps',
    # 'MLXLinearAlgOps',
    # 'MLXIOOps',
    # 'MLXVectorOps',
    # 'MLXStatsOps',
    # 'MLXLossOps',

    # Tensor classes removed from export
    # 'MLXDType',
    # 'MLXTensor',

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
    'negative',
    'clip',
    'var',
    'mod',
    'floor_divide',
    'sort',
    'gradient',
    'cumsum',

    # Casting operations
    'cast',
    
    # Creation operations
    'zeros',
    'ones',
    'zeros_like',
    'ones_like',
    'eye',
    'full',
    'full_like',
    'arange',
    'linspace',
    
    # Indexing operations
    'slice_tensor',
    'slice_update',
    'gather',
    'tensor_scatter_nd_update',
    'scatter',
    'scatter_add', 
    'scatter_max', 
    'scatter_min', 
    'scatter_mean', 
    'scatter_softmax',
    
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
    
    # I/O operations
    'save',
    'load',
    
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

    # Activation Ops Functions (added)
    'relu', 'sigmoid', 'tanh', 'softmax', 'softplus',
]