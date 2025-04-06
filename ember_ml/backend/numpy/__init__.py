"""
NumPy backend for ember_ml.

This module provides NumPy implementations of tensor operations.
"""

# Define the list of symbols to export
__all__ = [
    # Configuration variables
    'DEFAULT_DEVICE',
    'DEFAULT_DTYPE',
    
    # Ops classes
    'NumpyMathOps',
    'NumpyComparisonOps',
    'NumpyDeviceOps',
    'NumpySolverOps',
    'NumpyIOOps',
    
    'NumpyVectorOps',
    # Linear Algebra class
    'NumpyLinearAlgOps',
    
    # Stats class
    'NumpyStatsOps',
    
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
    'memory_info',
    
    # Linear Algebra operations
    'solve', 'inv', 'svd', 'eig', 'eigvals', 'det', 'norm', 'qr',
    'cholesky', 'lstsq', 'diag', 'diagonal',
    
    # I/O operations
    'save', 'load'
    
    # Vector operations
    'fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
    'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
    'normalize_vector', 'compute_energy_stability', 'compute_interference_strength',
    'compute_phase_coherence', 'partial_interference', 'euclidean_distance',
    'cosine_similarity', 'exponential_decay', 'gaussian',
]

# Import configuration variables
from ember_ml.backend.numpy.config import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE
)


# Import all ops classes
from ember_ml.backend.numpy.math_ops import NumpyMathOps
from ember_ml.backend.numpy.comparison_ops import NumpyComparisonOps
from ember_ml.backend.numpy.device_ops import NumpyDeviceOps
from ember_ml.backend.numpy.linearalg import NumpyLinearAlgOps
from ember_ml.backend.numpy.io_ops import NumpyIOOps
from ember_ml.backend.numpy.solver_ops import NumpySolverOps
from ember_ml.backend.numpy.stats import NumpyStatsOps
from ember_ml.backend.numpy.vector_ops import NumpyVectorOps

# Import specific functions from math_ops
from ember_ml.backend.numpy.math_ops import (
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
from ember_ml.backend.numpy.comparison_ops import (
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
from ember_ml.backend.numpy.device_ops import (
    to_device,
    get_device,
    get_available_devices,
    memory_usage,
    memory_info
)

# Import specific functions from linearalg
from ember_ml.backend.numpy.linearalg import (
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
from ember_ml.backend.numpy.io_ops import (
    save,
    load
)


from ember_ml.backend.numpy.vector_ops import (

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
    irfftn,
    normalize_vector,
    compute_energy_stability,
    compute_interference_strength,
    compute_phase_coherence,
    partial_interference,
    euclidean_distance,
    cosine_similarity,
    exponential_decay
)

# Set power function
power = pow