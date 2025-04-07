"""
NumPy backend for ember_ml.

This module provides NumPy implementations of tensor operations.
"""

# Define the list of symbols to export
__all__ = [
    # Configuration variables
    'DEFAULT_DEVICE',
    'DEFAULT_DTYPE',
    
    # Ops classes removed from __all__
    # 'NumpyMathOps',
    # 'NumpyComparisonOps',
    # 'NumpyDeviceOps',
    # 'NumpySolverOps', # Old name
    # 'NumpyIOOps',
    # 'NumpyVectorOps',
    # 'NumpyLinearAlgOps',
    # 'NumpyStatsOps',
    # 'NumpyLossOps',
    # Math operations
    'add', 'subtract', 'multiply', 'divide', 'matmul', 'dot',
    'mean', 'sum', 'max', 'min', 'exp', 'log', 'log10', 'log2',
    'pow', 'sqrt', 'square', 'abs', 'sign', 'sin', 'cos', 'tan',
    'sinh', 'cosh', 'clip', # Removed activations: tanh, sigmoid, relu, softmax
    'var', 'negative', 'mod', 'floor_divide', 'sort', 'gradient',
    'cumsum', 'eigh', 'power',
    
    # Comparison operations
    'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal',
    'logical_and', 'logical_or', 'logical_not', 'logical_xor', 'allclose', 'isclose', 'all', 'where', 'isnan', # Added isnan
    # Device operations
    'to_device', 'get_device', 'get_available_devices', 'memory_usage',
    'memory_info', 'synchronize', 'set_default_device', 'get_default_device', 'is_available', # Added missing device ops
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
    'cosine_similarity', 'exponential_decay', # Removed gaussian

    # Loss Ops Functions (added)
    'mean_squared_error', 'mean_absolute_error', 'binary_crossentropy', 'categorical_crossentropy',
    'sparse_categorical_crossentropy', 'huber_loss', 'log_cosh_loss',

    # Feature Ops Functions (added)
    'pca', 'transform', 'inverse_transform', 'standardize', 'normalize',

    # Stats Ops Functions (added)
    'gaussian', # Moved gaussian here
    # 'std', 'median', # etc. (Add specific functions exported by numpy/stats.py)

    # Functional Tensor Ops (from tensor.ops - added)
    'cast', 'zeros', 'ones', 'eye', 'zeros_like', 'ones_like', 'full', 'full_like', 'arange', 'linspace',
    'reshape', 'transpose', 'concatenate', 'stack', 'split', 'expand_dims', 'squeeze', 'tile', 'pad',
    'slice', 'slice_update', 'gather', 'tensor_scatter_nd_update', 'scatter',
    'to_numpy', 'item', 'shape', 'dtype', 'copy', 'argsort', 'maximum',
    'random_normal', 'random_uniform', 'random_binomial', 'random_gamma', 'random_exponential',
    'random_poisson', 'random_categorical', 'random_permutation', 'shuffle', 'set_seed', 'get_seed',
     # Activation Ops Functions (added)
     'relu', 'sigmoid', 'tanh', 'softmax', 'softplus',
]
# Import configuration variables
from ember_ml.backend.numpy.config import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE
)

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
    cosh, # Removed tanh, sigmoid, relu, softmax from this import
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
   where,
   isnan # Added import
)

# Import specific functions from device_ops
from ember_ml.backend.numpy.device_ops import (
    to_device,
    get_device,
    get_available_devices,
    memory_usage,
   memory_info,
   synchronize,        # Added import
   set_default_device, # Added import
   get_default_device, # Added import
   is_available        # Added import
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

# Import specific functions from loss_ops (Added)
from ember_ml.backend.numpy.loss_ops import (
   mean_squared_error, mean_absolute_error, binary_crossentropy, categorical_crossentropy,
   sparse_categorical_crossentropy, huber_loss, log_cosh_loss
)

# Import specific functions from feature_ops (Added)
from ember_ml.backend.numpy.feature_ops import (
   pca, transform, inverse_transform, standardize, normalize
)

# Import specific functions from stats (Added - adjust based on actual exports)
from ember_ml.backend.numpy.stats import gaussian # Assuming stats/__init__ exports it or ops.py does
# from ember_ml.backend.numpy.stats import std, median # etc.

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
   exponential_decay,
   # gaussian # Removed import from vector_ops
)

# Import functional tensor ops directly (Added)
from ember_ml.backend.numpy.tensor.ops import (
   cast, zeros, ones, eye, zeros_like, ones_like, full, full_like, arange, linspace,
   reshape, transpose, concatenate, stack, split, expand_dims, squeeze, tile, pad,
   slice, slice_update, gather, tensor_scatter_nd_update, scatter,
   to_numpy, item, shape, dtype, copy, var as tensor_var, sort as tensor_sort, argsort, maximum, # aliased var/sort
   random_normal, random_uniform, random_binomial, random_gamma, random_exponential,
   random_poisson, random_categorical, random_permutation, shuffle, set_seed, get_seed
)
# Set power function
power = pow