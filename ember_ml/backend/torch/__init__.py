"""
PyTorch backend for ember_ml.

This module provides PyTorch implementations of the backend operations
required by ember_ml.
"""

# Import all components from their respective modules
from ember_ml.backend.torch.config import (
    __version__,
    has_gpu,
    has_mps,
    default_float_type,
    DEFAULT_DEVICE
)

# Import all ops classes
from ember_ml.backend.torch.tensor_ops import TorchTensorOps
from ember_ml.backend.torch.math_ops import TorchMathOps
from ember_ml.backend.torch.random_ops import TorchRandomOps
from ember_ml.backend.torch.comparison_ops import TorchComparisonOps
from ember_ml.backend.torch.device_ops import TorchDeviceOps
from ember_ml.backend.torch.dtype_ops import TorchDTypeOps
from ember_ml.backend.torch.solver_ops import TorchSolverOps
from ember_ml.backend.torch.io_ops import TorchIOOps

# Import all functions
from ember_ml.backend.torch.tensor_ops import (
    convert_to_tensor,
    zeros,
    ones,
    zeros_like,
    ones_like,
    eye,
    reshape,
    transpose,
    expand_dims,
    concatenate,
    stack,
    split,
    squeeze,
    tile,
    gather,
    shape,
    dtype,
    cast,
    copy,
    to_numpy,
    full,
    full_like,
    linspace,
    arange
)

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

from ember_ml.backend.torch.random_ops import (
    random_normal,
    random_uniform,
    random_binomial,
    random_permutation,
    set_seed,
    random_categorical,
    random_exponential,
    random_gamma,
    random_poisson,
    shuffle
)

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

from ember_ml.backend.torch.dtype_ops import (
    from_dtype_str as ember_dtype_to_torch,
    to_dtype_str as torch_to_ember_dtype
)
from ember_ml.backend.torch.solver_ops import (
    solve
)

from ember_ml.backend.torch.io_ops import (
    save,
    load
)

# Set power function
power = pow
power = pow
