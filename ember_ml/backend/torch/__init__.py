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
    DEFAULT_DTYPE,
    DEFAULT_DEVICE
)

# Import tensor classes
from ember_ml.backend.torch.tensor import TorchDType, TorchTensor

# Import ops classes
from ember_ml.backend.torch.math_ops import TorchMathOps
from ember_ml.backend.torch.comparison_ops import TorchComparisonOps
from ember_ml.backend.torch.device_ops import TorchDeviceOps
from ember_ml.backend.torch.solver_ops import TorchSolverOps
from ember_ml.backend.torch.io_ops import TorchIOOps

# Tensor operations are now methods of the TorchTensor class
# We no longer import them from tensor_ops

# Import math operations
from ember_ml.backend.torch.math_ops import (
    add, subtract, multiply, divide, matmul, dot, mean, sum, max, min,
    exp, log, log10, log2, pow, sqrt, square, abs, sign, sin, cos, tan,
    sinh, cosh, tanh, sigmoid, relu, softmax, clip, var, pi
)

# Import comparison operations
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

# Import device operations
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

# DType operations are now methods of the TorchDType class
from ember_ml.backend.torch.solver_ops import (
    solve
)

from ember_ml.backend.torch.io_ops import (
    save,
    load
)

# Set power function
power = pow
