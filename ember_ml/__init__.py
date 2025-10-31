"""Public package interface for Ember ML."""

from __future__ import annotations

from ember_ml.backend import (
    auto_select_backend,
    get_available_backends,
    get_backend,
    set_backend as _set_backend_impl,
    using_backend,
)
from ember_ml import ops
from ember_ml.tensor import (
    arange,
    array,
    cast,
    convert_to_tensor,
    copy,
    dtype,
    eye,
    full,
    maximum,
    ones,
    reshape,
    shape,
    to_numpy,
    transpose,
    zeros,
    # Dtypes
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    bool_,
)

# Lazily select a backend on import if none has been chosen yet.
if get_backend() is None:
    backend_name, _ = auto_select_backend()
    if backend_name is not None:
        set_backend(backend_name)


def set_seed(seed: int) -> None:
    """Seed random number generators for the active backend."""

    from ember_ml import tensor as tensor_module

    tensor_module.set_seed(seed)


def set_backend(backend: str) -> None:
    """Set the backend and update all cached operations."""
    # Use ops.set_backend which properly clears the cache
    ops.set_backend(backend)


# Create tensor alias
tensor = convert_to_tensor

# Specialized namespaces - these are dynamic and update automatically
linalg = ops.linearalg
stats = ops.stats
activations = ops.activations
bitwise = ops.bitwise
random = ops.random


def __getattr__(name: str):
    """
    Dynamically resolve operations from ops module.
    
    This allows operations like add, multiply, etc. to be accessed directly
    from ember_ml while still using the dynamic backend dispatch from ops.
    """
    # Try to get from ops first (for operations)
    if hasattr(ops, name):
        return getattr(ops, name)
    
    raise AttributeError(f"module 'ember_ml' has no attribute '{name}'")

__all__ = [
    # Backend
    "auto_select_backend",
    "get_available_backends",
    "get_backend",
    "ops",
    "set_backend",
    "set_seed",
    "using_backend",
    # Tensor creation and manipulation
    "arange",
    "array",
    "cast",
    "convert_to_tensor",
    "copy",
    "dtype",
    "eye",
    "full",
    "maximum",
    "ones",
    "reshape",
    "shape",
    "tensor",
    "to_numpy",
    "transpose",
    "zeros",
    # Dtypes
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool_",
    # Specialized namespaces
    "linalg",
    "stats",
    "activations",
    "bitwise",
    "random",
    # Math operations
    "add",
    "subtract",
    "multiply",
    "divide",
    "matmul",
    "dot",
    "abs",
    "sqrt",
    "square",
    "exp",
    "log",
    "log2",
    "log10",
    "pow",
    "power",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "negative",
    "floor",
    "ceil",
    "clip",
    "sign",
    "mod",
    "floor_divide",
    # Reduction operations
    "sum",
    "mean",
    "max",
    "min",
    "std",
    "var",
    "median",
    "percentile",
    "cumsum",
    # Comparison operations
    "equal",
    "not_equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "allclose",
    "isclose",
    "isnan",
    # Logical operations
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",
    "all",
    "any",
    # Activation functions
    "relu",
    "sigmoid",
    "softmax",
    "softplus",
    # Linear algebra
    "svd",
    "qr",
    "eigh",
    "solve",
]

__version__ = "0.2.0"
