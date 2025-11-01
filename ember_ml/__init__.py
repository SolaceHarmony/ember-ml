"""Public package interface for Ember ML.

This module provides a flat, PyTorch-like API where common operations are accessible
at the top level. Most operations (math, reduction, comparison, logical, activation,
and linear algebra functions) are dynamically resolved via __getattr__ from the ops
module, ensuring automatic backend dispatch and preventing stale references after
backend switching.

Explicitly defined exports include tensor creation functions, dtype constants, and
backend management utilities. The __all__ list includes both explicitly defined
attributes and those dynamically resolved via __getattr__ for complete API documentation.
"""

from __future__ import annotations

from ember_ml.backend import (
    auto_select_backend,
    get_available_backends,
    get_backend,
    using_backend,
)
from ember_ml import ops
from ember_ml.tensor import (
    arange,
    array,
    cast,
    concatenate,
    convert_to_tensor,
    copy,
    dtype,
    eye,
    full,
    maximum,
    ones,
    ones_like,
    reshape,
    shape,
    stack,
    to_numpy,
    transpose,
    zeros,
    zeros_like,
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

# Namespace mapping for dynamic attribute resolution
# Maps public API names to internal ops module names
_NAMESPACE_MAPPING = {
    'linalg': 'linearalg',  # Note: ops uses "linearalg" spelling
    'stats': 'stats',
    'activations': 'activations',
    'bitwise': 'bitwise',
    'random': 'random',
}


def __getattr__(name: str):
    """
    Dynamically resolve operations and namespaces from ops module.
    
    This enables a flat API where operations like add, multiply, matmul, etc. can be
    accessed directly from ember_ml (e.g., `em.add`) while still using the dynamic
    backend dispatch mechanism from the ops module.
    
    Specialized namespaces (linalg, stats, activations, bitwise, random) are also
    dynamically resolved on each access to avoid stale references after backend
    switching. The namespace mapping handles name differences between the public API
    and internal ops module (e.g., 'linalg' -> 'linearalg').
    
    Args:
        name: The attribute name being accessed from the ember_ml module.
    
    Returns:
        The requested operation function or namespace module from ops.
    
    Raises:
        AttributeError: If the requested attribute doesn't exist in the ops module.
    
    Examples:
        >>> import ember_ml as em
        >>> em.add(em.array([1, 2]), em.array([3, 4]))
        >>> em.linalg.svd(em.array([[1, 2], [3, 4]]))
    """
    # Handle specialized namespaces dynamically
    if name in _NAMESPACE_MAPPING:
        ops_name = _NAMESPACE_MAPPING[name]
        return getattr(ops, ops_name)
    
    # Try to get from ops for operations
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
    "concatenate",
    "convert_to_tensor",
    "copy",
    "dtype",
    "eye",
    "full",
    "maximum",
    "ones",
    "ones_like",
    "reshape",
    "shape",
    "stack",
    "tensor",
    "to_numpy",
    "transpose",
    "zeros",
    "zeros_like",
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
    # Specialized namespaces (dynamically resolved via __getattr__)
    "linalg",
    "stats",
    "activations",
    "bitwise",
    "random",
    # Math operations (dynamically resolved via __getattr__)
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
    # Reduction operations (dynamically resolved via __getattr__)
    "sum",
    "mean",
    "max",
    "min",
    "std",
    "var",
    "median",
    "percentile",
    "cumsum",
    # Comparison operations (dynamically resolved via __getattr__)
    "equal",
    "not_equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "allclose",
    "isclose",
    "isnan",
    # Logical operations (dynamically resolved via __getattr__)
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",
    "all",
    "any",
    # Activation functions (dynamically resolved via __getattr__)
    "relu",
    "sigmoid",
    "softmax",
    "softplus",
    # Linear algebra (dynamically resolved via __getattr__)
    "svd",
    "qr",
    "eigh",
    "solve",
]

__version__ = "0.2.0"
