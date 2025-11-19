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
from ember_ml import training

def _optional_module(import_path: str, placeholder_name: str):
    try:
        module = __import__(import_path, fromlist=["*"])
    except Exception as exc:  # pragma: no cover - optional path
        class _UnavailableModule:
            def __getattr__(self, item):
                raise ImportError(
                    f"{import_path} requires optional dependencies that are not available."
                ) from exc
        return _UnavailableModule()
    return module

visualization = _optional_module("ember_ml.visualization", "visualization")
wave = _optional_module("ember_ml.wave", "wave")
from ember_ml import utils

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

# Lazy-loaded modules
_asyncml = None


def __getattr__(name: str):
    """
    Dynamically resolve operations, namespaces, and lazy modules.

    This enables:
    1. Flat API where operations like add, multiply, matmul can be accessed directly
       from ember_ml (e.g., `ember.add`, `ember.tensor`)
    2. Specialized namespaces (linalg, stats, etc.) with automatic backend dispatch
    3. Lazy loading of optional dependencies (asyncml requires ray)

    The namespace mapping handles differences between public API and internal ops
    module (e.g., 'linalg' -> 'linearalg').

    Args:
        name: The attribute name being accessed from the ember_ml module.

    Returns:
        The requested operation function, namespace module, or lazy-loaded module.

    Raises:
        AttributeError: If the requested attribute doesn't exist.
        ImportError: If lazy-loaded dependency is missing (e.g., ray for asyncml).

    Examples:
        >>> import ember_ml as ember
        >>> ember.add(ember.array([1, 2]), ember.array([3, 4]))
        >>> ember.linalg.svd(ember.array([[1, 2], [3, 4]]))
        >>> await ember.asyncml.compute(...)  # Requires ray
    """
    global _asyncml

    # Handle lazy-loaded asyncml module
    if name == 'asyncml':
        if _asyncml is None:
            try:
                import ember_ml.asyncml as _asyncml_module
                _asyncml = _asyncml_module
            except ImportError:
                raise ImportError(
                    "Could not import ember_ml.asyncml. "
                    "Make sure 'ray' is installed for async distributed operations."
                )
        return _asyncml

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
    "asyncml",  # Lazy-loaded async operations
    # Additional namespaces
    "training",
    "visualization",
    "wave",
    "utils",
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
