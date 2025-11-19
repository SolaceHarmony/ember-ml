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

from ember_ml import ops
from ember_ml.backend import (
    auto_select_backend,
    get_available_backends,
    get_backend,
    using_backend,
)
import importlib
from typing import Any, Optional
from ember_ml.dtypes import (
    get_dtype,
    to_dtype_str,
    from_dtype_str,
    float32,
    float64,
    int32,
    int64,
    bool_,
    int8,
    int16,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
)

# Cache for imported backend tensor ops modules
_BACKEND_TENSOR_OPS_MODULES = {}


def _get_backend_tensor_ops_module():
    """Dynamically import and return the tensor ops module for the current backend."""

    backend_name = get_backend()
    if backend_name not in _BACKEND_TENSOR_OPS_MODULES:
        try:
            module_path = f"ember_ml.backend.{backend_name}.tensor.ops"
            _BACKEND_TENSOR_OPS_MODULES[backend_name] = importlib.import_module(
                module_path
            )
        except ImportError as exc:  # pragma: no cover - defensive
            raise ImportError(
                f"Could not import tensor ops module for backend '{backend_name}': {exc}"
            )
    return _BACKEND_TENSOR_OPS_MODULES[backend_name]


zeros = lambda *args, **kwargs: _get_backend_tensor_ops_module().zeros(
    *args, **kwargs
)
ones = lambda *args, **kwargs: _get_backend_tensor_ops_module().ones(
    *args, **kwargs
)
zeros_like = lambda *args, **kwargs: _get_backend_tensor_ops_module().zeros_like(
    *args, **kwargs
)
ones_like = lambda *args, **kwargs: _get_backend_tensor_ops_module().ones_like(
    *args, **kwargs
)
eye = lambda *args, **kwargs: _get_backend_tensor_ops_module().eye(
    *args, **kwargs
)
arange = lambda *args, **kwargs: _get_backend_tensor_ops_module().arange(
    *args, **kwargs
)
linspace = lambda *args, **kwargs: _get_backend_tensor_ops_module().linspace(
    *args, **kwargs
)
nonzero = lambda *args, **kwargs: _get_backend_tensor_ops_module().nonzero(
    *args, **kwargs
)
full = lambda *args, **kwargs: _get_backend_tensor_ops_module().full(
    *args, **kwargs
)
full_like = lambda *args, **kwargs: _get_backend_tensor_ops_module().full_like(
    *args, **kwargs
)
reshape = lambda *args, **kwargs: _get_backend_tensor_ops_module().reshape(
    *args, **kwargs
)
transpose = lambda *args, **kwargs: _get_backend_tensor_ops_module().transpose(
    *args, **kwargs
)
concatenate = lambda *args, **kwargs: _get_backend_tensor_ops_module().concatenate(
    *args, **kwargs
)
stack = lambda *args, **kwargs: _get_backend_tensor_ops_module().stack(
    *args, **kwargs
)
split = lambda *args, **kwargs: _get_backend_tensor_ops_module().split(
    *args, **kwargs
)
split_tensor = lambda *args, **kwargs: _get_backend_tensor_ops_module().split_tensor(
    *args, **kwargs
)
expand_dims = lambda *args, **kwargs: _get_backend_tensor_ops_module().expand_dims(
    *args, **kwargs
)
squeeze = lambda *args, **kwargs: _get_backend_tensor_ops_module().squeeze(
    *args, **kwargs
)
tile = lambda *args, **kwargs: _get_backend_tensor_ops_module().tile(
    *args, **kwargs
)
gather = lambda *args, **kwargs: _get_backend_tensor_ops_module().gather(
    *args, **kwargs
)
scatter = lambda *args, **kwargs: _get_backend_tensor_ops_module().scatter(
    *args, **kwargs
)
tensor_scatter_nd_update = (
    lambda *args, **kwargs: _get_backend_tensor_ops_module().tensor_scatter_nd_update(
        *args, **kwargs
    )
)
index_update = lambda *args, **kwargs: _get_backend_tensor_ops_module().index_update(
    *args, **kwargs
)
slice_tensor = lambda *args, **kwargs: _get_backend_tensor_ops_module().slice_tensor(
    *args, **kwargs
)
slice_update = lambda *args, **kwargs: _get_backend_tensor_ops_module().slice_update(
    *args, **kwargs
)
shape = lambda *args, **kwargs: _get_backend_tensor_ops_module().shape(
    *args, **kwargs
)
cast = lambda *args, **kwargs: _get_backend_tensor_ops_module().cast(
    *args, **kwargs
)
copy = lambda *args, **kwargs: _get_backend_tensor_ops_module().copy(
    *args, **kwargs
)
pad = lambda *args, **kwargs: _get_backend_tensor_ops_module().pad(
    *args, **kwargs
)
item = lambda *args, **kwargs: _get_backend_tensor_ops_module().item(
    *args, **kwargs
)
to_numpy = lambda *args, **kwargs: _get_backend_tensor_ops_module().to_numpy(
    *args, **kwargs
)
tolist = lambda *args, **kwargs: getattr(
    _get_backend_tensor_ops_module(), "tolist", lambda x: x.tolist()
)(*args, **kwargs)
random_uniform = (
    lambda *args, **kwargs: _get_backend_tensor_ops_module().random_uniform(
        *args, **kwargs
    )
)
random_normal = (
    lambda *args, **kwargs: _get_backend_tensor_ops_module().random_normal(
        *args, **kwargs
    )
)
maximum = lambda *args, **kwargs: _get_backend_tensor_ops_module().maximum(
    *args, **kwargs
)


def random_bernoulli(*args, **kwargs):
    """Generates Bernoulli random values."""

    seed = kwargs.pop("seed", None)
    if seed is not None:
        set_seed(seed)
    ops_module = _get_backend_tensor_ops_module()
    func = getattr(
        ops_module,
        "random_binomial",
        getattr(ops_module, "random_bernoulli", None),
    )
    if func:
        return func(*args, **kwargs)
    raise AttributeError(
        f"Backend '{get_backend()}' tensor ops module does not have a 'random_binomial' or 'random_bernoulli' function."
    )


random_gamma = lambda *args, **kwargs: _get_backend_tensor_ops_module().random_gamma(
    *args, **kwargs
)
random_exponential = (
    lambda *args, **kwargs: _get_backend_tensor_ops_module().random_exponential(
        *args, **kwargs
    )
)
random_poisson = lambda *args, **kwargs: _get_backend_tensor_ops_module().random_poisson(
    *args, **kwargs
)
random_categorical = (
    lambda *args, **kwargs: _get_backend_tensor_ops_module().random_categorical(
        *args, **kwargs
    )
)
random_permutation = (
    lambda *args, **kwargs: _get_backend_tensor_ops_module().random_permutation(
        *args, **kwargs
    )
)
shuffle = lambda *args, **kwargs: _get_backend_tensor_ops_module().shuffle(
    *args, **kwargs
)
random_shuffle = (
    lambda *args, **kwargs: _get_backend_tensor_ops_module().random_shuffle(
        *args, **kwargs
    )
)
set_seed = lambda *args, **kwargs: _get_backend_tensor_ops_module().set_seed(
    *args, **kwargs
)
get_seed = lambda *args, **kwargs: _get_backend_tensor_ops_module().get_seed(
    *args, **kwargs
)
meshgrid = lambda *args, **kwargs: _get_backend_tensor_ops_module().meshgrid(
    *args, **kwargs
)


class Index:
    """Simple index helper returning the key when indexed."""

    def __getitem__(self, key):  # pragma: no cover - trivial
        return key


index = Index()


_convert_to_backend_tensor = lambda *args, **kwargs: getattr(
    importlib.import_module(
        f"ember_ml.backend.{get_backend()}.tensor.ops.utility"
    ),
    "_convert_to_tensor",
)(*args, **kwargs)


def dtype(obj: Any = None):
    """Dtype helper.

    - dtype(tensor_obj) -> backend dtype of tensor_obj
    - dtype("float32") -> backend-native dtype (or canonical string)
    - dtype() -> returns callable so dtype()("float32") works
    """
    if obj is None:
        return get_dtype
    if isinstance(obj, str):
        return get_dtype(obj)
    return _get_backend_tensor_ops_module().dtype(obj)


def _tensor_callable(data: Any, dtype: Any = None, device: Optional[str] = None) -> Any:
    """Core tensor creation helper used by callers and the `tensor` namespace."""

    return _convert_to_backend_tensor(data, dtype=dtype, device=device)


def convert_to_tensor(data: Any, dtype: Any = None, device: Optional[str] = None) -> Any:
    """Alias for the tensor callable for backward compatibility."""

    return _tensor_callable(data, dtype=dtype, device=device)


def array(data: Any, dtype: Any = None, device: Optional[str] = None) -> Any:
    """Shortcut alias for :func:`convert_to_tensor` to maintain legacy `ember_ml.array`."""

    return convert_to_tensor(data, dtype=dtype, device=device)


class TensorNamespace:
    """Callable namespace exposing tensor creation plus helper attributes."""

    def __call__(self, data: Any, dtype: Any = None, device: Optional[str] = None) -> Any:
        return _tensor_callable(data, dtype=dtype, device=device)

    def __getattr__(self, name: str) -> Any:
        if name in globals():
            value = globals()[name]
            if callable(value):
                return value
        raise AttributeError(f"'ember_ml.tensor' has no attribute '{name}'")


tensor = TensorNamespace()

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
training = _optional_module("ember_ml.training", "training")
from ember_ml import utils

# Lazily select a backend on import if none has been chosen yet.
if get_backend() is None:
    backend_name, _ = auto_select_backend()
    if backend_name is not None:
        ops.set_backend(backend_name)


def set_backend(backend: str, *, persist: bool = True) -> None:
    """Set the backend and update all cached operations."""
    ops.set_backend(backend)

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
    "array",
    "arange",
    "cast",
    "concatenate",
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
    "convert_to_tensor",
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
    "stats",
    "bitwise",
    "asyncml",  # Lazy-loaded async operations
    # Additional namespaces
    "training",
    "visualization",
    "wave",
    "utils"
]

__version__ = "0.2.0"
