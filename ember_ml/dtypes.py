"""
Data types for ember_ml.nn.tensor.

This module provides a backend-agnostic data type system that can be used
across different backends.
"""

from typing import Any, Union, Optional

from ember_ml.backend import get_backend, get_backend_module
from ember_ml.backend.registry import BackendRegistry

# Cache for backend instances
_CURRENT_INSTANCES = {}

def _get_backend_module():
    """Get the current backend module."""
    return get_backend_module()

###############################################################################
# Deprecation note:
#   EmberDType has been removed. Dtypes are now represented by plain strings
#   (e.g. "float32") or backend-native dtype objects. Public helpers below
#   provide minimal conversion utilities. Any legacy code importing EmberDType
#   should migrate to using string constants.
###############################################################################

# Removed EmberDType abstraction: canonical dtypes are plain strings.

# Get the backend dtype class
def _get_backend_dtype():
    """Get the dtype helper instance for the active backend.

    Uses the centralized ``BackendRegistry`` to fetch the already-imported
    backend module, then attempts to retrieve a conventional ``dtype`` or
    ``dtypes`` attribute. Falls back gracefully to a lightweight proxy that
    simply returns string names if a structured dtype helper is unavailable.
    """
    backend_name = get_backend()
    registry = BackendRegistry()
    _backend_name, module = registry.get_backend()
    if module is None:
        raise RuntimeError("Backend module not initialized (no active backend module in registry).")

    # Common attribute patterns (prefer explicit helper objects)
    candidate_attr_names = [
        'dtype',      # e.g., module-level accessor object
        'dtypes',     # plural variant
        'DType',      # class or factory
        'dtype_ops',  # legacy pattern
    ]

    for attr in candidate_attr_names:
        if hasattr(module, attr):
            helper = getattr(module, attr)
            # If it's a class, instantiate (best-effort, no-arg only)
            try:
                if isinstance(helper, type):  # class
                    return helper()  # type: ignore
            except Exception:  # pragma: no cover - defensive
                pass
            return helper

    # Fallback proxy provides minimal attribute-based access
    class _FallbackDTypeProxy:
        def __getattr__(self, name: str):  # pragma: no cover - trivial
            return name

    return _FallbackDTypeProxy()

# Define a function to get a data type from the backend
def get_dtype(name: str):
    """Get backend-native dtype object for canonical name if possible.

    Falls back to the canonical string if backend helper lacks attribute.
    """
    helper = _get_backend_dtype()
    # Preferred: helper has attribute with same name
    if hasattr(helper, name):
        attr = getattr(helper, name)
        # Some helpers expose properties returning native dtype objects
        return attr
    # Next: generic get_dtype method
    if hasattr(helper, 'get_dtype'):
        try:
            return helper.get_dtype(name)  # type: ignore
        except Exception:
            pass
    return name  # Fallback to string

# Define a function to convert a dtype to a string
def to_dtype_str(dtype: Any) -> str:
    """Convert a dtype-like object to a canonical string.

    Accepts strings or backend-native dtype objects. Falls back to ``str``.
    """
    helper = _get_backend_dtype()
    if hasattr(helper, 'to_dtype_str'):
        try:
            return helper.to_dtype_str(dtype)  # type: ignore
        except Exception:
            pass
    if hasattr(dtype, 'name') and isinstance(getattr(dtype, 'name'), str):
        return getattr(dtype, 'name')  # e.g., numpy/torch dtype .name
    return str(dtype)

# Define a function to convert a string to a dtype
def from_dtype_str(dtype_str: str):
    """Convert a dtype string to backend-native dtype where possible."""
    helper = _get_backend_dtype()
    if hasattr(helper, 'from_dtype_str'):
        try:
            return helper.from_dtype_str(dtype_str)  # type: ignore
        except Exception:
            pass
    return get_dtype(dtype_str)

# Define a class to dynamically get data types from the backend
_CANONICAL_DTYPES = (
    'float32','float64','int32','int64','bool_',
    'int8','int16','uint8','uint16','uint32','uint64','float16'
)

def list_dtypes() -> tuple:
    """Return tuple of canonical dtype names."""
    return _CANONICAL_DTYPES

class _DTypeAccessor:
    """Attribute access returning canonical dtype strings.

    Accessing e.g. ``dtypes.float32`` returns the string ``"float32"``.
    """
    def __getattr__(self, name: str) -> str:  # pragma: no cover - trivial
        if name in _CANONICAL_DTYPES:
            return name
        raise AttributeError(name)

dtypes = _DTypeAccessor()

# Create a DType class with dynamic properties for each data type
dtype = dtypes  # Backwards alias
# Canonical string constants (legacy compatibility)
float32 = 'float32'
float64 = 'float64'
int32 = 'int32'
int64 = 'int64'
bool_ = 'bool_'
int8 = 'int8'
int16 = 'int16'
uint8 = 'uint8'
uint16 = 'uint16'
uint32 = 'uint32'
uint64 = 'uint64'
float16 = 'float16'
# Define a list of all available data types and functions
__all__ = [
    'dtype', 'dtypes', 'list_dtypes',
    'get_dtype', 'to_dtype_str', 'from_dtype_str',
    'float32', 'float64', 'int32', 'int64', 'bool_',
    'int8', 'int16', 'uint8', 'uint16', 'uint32', 'uint64', 'float16'
]