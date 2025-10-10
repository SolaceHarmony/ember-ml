"""Backend-agnostic dtype helpers."""

from __future__ import annotations

from typing import Any

from ember_ml.backend import get_backend_module

_CANONICAL_DTYPES = (
    "float32",
    "float64",
    "int32",
    "int64",
    "bool_",
    "int8",
    "int16",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
)


def _backend_dtype_helper() -> Any:
    module = get_backend_module()
    for attr in ("dtype", "dtypes", "DType", "dtype_ops"):
        if not hasattr(module, attr):
            continue
        helper = getattr(module, attr)
        if isinstance(helper, type):  # class helper without state
            try:
                return helper()  # type: ignore[call-arg]
            except Exception:  # pragma: no cover - defensive
                continue
        return helper
    return None


def get_dtype(name: str):
    """Return the backend-native dtype for ``name`` when available."""

    helper = _backend_dtype_helper()
    if helper is None:
        return name
    if hasattr(helper, name):
        return getattr(helper, name)
    if hasattr(helper, "get_dtype"):
        try:
            return helper.get_dtype(name)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - backend-specific errors
            pass
    return name


def to_dtype_str(dtype: Any) -> str:
    """Convert ``dtype`` to a canonical string when possible."""

    helper = _backend_dtype_helper()
    if helper is not None and hasattr(helper, "to_dtype_str"):
        try:
            return helper.to_dtype_str(dtype)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - backend-specific errors
            pass
    if hasattr(dtype, "name") and isinstance(getattr(dtype, "name"), str):
        return getattr(dtype, "name")
    return str(dtype)


def from_dtype_str(dtype_str: str):
    """Convert ``dtype_str`` to backend-native dtype when supported."""

    helper = _backend_dtype_helper()
    if helper is not None and hasattr(helper, "from_dtype_str"):
        try:
            return helper.from_dtype_str(dtype_str)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - backend-specific errors
            pass
    return get_dtype(dtype_str)


def list_dtypes() -> tuple[str, ...]:
    """Return the canonical dtype names supported by Ember ML."""

    return _CANONICAL_DTYPES


class _DTypeAccessor:
    def __getattr__(self, name: str) -> str:  # pragma: no cover - simple mapping
        if name in _CANONICAL_DTYPES:
            return name
        raise AttributeError(name)


dtypes = _DTypeAccessor()
dtype = dtypes  # legacy alias

float32 = "float32"
float64 = "float64"
int32 = "int32"
int64 = "int64"
bool_ = "bool_"
int8 = "int8"
int16 = "int16"
uint8 = "uint8"
uint16 = "uint16"
uint32 = "uint32"
uint64 = "uint64"
float16 = "float16"

__all__ = [
    "dtype",
    "dtypes",
    "from_dtype_str",
    "get_dtype",
    "list_dtypes",
    "to_dtype_str",
    "float32",
    "float64",
    "int32",
    "int64",
    "bool_",
    "int8",
    "int16",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
]
