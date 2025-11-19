"""Unified operations interface dispatching to the active backend."""

from __future__ import annotations

from typing import Any, Dict, Set

from ember_ml.backend import (
    auto_select_backend,
    get_backend,
    get_backend_module,
    set_backend as _set_backend,
)

_CACHE: Dict[str, Any] = {}
_RESOLVED_NAMES: Set[str] = set()


def set_backend(backend: str, *, persist: bool = True) -> None:
    _set_backend(backend, persist=persist)
    _clear_cached_ops()


def set_ops(backend: str) -> None:
    """Compatibility wrapper matching the historical ``ops.set_ops`` API."""

    set_backend(backend)


def _resolve(name: str) -> Any:
    module = get_backend_module()
    if not hasattr(module, name):
        raise AttributeError(f"module 'ember_ml.ops' has no attribute '{name}'")
    value = getattr(module, name)
    _CACHE[name] = value
    globals()[name] = value
    _RESOLVED_NAMES.add(name)
    return value


def __getattr__(name: str) -> Any:  # pragma: no cover - thin delegation
    if name in _CACHE:
        return _CACHE[name]
    return _resolve(name)


def __dir__() -> list[str]:  # pragma: no cover - helper for introspection
    attrs = {"set_backend", "get_backend", "auto_select_backend", "set_ops"}
    attrs.update(dir(get_backend_module()))
    return sorted(attrs)


class _RandomNamespace:
    """Attribute wrapper resolving ``ops.random`` helpers dynamically."""

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - trivial delegation
        return _resolve(f"random_{name}")


random = _RandomNamespace()


def _clear_cached_ops() -> None:
    for name in list(_RESOLVED_NAMES):
        _CACHE.pop(name, None)
        globals().pop(name, None)
        _RESOLVED_NAMES.discard(name)


__all__ = [
    "set_backend",
    "get_backend",
    "auto_select_backend",
    "set_ops",
    "random",
]
