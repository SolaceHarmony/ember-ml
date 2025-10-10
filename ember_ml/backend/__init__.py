"""Backend selection and management utilities."""

from __future__ import annotations

import importlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple

_BACKEND_MODULES: Dict[str, str] = {
    "numpy": "ember_ml.backend.numpy",
    "torch": "ember_ml.backend.torch",
    "mlx": "ember_ml.backend.mlx",
}

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "backend_config.json"


def load_backend_config() -> Dict[str, bool]:
    """Load optional backend configuration from ``backend_config.json``."""

    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        data = {}
    except json.JSONDecodeError:
        data = {}
    config: Dict[str, bool] = {name: True for name in _BACKEND_MODULES}
    for name, enabled in data.items():
        if name in config:
            config[name] = bool(enabled)
    return config


_CONFIG = load_backend_config()


def _module_path(backend: str) -> str:
    try:
        return _BACKEND_MODULES[backend]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown backend '{backend}'.") from exc


def _try_import_backend(backend: str) -> Optional[ModuleType]:
    path = _module_path(backend)
    try:
        return importlib.import_module(path)
    except ImportError:
        return None


def _import_backend(backend: str) -> ModuleType:
    module = _try_import_backend(backend)
    if module is None:
        raise ValueError(f"Backend '{backend}' is not available.")
    return module


def _candidate_order() -> List[str]:
    preferred = [name for name, enabled in _CONFIG.items() if enabled and name in _BACKEND_MODULES]
    for name in _BACKEND_MODULES:
        if name not in preferred:
            preferred.append(name)
    return preferred


def _discover_available_backends() -> List[str]:
    available: List[str] = []
    for name in _candidate_order():
        if name in available:
            continue
        if _try_import_backend(name) is not None:
            available.append(name)
    return available


_AVAILABLE_BACKENDS: List[str] = _discover_available_backends()
_CURRENT_BACKEND_NAME: Optional[str] = None
_CURRENT_BACKEND_MODULE: Optional[ModuleType] = None


def get_available_backends() -> List[str]:
    """Return the list of backends importable in this environment."""

    return list(_AVAILABLE_BACKENDS)


def auto_select_backend() -> Tuple[Optional[str], Optional[str]]:
    """Choose the first configured backend that can be imported."""

    for name in _candidate_order():
        module = _try_import_backend(name)
        if module is None:
            continue
        device: Optional[str] = None
        if name == "torch":
            device = _detect_torch_device(module)
        elif name == "numpy":
            device = "cpu"
        return name, device
    return None, None


def set_backend(backend: str) -> None:
    """Activate ``backend`` as the current backend."""

    global _CURRENT_BACKEND_NAME, _CURRENT_BACKEND_MODULE

    module = _import_backend(backend)
    _CURRENT_BACKEND_NAME = backend
    _CURRENT_BACKEND_MODULE = module
    if backend not in _AVAILABLE_BACKENDS:
        _AVAILABLE_BACKENDS.append(backend)


def get_backend() -> Optional[str]:
    """Return the active backend name, auto-selecting if unset."""

    if _CURRENT_BACKEND_NAME is not None:
        return _CURRENT_BACKEND_NAME
    backend, _ = auto_select_backend()
    if backend is not None:
        set_backend(backend)
    return _CURRENT_BACKEND_NAME


def get_backend_module() -> ModuleType:
    """Return the module implementing the active backend."""

    global _CURRENT_BACKEND_MODULE

    backend = get_backend()
    if backend is None or _CURRENT_BACKEND_MODULE is None:
        raise RuntimeError("No backend is currently selected.")
    return _CURRENT_BACKEND_MODULE


def get_device(*args, **kwargs):
    """Delegate to the active backend's ``get_device`` helper."""

    module = get_backend_module()
    if not hasattr(module, "get_device"):
        raise AttributeError(f"Backend '{get_backend()}' does not expose get_device().")
    return getattr(module, "get_device")(*args, **kwargs)


def set_device(*args, **kwargs) -> None:
    """Delegate to the active backend's ``set_device`` helper when available."""

    module = get_backend_module()
    if not hasattr(module, "set_device"):
        raise AttributeError(f"Backend '{get_backend()}' does not expose set_device().")
    getattr(module, "set_device")(*args, **kwargs)


@contextmanager
def using_backend(backend: str):
    """Context manager that temporarily switches the active backend."""

    original = get_backend()
    if backend != original:
        set_backend(backend)
    try:
        yield
    finally:
        if original is not None and original != backend:
            set_backend(original)


def _detect_torch_device(module: ModuleType) -> str:
    try:
        cuda_available = bool(getattr(module, "cuda", None) and module.cuda.is_available())
    except Exception:  # pragma: no cover - defensive
        cuda_available = False
    if cuda_available:
        return "cuda"
    try:
        mps = getattr(getattr(module, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - defensive
        pass
    return "cpu"


__all__ = [
    "auto_select_backend",
    "get_available_backends",
    "get_backend",
    "get_backend_module",
    "get_device",
    "load_backend_config",
    "set_backend",
    "set_device",
    "using_backend",
    "_AVAILABLE_BACKENDS",
]
