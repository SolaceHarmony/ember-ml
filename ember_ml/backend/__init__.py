"""Backend selection and management utilities."""

from __future__ import annotations

import importlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple

_BACKEND_MODULES: Dict[str, str] = {
    "torch": "ember_ml.backend.torch",
    "mlx": "ember_ml.backend.mlx",
    "numpy": "ember_ml.backend.numpy",
}

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "backend_config.json"

_BACKEND_STATE_DIR = Path(__file__).resolve().parent
_BACKEND_PERSIST_FILE = _BACKEND_STATE_DIR / ".backend"
_DEVICE_PERSIST_FILE = _BACKEND_STATE_DIR / ".device"
_USER_CONFIG_FILE = Path.home() / ".ember_ml" / "config"


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


def _read_text_file(path: Path) -> Optional[str]:
    try:
        text = path.read_text().strip()
    except FileNotFoundError:
        return None
    return text or None


def _load_env_override() -> Tuple[Optional[str], Optional[str]]:
    backend = os.environ.get("ember_ml_BACKEND") or os.environ.get("EMBER_ML_BACKEND")
    device = os.environ.get("ember_ml_DEVICE") or os.environ.get("EMBER_ML_DEVICE")
    return (backend.strip(), device.strip()) if backend else (None, None)


def _read_user_config_override() -> Tuple[Optional[str], Optional[str]]:
    if not _USER_CONFIG_FILE.exists():
        return None, None
    backend = None
    device = None
    try:
        with _USER_CONFIG_FILE.open("r", encoding="utf-8") as handle:
            for line in handle:
                if "=" not in line:
                    continue
                key, value = map(str.strip, line.split("=", 1))
                if not key or not value:
                    continue
                lower_key = key.lower()
                if lower_key == "backend":
                    backend = value
                elif lower_key == "device":
                    device = value
    except Exception:
        return None, None
    return backend, device


def _read_persisted_override() -> Tuple[Optional[str], Optional[str]]:
    backend = _read_text_file(_BACKEND_PERSIST_FILE)
    device = _read_text_file(_DEVICE_PERSIST_FILE)
    return backend, device


def _persist_backend_choice(backend: str, device: Optional[str]) -> None:
    try:
        _BACKEND_PERSIST_FILE.write_text(backend, encoding="utf-8")
    except OSError:
        pass
    if device:
        try:
            _DEVICE_PERSIST_FILE.write_text(device, encoding="utf-8")
        except OSError:
            pass


def _default_device_for_backend(backend: str, module: ModuleType) -> Optional[str]:
    if backend == "torch":
        return _detect_torch_device(module)
    if backend == "mlx":
        get_device_func = getattr(module, "get_device", None)
        if callable(get_device_func):
            try:
                return get_device_func()
            except Exception:
                return None
    if backend == "numpy":
        return "cpu"
    return None


def _override_candidates() -> List[Tuple[str, Optional[str]]]:
    overrides: List[Tuple[str, Optional[str]]] = []
    entries = [
        _load_env_override(),
        _read_persisted_override(),
        _read_user_config_override(),
    ]
    seen: set[str] = set()
    for backend_name, device in entries:
        if not backend_name:
            continue
        backend_name = backend_name.strip()
        if not backend_name or backend_name in seen:
            continue
        overrides.append((backend_name, device))
        seen.add(backend_name)
    return overrides


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

    seen: set[str] = set()
    for backend_name, device_override in _override_candidates():
        if backend_name not in _BACKEND_MODULES:
            continue
        module = _try_import_backend(backend_name)
        if module is None:
            continue
        seen.add(backend_name)
        return backend_name, device_override or _default_device_for_backend(
            backend_name, module
        )

    for backend_name in _candidate_order():
        if backend_name in seen:
            continue
        if not _CONFIG.get(backend_name, False):
            continue
        module = _try_import_backend(backend_name)
        if module is None:
            continue
        seen.add(backend_name)
        return backend_name, _default_device_for_backend(backend_name, module)

    return None, None


def set_backend(backend: str, *, persist: bool = True) -> None:
    """Activate ``backend`` as the current backend."""

    global _CURRENT_BACKEND_NAME, _CURRENT_BACKEND_MODULE

    module = _import_backend(backend)
    _CURRENT_BACKEND_NAME = backend
    _CURRENT_BACKEND_MODULE = module
    if backend not in _AVAILABLE_BACKENDS:
        _AVAILABLE_BACKENDS.append(backend)
    if persist:
        device_value = _default_device_for_backend(backend, module)
        _persist_backend_choice(backend, device_value)


def get_backend() -> Optional[str]:
    """Return the active backend name, auto-selecting if unset."""

    if _CURRENT_BACKEND_NAME is not None:
        return _CURRENT_BACKEND_NAME
    backend, _ = auto_select_backend()
    if backend is not None:
        set_backend(backend, persist=False)
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
        set_backend(backend, persist=False)
    try:
        yield
    finally:
        if original is not None and original != backend:
            set_backend(original, persist=False)


def _detect_torch_device(module: ModuleType) -> str:
    try:
        # Prioritize MPS on darwin systems
        mps_backend = getattr(module, "backends", {}).get("mps")
        if mps_backend and mps_backend.is_available():
            return "mps"
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        cuda_available = bool(getattr(module, "cuda", None) and module.cuda.is_available())
    except Exception:  # pragma: no cover - defensive
        cuda_available = False
    if cuda_available:
        return "cuda"
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
