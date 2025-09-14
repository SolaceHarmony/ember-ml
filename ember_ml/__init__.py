"""
Ember ML: A backend-agnostic neural network library.

This library provides a unified interface for neural network operations
that can work with different backends (NumPy, PyTorch, MLX).
"""

import importlib
from typing import Union, Literal

# Re-export common tensor helpers and dtype constants for convenience
import ember_ml.dtypes as _dtypes_module
import ember_ml.tensor as _tensor_module
from ember_ml.dtypes import *  # noqa: F401,F403 - re-exported names

# Default backend
_CURRENT_BACKEND = None
_BACKEND_MODULE = None

def set_backend(backend_name: Union[str, Literal['numpy', 'torch', 'mlx']]) -> None:
    """
    Set the backend for neural network operations.

    Args:
        backend_name: Name of the backend ('numpy', 'torch', 'mlx')
    """
    global _CURRENT_BACKEND, _BACKEND_MODULE

    if backend_name == 'torch':
        _BACKEND_MODULE = importlib.import_module('ember_ml.backend.torch')
    elif backend_name == 'numpy':
        _BACKEND_MODULE = importlib.import_module('ember_ml.backend.numpy')
    elif backend_name == 'mlx':
        _BACKEND_MODULE = importlib.import_module('ember_ml.backend.mlx')
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    _CURRENT_BACKEND = backend_name

    # Import all functions from the backend module into the current namespace
    for name in dir(_BACKEND_MODULE):
        if not name.startswith('_'):
            globals()[name] = getattr(_BACKEND_MODULE, name)

# Import auto_select_backend from the backend module
from ember_ml.backend import auto_select_backend

# Set default backend using auto_select_backend
# This will respect the backend configuration and choose the best available backend
try:
    backend_name, _ = auto_select_backend()
    if backend_name:
        set_backend(backend_name)
    else:
        print("Warning: No default backend could be selected. Imports may fail if backend operations are used without calling set_backend().")
except Exception as e:
    print(f"Warning: Error selecting default backend: {e}. Imports may fail if backend operations are used without calling set_backend().")
    pass # Allow import to proceed without a default backend set

# Submodules are imported lazily via ``__getattr__`` below to avoid
# unnecessary import-time side effects and potential circular imports.


class _TensorProxy:
    """Callable wrapper around :mod:`ember_ml.tensor` mimicking ``torch.tensor``."""

    def __init__(self):
        self._module = importlib.import_module("ember_ml.tensor")

    def __getattr__(self, name):  # pragma: no cover - simple delegation
        return getattr(self._module, name)

    def __call__(self, *args, **kwargs):
        return self._module.convert_to_tensor(*args, **kwargs)


tensor = _TensorProxy()

# Export tensor ops at top level (except the ``tensor`` constructor)
for _name in getattr(_tensor_module, "__all__", []):
    if _name != "tensor":
        globals()[_name] = getattr(_tensor_module, _name)


def set_seed(seed):
    """Set the random seed for all backends."""
    from ember_ml import tensor

    tensor.set_seed(seed)
# Version of the Ember ML package
__version__ = '0.2.0'

# List of public objects exported by this module
__all__ = [
    'set_backend',
    'set_seed',
    # 'auto_select_backend', # Removed - moved to ops
    'data', 'models', 'nn', 'ops', 'training', 'visualization', 'utils', 'asyncml',
    'tensor', '__version__'
]

# Include tensor ops and dtype helpers in the public API
__all__ += [n for n in getattr(_tensor_module, '__all__', []) if n != 'tensor']
__all__ += list(getattr(_dtypes_module, '__all__', []))

_SUBMODULES = {
    'data', 'models', 'nn', 'ops', 'training', 'visualization', 'utils', 'asyncml'
}

def __getattr__(name):
    """Lazily import top-level submodules on first access."""
    if name in _SUBMODULES:
        module = importlib.import_module(f'ember_ml.{name}')
        globals()[name] = module
        return module
    raise AttributeError(f"module 'ember_ml' has no attribute '{name}'")
