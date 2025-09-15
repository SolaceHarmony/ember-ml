"""
Ember ML: A backend-agnostic neural network library.

This library provides a unified interface for neural network operations
that can work with different backends (NumPy, PyTorch, MLX).
"""

import importlib
from typing import Union, Literal

# Import core backend management functions directly
from ember_ml.backend import (
    set_backend as _set_backend,
    get_backend, 
    auto_select_backend,
    get_available_backends,
    using_backend
)

def set_backend(backend_name: Union[str, Literal['numpy', 'torch', 'mlx']]) -> None:
    """
    Set the backend for neural network operations.

    Args:
        backend_name: Name of the backend ('numpy', 'torch', 'mlx')
    """
    _set_backend(backend_name)

# Initialize backend system
try:
    # Try to get the current backend, which will trigger auto-selection if needed
    current_backend = get_backend()
    if not current_backend:
        # If no backend is available, auto-select one
        backend_name, _ = auto_select_backend()
        if backend_name:
            set_backend(backend_name)
        else:
            print("Warning: No default backend could be selected. Some functionality may not work.")
except Exception as e:
    print(f"Warning: Error initializing backend system: {e}. Some functionality may not work.")

# Import core modules - these should always be available
from ember_ml import ops

# Create a lazy loader to handle module imports on-demand
class _ModuleAccessor:
    """Lazy loader for optional modules."""
    
    def __init__(self):
        self._loaded = {}
    
    @property
    def data(self):
        if 'data' not in self._loaded:
            try:
                from ember_ml import data
                self._loaded['data'] = data
            except ImportError as e:
                print(f"Warning: Could not import data module: {e}")
                self._loaded['data'] = None
        return self._loaded['data']
    
    @property
    def models(self):
        if 'models' not in self._loaded:
            try:
                from ember_ml import models
                self._loaded['models'] = models
            except ImportError as e:
                print(f"Warning: Could not import models module: {e}")
                self._loaded['models'] = None
        return self._loaded['models']
    
    @property
    def nn(self):
        if 'nn' not in self._loaded:
            try:
                from ember_ml import nn
                self._loaded['nn'] = nn
            except ImportError as e:
                print(f"Warning: Could not import nn module: {e}")
                self._loaded['nn'] = None
        return self._loaded['nn']
    
    @property
    def training(self):
        if 'training' not in self._loaded:
            try:
                from ember_ml import training
                self._loaded['training'] = training
            except ImportError as e:
                print(f"Warning: Could not import training module: {e}")
                self._loaded['training'] = None
        return self._loaded['training']
    
    @property
    def visualization(self):
        if 'visualization' not in self._loaded:
            try:
                from ember_ml import visualization
                self._loaded['visualization'] = visualization
            except ImportError as e:
                print(f"Warning: Could not import visualization module: {e}")
                self._loaded['visualization'] = None
        return self._loaded['visualization']
    
    @property
    def utils(self):
        if 'utils' not in self._loaded:
            try:
                from ember_ml import utils
                self._loaded['utils'] = utils
            except ImportError as e:
                print(f"Warning: Could not import utils module: {e}")
                self._loaded['utils'] = None
        return self._loaded['utils']
    
    @property
    def asyncml(self):
        if 'asyncml' not in self._loaded:
            try:
                from ember_ml import asyncml
                self._loaded['asyncml'] = asyncml
            except ImportError as e:
                print(f"Warning: Could not import asyncml module: {e}")
                self._loaded['asyncml'] = None
        return self._loaded['asyncml']

# Create the module accessor
_modules = _ModuleAccessor()

# Function to handle dynamic attribute access for the module
def __getattr__(name):
    """Lazy load modules on demand."""
    if hasattr(_modules, name):
        return getattr(_modules, name)
    else:
        raise AttributeError(f"module 'ember_ml' has no attribute '{name}'")


class _TensorProxy:
    """Callable wrapper around :mod:`ember_ml.tensor` mimicking ``torch.tensor``."""

    def __init__(self):
        self._module = None

    def _get_module(self):
        if self._module is None:
            try:
                self._module = importlib.import_module("ember_ml.tensor")
            except ImportError as e:
                print(f"Warning: Could not import tensor module: {e}")
                self._module = None
        return self._module

    def __getattr__(self, name):  # pragma: no cover - simple delegation
        module = self._get_module()
        if module is None:
            raise AttributeError(f"tensor module not available")
        return getattr(module, name)

    def __call__(self, *args, **kwargs):
        module = self._get_module()
        if module is None:
            raise RuntimeError("tensor module not available")
        return module.convert_to_tensor(*args, **kwargs)


tensor = _TensorProxy()


def set_seed(seed):
    """Set the random seed for all backends."""
    try:
        from ember_ml import tensor as tensor_module
        tensor_module.set_seed(seed)
    except ImportError:
        print("Warning: Could not set seed, tensor module not available")

# Version of the Ember ML package
__version__ = '0.2.0'

# List of public objects exported by this module
__all__ = [
    'set_backend',
    'get_backend', 
    'auto_select_backend',
    'get_available_backends',
    'using_backend',
    'set_seed',
    'ops',
    'tensor',
    '__version__'
]
