"""
Ember ML: A backend-agnostic neural network library.

This library provides a unified interface for neural network operations
that can work with different backends (NumPy, PyTorch, MLX).
"""

import importlib
from typing import Union, Literal

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

# Import auto_select_backend from the backend module - REMOVED
# from ember_ml.backend import auto_select_backend

# Set default backend - REMOVED auto_select_backend usage
# User should explicitly call set_backend() initially.
# Fallback backend setting if none explicitly set:
try:
    set_backend('torch')
except ImportError:
    # Fallback to NumPy if PyTorch is not available
    try:
        set_backend('numpy')
    except ImportError:
        # Fallback to MLX if NumPy is not available
        try:
            set_backend('mlx')
        except ImportError:
            # If no backend is available, don't raise error here,
            # let subsequent ops fail naturally if backend is needed.
            # Consider adding a warning.
            print("Warning: No default backend (torch, numpy, mlx) found. Imports may fail if backend operations are used without calling set_backend().")
            pass # Allow import to proceed without a default backend set

# Import submodules
# from ember_ml import benchmarks # Removed - moved out of package
from ember_ml import data
from ember_ml import models
from ember_ml import nn
from ember_ml import ops
from ember_ml import training
from ember_ml import visualization
from ember_ml import wave
from ember_ml import utils
from ember_ml import asyncml
# Version of the Ember ML package
__version__ = '0.2.0'

# List of public objects exported by this module
__all__ = [
    'set_backend',
    # 'auto_select_backend', # Removed - moved to ops
    'data',
    'models',
    'nn',
    'ops',
    'training',
    'visualization',
    'wave',
    'utils',
    'asyncml',
    '__version__'
]
