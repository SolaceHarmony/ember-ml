"""
EmberHarmony: A backend-agnostic neural network library.

This library provides a unified interface for neural network operations
that can work with different backends (NumPy, PyTorch, MLX).
"""

import importlib
from typing import Optional, Union, Literal

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
        _BACKEND_MODULE = importlib.import_module('emberharmony.backend.torch_backend')
    elif backend_name == 'numpy':
        _BACKEND_MODULE = importlib.import_module('emberharmony.backend.numpy_backend')
    elif backend_name == 'mlx':
        _BACKEND_MODULE = importlib.import_module('emberharmony.backend.mlx_backend')
    else:
        raise ValueError(f"Unknown backend: {backend_name}")
    
    _CURRENT_BACKEND = backend_name
    
    # Import all functions from the backend module into the current namespace
    for name in dir(_BACKEND_MODULE):
        if not name.startswith('_'):
            globals()[name] = getattr(_BACKEND_MODULE, name)

# Import auto_select_backend from the backend module
from ember_ml.backend import auto_select_backend

# Set default backend using auto-selection
try:
    backend, _ = auto_select_backend()
    set_backend(backend)
except ImportError:
    # Fallback to PyTorch if auto-selection fails
    try:
        set_backend('torch')
    except ImportError:
        # Fallback to NumPy if PyTorch is not available
        try:
            set_backend('numpy')
        except ImportError:
            raise ImportError("No backend is available. Please install PyTorch, MLX, or NumPy.")

# Import submodules
from ember_ml import nn
from ember_ml import core
from ember_ml import attention
from ember_ml import wave
from ember_ml import keras_3_8
from ember_ml import utils
from ember_ml import models
from ember_ml import features
from ember_ml import audio
from ember_ml import math
from ember_ml import training
from ember_ml import visualization
from ember_ml import solvers
from ember_ml import data
from ember_ml import ops
from ember_ml import wirings
from ember_ml import random

# Import random functions directly
from ember_ml.random import seed, set_random_seed

# Version of the emberharmony package
__version__ = '0.2.0'

# List of public objects exported by this module
__all__ = [
    'core',
    'attention',
    'wave',
    'keras_3_8',
    'utils',
    'models',
    'features',
    'audio',
    'math',
    'training',
    'visualization',
    'solvers',
    'data',
    'ops',
    'wirings',
    'nn',
    'random',
    'set_backend',
    'seed',
    'set_random_seed',
]
