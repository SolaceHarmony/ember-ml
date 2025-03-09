"""
Backend module.

This module provides backend implementations,
including PyTorch, NumPy, and MLX.
"""

import os
import sys
import platform
import importlib
from typing import Optional, Tuple

# Available backends
_BACKENDS = {
    'numpy': 'emberharmony.backend.numpy',
    'torch': 'emberharmony.backend.torch',
    'torch_optimized': 'emberharmony.backend.torch_backend_optimized',
    'mlx': 'emberharmony.backend.mlx'
}

# Current backend
_CURRENT_BACKEND = None
_CURRENT_BACKEND_MODULE = None

def get_backend():
    """Get the current backend."""
    global _CURRENT_BACKEND
    
    if _CURRENT_BACKEND is None:
        # Try to get the backend from environment variable
        backend = os.environ.get('EMBERHARMONY_BACKEND')
        
        # If not set, use the default backend
        if backend is None:
            # Default to MLX on Apple Silicon, PyTorch otherwise
            if platform.system() == 'Darwin' and platform.machine() == 'arm64':
                backend = 'mlx'
            else:
                backend = 'torch'
        
        # Set the backend
        set_backend(backend)
    
    return _CURRENT_BACKEND

def set_backend(backend: str):
    """Set the current backend."""
    global _CURRENT_BACKEND, _CURRENT_BACKEND_MODULE
    
    # Check if the backend is valid
    if backend not in _BACKENDS:
        raise ValueError(f"Invalid backend: {backend}. Available backends: {list(_BACKENDS.keys())}")
    
    # Set the current backend
    _CURRENT_BACKEND = backend
    
    # Store the backend in an environment variable for persistence across module reloads
    os.environ['EMBERHARMONY_BACKEND'] = backend
    
    # Clear the current backend module
    _CURRENT_BACKEND_MODULE = None

def get_backend_module():
    """Get the current backend module."""
    global _CURRENT_BACKEND_MODULE
    
    if _CURRENT_BACKEND_MODULE is None:
        # Get the current backend
        backend = get_backend()
        
        # Import the backend module
        _CURRENT_BACKEND_MODULE = importlib.import_module(_BACKENDS[backend])
    
    return _CURRENT_BACKEND_MODULE

def get_device():
    """Get the current device."""
    backend = get_backend()
    
    if backend == 'numpy':
        return 'cpu'
    elif backend == 'torch':
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    elif backend == 'mlx':
        import mlx.core as mx
        return mx.default_device().type
    else:
        return 'cpu'

def auto_select_backend():
    """Automatically select the best backend based on the available hardware."""
    # Check for MLX (Apple Silicon)
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        try:
            import mlx.core
            return 'mlx', 'Metal'
        except ImportError:
            pass
    
    # Check for PyTorch with CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return 'torch', 'CUDA'
    except ImportError:
        pass
    
    # Check for PyTorch
    try:
        import torch
        return 'torch', 'CPU'
    except ImportError:
        pass
    
    # Fallback to NumPy
    return 'numpy', 'CPU'
