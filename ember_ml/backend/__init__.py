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
    'numpy': 'ember_ml.backend.numpy',
    'torch': 'ember_ml.backend.torch',
    'mlx': 'ember_ml.backend.mlx'
}

# Current backend
_CURRENT_BACKEND = None
_CURRENT_BACKEND_MODULE = None

def get_backend():
    """Get the current backend."""
    global _CURRENT_BACKEND
    
    if _CURRENT_BACKEND is None:
        # Try to get the backend from environment variable
        backend = os.environ.get('EMBER_ML_BACKEND')
        
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
    os.environ['EMBER_ML_BACKEND'] = backend
    
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

def get_device(tensor=None):
    """
    Get the current device.
    
    Args:
        tensor: Optional tensor to get the device from
        
    Returns:
        Device name as a string
    """
    backend = get_backend()
    
    if tensor is not None:
        # If a tensor is provided, try to get its device
        if hasattr(tensor, 'device'):
            return str(tensor.device)
    
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

def set_device(device):
    """
    Set the current device.
    
    Args:
        device: Device name as a string or device object
        
    Raises:
        ValueError: If the device is not valid for the current backend
    """
    backend = get_backend()
    
    # Handle MLX DeviceType objects directly
    if str(device) == 'DeviceType.gpu':
        if backend == 'mlx':
            # MLX doesn't support explicit device setting yet
            return
        else:
            device = 'gpu'
    
    # Convert device to string if it's an object
    if hasattr(device, 'type'):
        device = device.type
    
    # Convert to lowercase string for consistency
    if isinstance(device, str):
        device = device.lower()
    
    if backend == 'torch':
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        if device == 'mps' and not (hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise ValueError("MPS is not available")
        if device not in ['cpu', 'cuda', 'mps']:
            raise ValueError(f"Invalid device for PyTorch: {device}")
        torch.device(device)
    elif backend == 'mlx':
        import mlx.core as mx
        # MLX uses 'gpu' or 'cpu' internally
        if device in ['metal', 'gpu']:
            # MLX doesn't support explicit device setting yet
            return
        if device != 'cpu':
            raise ValueError(f"Invalid device for MLX: {device}")
        # MLX doesn't support explicit device setting yet
    elif device != 'cpu':
        raise ValueError(f"Backend {backend} only supports 'cpu' device")

def auto_select_backend():
    """Automatically select the best backend based on the available hardware."""
    # Check for MLX (Apple Silicon)
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        try:
            import mlx.core
            return 'mlx', None
        except ImportError:
            pass
    
    # Check for PyTorch with CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return 'torch', 'cuda'
    except ImportError:
        pass
    
    # Check for PyTorch
    try:
        import torch
        return 'torch', 'cpu'
    except ImportError:
        pass
    
    # Fallback to NumPy
    return 'numpy', None
