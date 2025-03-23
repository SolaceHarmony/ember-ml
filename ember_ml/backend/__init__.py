"""
Backend module.

This module provides backend implementations,
including PyTorch, NumPy, and MLX.
"""

import importlib
import os
import platform
from pathlib import Path

# Import tensor classes from all backends
from ember_ml.backend.torch.tensor import TorchDType, TorchTensor
from ember_ml.backend.numpy.tensor import NumpyDType, NumpyTensor
from ember_ml.backend.mlx.tensor import MLXDType, MLXTensor

# Available backends
_BACKENDS = {
    'numpy': 'ember_ml.backend.numpy',
    'torch': 'ember_ml.backend.torch',
    'mlx': 'ember_ml.backend.mlx'
}

# Path to the .ember directory in the user's home directory
EMBER_CONFIG_DIR = Path.home() / '.ember'
EMBER_BACKEND_FILE = EMBER_CONFIG_DIR / 'backend'

# Current backend
_CURRENT_BACKEND = None
_CURRENT_BACKEND_MODULE = None

def _get_backend_from_file():
    """Get the backend from the .ember/backend file."""
    try:
        return EMBER_BACKEND_FILE.read_text().strip()
    except Exception:
        return None

def _save_backend_to_file(backend):
    """Save the backend to the .ember/backend file."""
    EMBER_BACKEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    EMBER_BACKEND_FILE.write_text(backend)

def _reload_ops_module():
    """Reload the ops module to ensure it uses the new backend."""
    try:
        ops_module = importlib.import_module('ember_ml.ops')
        if hasattr(ops_module, '_CURRENT_INSTANCES'):
            setattr(ops_module, '_CURRENT_INSTANCES', {})
        importlib.reload(ops_module)
    except Exception as e:
        print(f"Warning: Error updating ops module after backend switch: {e}")

def get_backend():
    """Get the current backend."""
    global _CURRENT_BACKEND
    if _CURRENT_BACKEND is not None:
        return _CURRENT_BACKEND

    backend = _get_backend_from_file() or os.environ.get('EMBER_ML_BACKEND')
    if backend is None:
        backend, _ = auto_select_backend()
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
    
    # Save the backend to the .ember/backend file
    _save_backend_to_file(backend)
    
    # Also store in environment variable for backward compatibility
    os.environ['EMBER_ML_BACKEND'] = backend
    
    # Clear the current backend module
    _CURRENT_BACKEND_MODULE = None
    
    _reload_ops_module()

def get_backend_module():
    """Get the current backend module."""
    global _CURRENT_BACKEND_MODULE
    
    if _CURRENT_BACKEND_MODULE is None:
        # Get the current backend
        backend = get_backend()
        
        # Import the backend module
        if backend in _BACKENDS:
            _CURRENT_BACKEND_MODULE = importlib.import_module(_BACKENDS[backend])
            
            # Add tensor classes to the backend module
            if backend == 'numpy':
                setattr(_CURRENT_BACKEND_MODULE, 'DType', NumpyDType)
                setattr(_CURRENT_BACKEND_MODULE, 'Tensor', NumpyTensor)
            elif backend == 'torch':
                setattr(_CURRENT_BACKEND_MODULE, 'DType', TorchDType)
                setattr(_CURRENT_BACKEND_MODULE, 'Tensor', TorchTensor)
            elif backend == 'mlx':
                setattr(_CURRENT_BACKEND_MODULE, 'DType', MLXDType)
                setattr(_CURRENT_BACKEND_MODULE, 'Tensor', MLXTensor)
        else:
            raise ValueError(f"Invalid backend: {backend}. Available backends: {list(_BACKENDS.keys())}")
    
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
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
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
    
    # Convert device to string for consistency
    device_str = str(device)
    
    # Handle MLX DeviceType objects directly
    if device_str == 'DeviceType.gpu':
        if backend == 'mlx':
            # MLX doesn't support explicit device setting yet
            return
        else:
            device_str = 'gpu'
    
    # Convert to lowercase string for consistency
    device_str = device_str.lower()
    
    if backend == 'torch':
        import torch
        if device_str == 'cuda' and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        if device_str == 'mps' and not (hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise ValueError("MPS is not available")
        if device_str not in ['cpu', 'cuda', 'mps']:
            raise ValueError(f"Invalid device for PyTorch: {device_str}")
        torch.device(device_str)
    elif backend == 'mlx':
        import mlx.core as mx
        # MLX uses 'gpu' or 'cpu' internally
        if device_str in ['metal', 'gpu']:
            # MLX doesn't support explicit device setting yet
            return
        if device_str != 'cpu':
            raise ValueError(f"Invalid device for MLX: {device_str}")
        # MLX doesn't support explicit device setting yet
    elif device_str != 'cpu':
        raise ValueError(f"Backend {backend} only supports 'cpu' device")

def auto_select_backend():
    """Automatically select the best backend based on the available hardware."""
    # Check for PyTorch with CUDA
    def _check_torch_cuda():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # Check for PyTorch with MPS (Apple Silicon)
    def _check_torch_mps():
        try:
            import torch
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except ImportError:
            return False

    # Check for PyTorch
    def _check_torch():
        try:
            import torch
            return True
        except ImportError:
            return False

    if _check_torch_cuda():
        return 'torch', 'cuda'
    if _check_torch_mps():
        return 'torch', 'mps'
    if _check_torch():
        return 'torch', 'cpu'
    
    # Check for MLX (Apple Silicon)
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        try:
            import mlx.core
            return 'mlx', None
        except ImportError:
            pass

