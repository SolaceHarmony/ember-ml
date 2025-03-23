"""
MLX device operations for ember_ml.

This module provides MLX implementations of device operations.
"""

import mlx.core as mx
import mlx.core
from typing import Optional, Dict, Any

# Import from tensor_ops
from ember_ml.backend.mlx.types import TensorLike

def to_device(x: TensorLike, device: str) -> mx.array:
    """
    Move an MLX array to the specified device.
    
    Args:
        x: Input array
        device: Target device (ignored for MLX backend)
        
    Returns:
        MLX array (unchanged)
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    return Tensor.convert_to_tensor(x)

def get_device(x: mx.array) -> str:
    """
    Get the device of an MLX array.
    
    Args:
        x: Input array
        
    Returns:
        Device of the array (always 'mps' for MLX backend on Apple Silicon)
    """
    device = mlx.core.default_device()
    if device == mx.cpu:
        return 'cpu'
    else:
        return 'gpu'

def get_available_devices() -> list[str]:
    """
    Get a list of available devices.
    
    Returns:
        List of available devices (always ['mps'] for MLX backend on Apple Silicon)
    """
    return ['mps']  # MLX uses Metal on Apple Silicon

def memory_usage(device: Optional[str] = None) -> Dict[str, int]:
    """
    Get the memory usage of the specified device.
    
    Args:
        device: Target device (ignored for MLX backend)
        
    Returns:
        Dictionary containing memory usage statistics
    """
    # Use MLX's metal.get_active_memory function to get the active memory usage
    try:
        active_memory = mx.metal.get_active_memory()
        return {
            'used': active_memory,
            'total': 0,  # MLX doesn't provide total memory info directly
            'free': 0    # MLX doesn't provide free memory info directly
        }
    except (AttributeError, ImportError):
        # Fallback if the function is not available
        return {
            'used': 0,
            'total': 0,
            'free': 0
        }


def memory_info(device: Optional[str] = None) -> Dict[str, Any]:
    """
    Get memory information for the specified device.
    
    Args:
        device: Device to get memory information for (default: current device)
    
    Returns:
        Dictionary containing memory information
    """
    # Use MLX's metal.device_info function to get detailed device information
    try:
        device_info = mx.metal.device_info()
        active_memory = mx.metal.get_active_memory()
        
        # Extract relevant information from device_info and ensure it's an integer
        total_memory = device_info.get('memory_size', 0)
        if not isinstance(total_memory, int):
            total_memory = 0
        # Convert to MLX tensors
        total_memory_tensor = mx.array(total_memory)
        active_memory_tensor = mx.array(active_memory)
        
        # Calculate available memory using mx functions
        available_memory_tensor = mx.maximum(mx.array(0), mx.subtract(total_memory_tensor, active_memory_tensor))
        
        # Return the tensor directly without conversion
        available_memory = available_memory_tensor
        
        # Calculate percentage safely using mx functions
        zero = mx.array(0)
        hundred = mx.array(100)
        
        if mx.greater(total_memory_tensor, zero).item():
            percent = mx.multiply(mx.divide(active_memory_tensor, total_memory_tensor), hundred)
        else:
            percent = mx.array(0.0)
        
        return {
            'total': total_memory,
            'used': active_memory,
            'available': available_memory,
            'percent': percent
        }
    except (AttributeError, ImportError, ZeroDivisionError, TypeError):
        # Fallback if the functions are not available
        return {
            'total': 0,
            'available': 0,
            'used': 0,
            'percent': 0
        }


def get_default_device() -> str:
    """
    Get the default device for MLX operations.
    
    Returns:
        Default device
    """
    # Convert MLX Device to string
    device = mx.default_device()
    device_str = str(device)
    
    # Extract just the device type without the "DeviceType." prefix
    if device_str.startswith("DeviceType."):
        return device_str.split(".")[-1]
    return device_str


def set_default_device(device: str) -> None:
    """
    Set the default device for MLX operations.
    
    Args:
        device: Default device
    """
    # MLX expects a Device object, but we'll handle string inputs
    if device == 'cpu':
        mx.set_default_device(mx.Device(mx.cpu))
    elif device == 'mps' or device == 'gpu':
        # On Apple Silicon, use Metal
        mx.set_default_device(mx.Device(mx.gpu))
    else:
        # Default to CPU for unknown devices
        mx.set_default_device(mx.Device(mx.cpu))


def is_available(device: str) -> bool:
    """
    Check if the specified device is available.
    
    Args:
        device: Device to check
    
    Returns:
        True if the device is available, False otherwise
    """
    if device == 'cpu':
        return True
    elif device == 'mps' or device == 'gpu':
        # Check if Metal is available
        return mx.metal.is_available()
    return False


def synchronize(device: Optional[str] = None) -> None:
    """
    Synchronize the specified device.
    
    Args:
        device: Target device (ignored for MLX backend)
    """
    # MLX handles synchronization automatically
    pass


class MLXDeviceOps:
    """MLX implementation of device operations."""
    
    def to_device(self, x, device):
        """Move a tensor to the specified device."""
        return to_device(x, device)
    
    def get_device(self, x):
        """Get the device of a tensor."""
        return get_device(x)
    
    def get_default_device(self):
        """Get the default device for tensor operations."""
        return get_default_device()
    
    def set_default_device(self, device):
        """Set the default device for tensor operations."""
        set_default_device(device)
    
    def get_available_devices(self):
        """Get a list of available devices."""
        return get_available_devices()
    
    def memory_usage(self, device=None):
        """Get the memory usage of the specified device."""
        return memory_usage(device)
    
    def memory_info(self, device=None):
        """Get memory information for the specified device."""
        return memory_info(device)
    
    def is_available(self, device_type):
        """Check if a device type is available."""
        return is_available(device_type)
    
    def synchronize(self, device=None):
        """Synchronize the specified device."""
        synchronize(device)