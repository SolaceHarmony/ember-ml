"""
MLX implementation of device operations.

This module provides MLX implementations of device operations.
"""

import mlx.core as mx
from typing import Union, Sequence, Optional, Tuple, Any, List

# Type aliases
ArrayLike = Union[mx.array, float, int, list, tuple]

# Import convert_to_tensor from tensor_ops
from ember_ml.backend.mlx.tensor_ops import convert_to_tensor

def to_device(x: ArrayLike, device: str) -> mx.array:
    """
    Move an MLX array to the specified device.
    
    Args:
        x: Input array
        device: Target device (ignored for MLX backend)
        
    Returns:
        MLX array (unchanged)
    """
    # MLX automatically uses the most efficient device (Metal on Apple Silicon)
    return convert_to_tensor(x)

def get_device(x: ArrayLike) -> str:
    """
    Get the device of an MLX array.
    
    Args:
        x: Input array
        
    Returns:
        Device of the array (always 'mps' for MLX backend on Apple Silicon)
    """
    return 'mps'  # MLX uses Metal on Apple Silicon

def get_available_devices() -> List[str]:
    """
    Get a list of available devices.
    
    Returns:
        List of available devices (always ['mps'] for MLX backend on Apple Silicon)
    """
    return ['mps']  # MLX uses Metal on Apple Silicon

def memory_usage(device: Optional[str] = None) -> int:
    """
    Get the memory usage of the specified device.
    
    Args:
        device: Target device (ignored for MLX backend)
        
    Returns:
        Memory usage in bytes (always 0 for MLX backend as it's not directly accessible)
    """
    # MLX doesn't provide a direct way to get memory usage
    return 0


class MLXDeviceOps:
    """MLX implementation of device operations."""
    
    def to_device(self, x, device):
        """Move a tensor to the specified device."""
        return to_device(x, device)
    
    def get_device(self, x):
        """Get the device of a tensor."""
        return get_device(x)
    
    def get_available_devices(self):
        """Get a list of available devices."""
        return get_available_devices()
    
    def memory_usage(self, device=None):
        """Get the memory usage of the specified device."""
        return memory_usage(device)
    
    def is_available(self, device_type):
        """Check if a device type is available."""
        return device_type == 'mps'
    
    def synchronize(self, device=None):
        """Synchronize the specified device."""
        # MLX handles synchronization automatically
        pass