"""
NumPy device operations for EmberHarmony.

This module provides NumPy implementations of device operations.
"""

import numpy as np
from typing import Optional, Union, List, Any, Sequence, Dict

# Type aliases
ArrayLike = Union[np.ndarray, float, int, list, tuple]
Shape = Union[int, Sequence[int]]
DType = Union[np.dtype, str, None]

# Import from config
from ember_ml.backend.numpy.config import DEFAULT_DEVICE

# Import from tensor_ops
from ember_ml.backend.numpy.tensor_ops import convert_to_tensor

# Try to import psutil for memory info
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False


def to_device(x: ArrayLike, device: str) -> np.ndarray:
    """
    Move a NumPy array to the specified device.
    
    Args:
        x: Input array
        device: Target device (ignored for NumPy backend)
        
    Returns:
        NumPy array (unchanged)
    """
    # NumPy doesn't have device concept, so just return the array
    return convert_to_tensor(x)


def get_device(x: ArrayLike) -> str:
    """
    Get the device of a NumPy array.
    
    Args:
        x: Input array
        
    Returns:
        Device of the array (always 'cpu' for NumPy backend)
    """
    return 'cpu'


def get_available_devices() -> List[str]:
    """
    Get a list of available devices.
    
    Returns:
        List of available devices (always ['cpu'] for NumPy backend)
    """
    return ['cpu']


def memory_usage(device: Optional[str] = None) -> float:
    """
    Get the memory usage of the specified device.
    
    Args:
        device: Device to get memory usage for (default: current device)
        
    Returns:
        Memory usage in bytes
    """
    if device is not None and device != 'cpu':
        raise ValueError(f"NumPy backend only supports 'cpu' device, got {device}")
    
    if HAVE_PSUTIL:
        # Get system memory usage
        mem = psutil.virtual_memory()
        return mem.used
    else:
        return 0.0


def memory_info(device: Optional[str] = None) -> Dict[str, Any]:
    """
    Get memory information for the specified device.
    
    Args:
        device: Device to get memory information for (default: current device)
        
    Returns:
        Dictionary containing memory information
    """
    if device is not None and device != 'cpu':
        raise ValueError(f"NumPy backend only supports 'cpu' device, got {device}")
    
    if HAVE_PSUTIL:
        # Get system memory information
        mem = psutil.virtual_memory()
        
        return {
            'total': mem.total,
            'available': mem.available,
            'used': mem.used,
            'percent': mem.percent
        }
    else:
        return {
            'total': 0,
            'available': 0,
            'used': 0,
            'percent': 0
        }


def synchronize(device: Optional[str] = None) -> None:
    """
    Synchronize the specified device.
    
    Args:
        device: Device to synchronize (default: current device)
    """
    # NumPy is synchronous, so this is a no-op
    pass


def set_default_device(device: str) -> None:
    """
    Set the default device for tensor operations.
    
    Args:
        device: Default device
    """
    if device != 'cpu':
        raise ValueError(f"NumPy backend only supports 'cpu' device, got {device}")


def get_default_device() -> str:
    """
    Get the default device for tensor operations.
    
    Returns:
        Default device
    """
    return DEFAULT_DEVICE


def is_available(device_type: str) -> bool:
    """
    Check if a device type is available.
    
    Args:
        device_type: Device type to check
        
    Returns:
        True if the device type is available, False otherwise
    """
    return device_type == 'cpu'


class NumpyDeviceOps:
    """NumPy implementation of device operations."""
    
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
    
    def memory_info(self, device=None):
        """Get memory information for the specified device."""
        return memory_info(device)
    
    def synchronize(self, device=None):
        """Synchronize the specified device."""
        synchronize(device)
    
    def set_default_device(self, device):
        """Set the default device for tensor operations."""
        set_default_device(device)
    
    def get_default_device(self):
        """Get the default device for tensor operations."""
        return get_default_device()
    
    def is_available(self, device_type):
        """Check if a device type is available."""
        return is_available(device_type)