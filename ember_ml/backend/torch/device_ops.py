"""
PyTorch device operations for ember_ml.

This module provides PyTorch implementations of device operations.
"""

import torch
from typing import Optional, List, Dict, Any

# Import from tensor_ops
from ember_ml.backend.torch.tensor_ops import convert_to_tensor, ArrayLike
from ember_ml.backend.torch.config import DEFAULT_DEVICE, _default_device


def to_device(x: ArrayLike, device: str) -> torch.Tensor:
    """
    Move a tensor to the specified device.
    
    Args:
        x: Input tensor
        device: Target device
        
    Returns:
        Tensor on the target device
    """
    x_tensor = convert_to_tensor(x)
    return x_tensor.to(device)


def get_device(x: ArrayLike) -> str:
    """
    Get the device of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Device of the tensor
    """
    x_tensor = convert_to_tensor(x)
    return str(x_tensor.device)


def get_available_devices() -> List[str]:
    """
    Get a list of available devices.
    
    Returns:
        List of available devices
    """
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append('mps')
    return devices


def set_default_device(device: str) -> None:
    """
    Set the default device for PyTorch operations.
    
    Args:
        device: Default device
    """
    global _default_device
    _default_device = device
    
    # Set the default device for PyTorch
    if device.startswith('cuda'):
        if torch.cuda.is_available():
            device_idx = 0
            if ':' in device:
                # Use convert_to_tensor and cast instead of int()
                device_idx_str = device.split(':')[1]
                device_idx_tensor = convert_to_tensor(device_idx_str)
                device_idx = device_idx_tensor.to(torch.int32).item()
            torch.cuda.set_device(device_idx)


def get_default_device() -> str:
    """
    Get the default device for PyTorch operations.
    
    Returns:
        Default device
    """
    return _default_device


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
    elif device.startswith('cuda'):
        return torch.cuda.is_available()
    elif device == 'mps':
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    return False


def memory_usage(device: Optional[str] = None) -> Dict[str, int]:
    """
    Get memory usage information for the specified device.
    
    Args:
        device: Target device
        
    Returns:
        Dictionary with memory usage information
    """
    if device is None:
        device = _default_device
        
    if device.startswith('cuda'):
        if torch.cuda.is_available():
            device_idx = 0
            if ':' in device:
                # Use convert_to_tensor and cast instead of int()
                device_idx_str = device.split(':')[1]
                device_idx_tensor = convert_to_tensor(device_idx_str)
                device_idx = device_idx_tensor.to(torch.int32).item()
            
            # Get memory information
            allocated = torch.cuda.memory_allocated(device_idx)
            reserved = torch.cuda.memory_reserved(device_idx)
            
            # Get total memory
            total = torch.cuda.get_device_properties(device_idx).total_memory
            
            # Calculate free memory using torch.subtract instead of direct subtraction
            free = torch.subtract(torch.tensor(total), torch.tensor(reserved)).item()
            
            return {
                'allocated': allocated,
                'reserved': reserved,
                'free': free,
                'total': total
            }
    
    # For CPU or other devices, return zeros
    return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0}


def memory_info(device: Optional[str] = None) -> Dict[str, int]:
    """
    Get memory information for the specified device.
    
    Args:
        device: Target device
        
    Returns:
        Dictionary with memory information
    """
    return memory_usage(device)


def synchronize(device: Optional[str] = None) -> None:
    """
    Synchronize the specified device.
    
    Args:
        device: Target device
    """
    if device is None:
        device = _default_device
        
    if device.startswith('cuda'):
        if torch.cuda.is_available():
            device_idx = 0
            if ':' in device:
                # Use convert_to_tensor and cast instead of int()
                device_idx_str = device.split(':')[1]
                device_idx_tensor = convert_to_tensor(device_idx_str)
                device_idx = device_idx_tensor.to(torch.int32).item()
            torch.cuda.synchronize(device_idx)


class TorchDeviceOps:
    """PyTorch implementation of device operations."""
    
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
        """Get memory usage information for the specified device."""
        return memory_usage(device)