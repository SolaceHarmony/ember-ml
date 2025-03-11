"""
Device operations interface.

This module defines the abstract interface for device operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List

class DeviceOps(ABC):
    """Abstract interface for device operations."""
    
    @abstractmethod
    def to_device(self, x: Any, device: str) -> Any:
        """
        Move a tensor to the specified device.
        
        Args:
            x: Input tensor
            device: Target device
            
        Returns:
            Tensor on the target device
        """
        pass
    
    @abstractmethod
    def get_device(self, x: Any) -> str:
        """
        Get the device of a tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Device of the tensor
        """
        pass
    
    @abstractmethod
    def set_default_device(self, device: str) -> None:
        """
        Set the default device for tensor operations.
        
        Args:
            device: Default device
        """
        pass
    
    @abstractmethod
    def get_default_device(self) -> str:
        """
        Get the default device for tensor operations.
        
        Returns:
            Default device
        """
        pass
    
    @abstractmethod
    def synchronize(self, device: Optional[str] = None) -> None:
        """
        Synchronize the specified device.
        
        Args:
            device: Device to synchronize (default: current device)
        """
        pass
    
    @abstractmethod
    def is_available(self, device_type: str) -> bool:
        """
        Check if a device type is available.
        
        Args:
            device_type: Device type to check
            
        Returns:
            True if the device type is available, False otherwise
        """
        pass
    
    @abstractmethod
    def memory_info(self, device: Optional[str] = None) -> dict:
        """
        Get memory information for the specified device.
        
        Args:
            device: Device to get memory information for (default: current device)
            
        Returns:
            Dictionary containing memory information
        """
        pass

    @abstractmethod
    def memory_usage(self, device: Optional[str] = None) -> dict:
        """
        Get memory usage statistics for the specified device.
        
        Args:
            device: Device to get memory usage for (default: current device)
            
        Returns:
            Dictionary containing memory usage statistics
        """
        pass

    @abstractmethod
    def get_available_devices(self) -> List[str]:
        """
        Get list of available devices.
        
        Returns:
            List of available device names
        """
        pass