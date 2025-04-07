"""
Device operations interface.
"""

import abc
from typing import Any, Optional

class DeviceOps(abc.ABC):
    """Abstract base class for backend-specific device operations."""

    @abc.abstractmethod
    def get_device(self, tensor: Optional[Any] = None) -> str:
        """
        Get the current default device or the device of a given tensor.

        Args:
            tensor: Optional tensor to get the device from.

        Returns:
            Device name as a string (e.g., 'cpu', 'cuda', 'mps').
        """
        pass

    @abc.abstractmethod
    def set_device(self, device: Any) -> None:
        """
        Set the current default device for the backend.

        Args:
            device: Device name as a string or a backend-specific device object.

        Raises:
            ValueError: If the device is not valid for the current backend.
        """
        pass