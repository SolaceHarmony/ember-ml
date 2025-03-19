"""
I/O operations interface.

This module defines the abstract interface for I/O operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

class IOOps(ABC):
    """Abstract interface for I/O operations."""
    
    @abstractmethod
    def save(self, filepath: str, obj: Any, allow_pickle: bool = True) -> None:
        """
        Save a tensor or dictionary of tensors to a file.
        
        Args:
            filepath: Path to save the object to
            obj: Tensor or dictionary of tensors to save
            allow_pickle: Whether to allow saving objects that can't be saved directly
            
        Returns:
            None
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str, allow_pickle: bool = True) -> Any:
        """
        Load a tensor or dictionary of tensors from a file.
        
        Args:
            filepath: Path to load the object from
            allow_pickle: Whether to allow loading objects that can't be loaded directly
            
        Returns:
            Loaded tensor or dictionary of tensors
        """
        pass