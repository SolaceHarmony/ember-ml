"""
Interface for data type operations.

This module defines the abstract interface for data type operations that abstract
machine learning library data type operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Type, Union


class DTypeOps(ABC):
    """Interface for data type operations."""
    
    @property
    @abstractmethod
    def float32(self) -> Any:
        """Get the float32 data type."""
        pass
    
    @property
    @abstractmethod
    def float64(self) -> Any:
        """Get the float64 data type."""
        pass
    
    @property
    @abstractmethod
    def int32(self) -> Any:
        """Get the int32 data type."""
        pass
    
    @property
    @abstractmethod
    def int64(self) -> Any:
        """Get the int64 data type."""
        pass
    
    @property
    @abstractmethod
    def bool_(self) -> Any:
        """Get the boolean data type."""
        pass
    
    @property
    @abstractmethod
    def int8(self) -> Any:
        """Get the int8 data type."""
        pass
    
    @property
    @abstractmethod
    def int16(self) -> Any:
        """Get the int16 data type."""
        pass
    
    @property
    @abstractmethod
    def uint8(self) -> Any:
        """Get the uint8 data type."""
        pass
    
    @property
    @abstractmethod
    def uint16(self) -> Any:
        """Get the uint16 data type."""
        pass
    
    @property
    @abstractmethod
    def uint32(self) -> Any:
        """Get the uint32 data type."""
        pass
    
    @property
    @abstractmethod
    def uint64(self) -> Any:
        """Get the uint64 data type."""
        pass
    
    @property
    @abstractmethod
    def float16(self) -> Any:
        """Get the float16 data type."""
        pass
    
    @abstractmethod
    def get_dtype(self, name: str) -> Any:
        """
        Get a data type by name.
        
        Args:
            name: The name of the data type
            
        Returns:
            The corresponding data type
        """
        pass
    
    @abstractmethod
    def to_dtype_str(self, dtype: Any) -> str:
        """
        Convert a data type to a string representation.
        
        Args:
            dtype: The data type to convert
            
        Returns:
            The string representation of the data type
        """
        pass
    
    @abstractmethod
    def from_dtype_str(self, dtype_str: str) -> Any:
        """
        Convert a string representation to a backend-specific data type.
        
        Args:
            dtype_str: The string representation of the data type
            
        Returns:
            The corresponding backend-specific data type
        """
        pass