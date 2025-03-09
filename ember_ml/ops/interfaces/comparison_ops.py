"""
Comparison operations interface.

This module defines the abstract interface for comparison operations.
"""

from abc import ABC, abstractmethod
from typing import Any

class ComparisonOps(ABC):
    """Abstract interface for comparison operations."""
    
    @abstractmethod
    def equal(self, x: Any, y: Any) -> Any:
        """
        Check if two tensors are equal element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean tensor with True where x == y
        """
        pass
    
    @abstractmethod
    def not_equal(self, x: Any, y: Any) -> Any:
        """
        Check if two tensors are not equal element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean tensor with True where x != y
        """
        pass
    
    @abstractmethod
    def less(self, x: Any, y: Any) -> Any:
        """
        Check if one tensor is less than another element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean tensor with True where x < y
        """
        pass
    
    @abstractmethod
    def less_equal(self, x: Any, y: Any) -> Any:
        """
        Check if one tensor is less than or equal to another element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean tensor with True where x <= y
        """
        pass
    
    @abstractmethod
    def greater(self, x: Any, y: Any) -> Any:
        """
        Check if one tensor is greater than another element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean tensor with True where x > y
        """
        pass
    
    @abstractmethod
    def greater_equal(self, x: Any, y: Any) -> Any:
        """
        Check if one tensor is greater than or equal to another element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean tensor with True where x >= y
        """
        pass
    
    @abstractmethod
    def logical_and(self, x: Any, y: Any) -> Any:
        """
        Compute the logical AND of two tensors element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean tensor with True where x AND y
        """
        pass
    
    @abstractmethod
    def logical_or(self, x: Any, y: Any) -> Any:
        """
        Compute the logical OR of two tensors element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean tensor with True where x OR y
        """
        pass
    
    @abstractmethod
    def logical_not(self, x: Any) -> Any:
        """
        Compute the logical NOT of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Boolean tensor with True where NOT x
        """
        pass
    
    @abstractmethod
    def logical_xor(self, x: Any, y: Any) -> Any:
        """
        Compute the logical XOR of two tensors element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean tensor with True where x XOR y
        """
        pass
    
    @abstractmethod
    def allclose(self, x: Any, y: Any, rtol: float = 1e-5, atol: float = 1e-8) -> Any:
        """
        Check if all elements of two tensors are close within a tolerance.
        
        Args:
            x: First tensor
            y: Second tensor
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            Boolean indicating if all elements are close
        """
        pass
    
    @abstractmethod
    def isclose(self, x: Any, y: Any, rtol: float = 1e-5, atol: float = 1e-8) -> Any:
        """
        Check if elements of two tensors are close within a tolerance element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            Boolean tensor with True where elements are close
        """
        pass
    
    @abstractmethod
    def all(self, x: Any, axis: Any = None) -> Any:
        """
        Check if all elements in a tensor are True.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to perform the reduction.
                If None, reduce over all dimensions.
                
        Returns:
            Boolean tensor with True if all elements are True, False otherwise.
            If axis is specified, the result is a tensor with the specified
            axes reduced.
        """
        pass