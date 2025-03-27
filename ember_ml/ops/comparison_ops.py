"""
Comparison operations interfaces.

This module defines the abstract interfaces for comparison operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class ComparisonOps(ABC):
    """Abstract interface for comparison operations."""
    
    @abstractmethod
    def equal(self, x: Any, y: Any) -> Any:
        """
        Element-wise equality comparison.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean tensor with element-wise equality result
        """
        pass
    
    @abstractmethod
    def not_equal(self, x: Any, y: Any) -> Any:
        """
        Element-wise inequality comparison.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean tensor with element-wise inequality result
        """
        pass
    
    @abstractmethod
    def less(self, x: Any, y: Any) -> Any:
        """
        Element-wise less than comparison.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean tensor with element-wise less than result
        """
        pass
    
    @abstractmethod
    def less_equal(self, x: Any, y: Any) -> Any:
        """
        Element-wise less than or equal comparison.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean tensor with element-wise less than or equal result
        """
        pass
    
    @abstractmethod
    def greater(self, x: Any, y: Any) -> Any:
        """
        Element-wise greater than comparison.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean tensor with element-wise greater than result
        """
        pass
    
    @abstractmethod
    def greater_equal(self, x: Any, y: Any) -> Any:
        """
        Element-wise greater than or equal comparison.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean tensor with element-wise greater than or equal result
        """
        pass
    
    @abstractmethod
    def logical_and(self, x: Any, y: Any) -> Any:
        """
        Element-wise logical AND.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean tensor with element-wise logical AND result
        """
        pass
    
    @abstractmethod
    def logical_or(self, x: Any, y: Any) -> Any:
        """
        Element-wise logical OR.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean tensor with element-wise logical OR result
        """
        pass
    
    @abstractmethod
    def logical_not(self, x: Any) -> Any:
        """
        Element-wise logical NOT.
        
        Args:
            x: Input tensor
            
        Returns:
            Boolean tensor with element-wise logical NOT result
        """
        pass
    
    @abstractmethod
    def logical_xor(self, x: Any, y: Any) -> Any:
        """
        Element-wise logical XOR.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean tensor with element-wise logical XOR result
        """
        pass
    
    @abstractmethod
    def allclose(self, x: Any, y: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """
        Returns True if all elements of x and y are close.
        
        Args:
            x: First tensor
            y: Second tensor
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            True if all elements are close, False otherwise
        """
        pass
    
    @abstractmethod
    def isclose(self, x: Any, y: Any, rtol: float = 1e-5, atol: float = 1e-8) -> Any:
        """
        Element-wise check if x and y are close.
        
        Args:
            x: First tensor
            y: Second tensor
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            Boolean tensor with element-wise closeness result
        """
        pass
    
    @abstractmethod
    def all(self, x: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """
        Test whether all elements evaluate to True along a given axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to perform the operation
            keepdims: Whether to keep the dimensions
            
        Returns:
            Boolean tensor with all result
        """
        pass
        
    @abstractmethod
    def isnan(self, x: Any) -> Any:
        """
        Test element-wise for NaN values.
        
        Args:
            x: Input tensor
            
        Returns:
            Boolean tensor with True where x is NaN, False otherwise
        """
        pass
        
    @abstractmethod
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """
        Return elements chosen from x or y depending on condition.
        
        Args:
            condition: Boolean tensor
            x: Tensor with values to use where condition is True
            y: Tensor with values to use where condition is False
            
        Returns:
            Tensor with values from x where condition is True, and values from y elsewhere
        """
        pass