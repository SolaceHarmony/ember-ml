"""
NumPy implementation of comparison operations.

This module provides NumPy implementations of comparison operations.
"""

import numpy as np
from typing import Union, Any

# Import from tensor_ops
from ember_ml.backend.numpy.tensor import NumpyTensor

convert_to_tensor = NumpyTensor().convert_to_tensor

def equal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Check if two tensors are equal element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean NumPy array with True where x == y
    """
    return np.equal(x, y)


def not_equal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Check if two tensors are not equal element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean NumPy array with True where x != y
    """
    return np.not_equal(x, y)


def less(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Check if one tensor is less than another element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean NumPy array with True where x < y
    """
    return np.less(x, y)


def less_equal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Check if one tensor is less than or equal to another element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean NumPy array with True where x <= y
    """
    return np.less_equal(x, y)


def greater(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Check if one tensor is greater than another element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean NumPy array with True where x > y
    """
    return np.greater(x, y)


def greater_equal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Check if one tensor is greater than or equal to another element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean NumPy array with True where x >= y
    """
    return np.greater_equal(x, y)


def logical_and(x: Any, y: Any) -> np.ndarray:
    """
    Compute the logical AND of two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean NumPy array with True where x AND y
    """
    return np.logical_and(x, y)


def logical_or(x: Any, y: Any) -> np.ndarray:
    """
    Compute the logical OR of two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean NumPy array with True where x OR y
    """
    return np.logical_or(x, y)


def logical_not(x: Any) -> np.ndarray:
    """
    Compute the logical NOT of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Boolean NumPy array with True where NOT x
    """
    return np.logical_not(x)


def logical_xor(x: Any, y: Any) -> np.ndarray:
    """
    Compute the logical XOR of two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean NumPy array with True where x XOR y
    """
    return np.logical_xor(x, y)


def allclose(x: Any, y: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
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
    return np.allclose(x, y, rtol=rtol, atol=atol)

def isclose(x: Any, y: Any, rtol: float = 1e-5, atol: float = 1e-8) -> np.ndarray:
    """
    Check if elements of two tensors are close within a tolerance element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Boolean NumPy array with True where elements are close
    """
    return np.isclose(x, y, rtol=rtol, atol=atol)


def all(x: Any, axis: Any = None) -> Any:
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
    return np.all(x, axis=axis)


def where(condition: Any, x: Any, y: Any) -> np.ndarray:
    """
    Return elements chosen from x or y depending on condition.
    
    Args:
        condition: Boolean tensor
        x: Tensor with values to use where condition is True
        y: Tensor with values to use where condition is False
        
    Returns:
        Tensor with values from x where condition is True, and values from y elsewhere
    """
    return np.where(condition, x, y)


class NumpyComparisonOps:
    """NumPy implementation of comparison operations."""
    
    def equal(self, x, y):
        """Check if two tensors are equal element-wise."""
        return equal(x, y)
    
    def not_equal(self, x, y):
        """Check if two tensors are not equal element-wise."""
        return not_equal(x, y)
    
    def less(self, x, y):
        """Check if one tensor is less than another element-wise."""
        return less(x, y)
    
    def less_equal(self, x, y):
        """Check if one tensor is less than or equal to another element-wise."""
        return less_equal(x, y)
    
    def greater(self, x, y):
        """Check if one tensor is greater than another element-wise."""
        return greater(x, y)
    
    def greater_equal(self, x, y):
        """Check if one tensor is greater than or equal to another element-wise."""
        return greater_equal(x, y)
    
    def logical_and(self, x, y):
        """Compute the logical AND of two tensors element-wise."""
        return logical_and(x, y)
    
    def logical_or(self, x, y):
        """Compute the logical OR of two tensors element-wise."""
        return logical_or(x, y)
    
    def logical_not(self, x):
        """Compute the logical NOT of a tensor element-wise."""
        return logical_not(x)
    
    def logical_xor(self, x, y):
        """Compute the logical XOR of two tensors element-wise."""
        return logical_xor(x, y)
    
    def allclose(self, x, y, rtol=1e-5, atol=1e-8):
        """Check if all elements of two tensors are close within a tolerance."""
        return allclose(x, y, rtol=rtol, atol=atol)
    
    def isclose(self, x, y, rtol=1e-5, atol=1e-8):
        """Check if elements of two tensors are close within a tolerance element-wise."""
        return isclose(x, y, rtol=rtol, atol=atol)
    
    def all(self, x, axis=None):
        """Check if all elements in a tensor are True."""
        return all(x, axis=axis)
        
    def where(self, condition, x, y):
        """Return elements chosen from x or y depending on condition."""
        return where(condition, x, y)