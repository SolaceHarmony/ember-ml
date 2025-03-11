"""
PyTorch comparison operations for ember_ml.

This module provides PyTorch implementations of comparison operations.
"""

import torch
from typing import Any

# Import from tensor_ops
from ember_ml.backend.torch.tensor_ops import convert_to_tensor


def equal(x: Any, y: Any) -> torch.Tensor:
    """
    Check if two tensors are equal element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean tensor with True where x == y
    """
    return torch.eq(convert_to_tensor(x), convert_to_tensor(y))


def not_equal(x: Any, y: Any) -> torch.Tensor:
    """
    Check if two tensors are not equal element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean tensor with True where x != y
    """
    return torch.ne(convert_to_tensor(x), convert_to_tensor(y))


def less(x: Any, y: Any) -> torch.Tensor:
    """
    Check if one tensor is less than another element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean tensor with True where x < y
    """
    return torch.lt(convert_to_tensor(x), convert_to_tensor(y))


def less_equal(x: Any, y: Any) -> torch.Tensor:
    """
    Check if one tensor is less than or equal to another element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean tensor with True where x <= y
    """
    return torch.le(convert_to_tensor(x), convert_to_tensor(y))


def greater(x: Any, y: Any) -> torch.Tensor:
    """
    Check if one tensor is greater than another element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean tensor with True where x > y
    """
    return torch.gt(convert_to_tensor(x), convert_to_tensor(y))


def greater_equal(x: Any, y: Any) -> torch.Tensor:
    """
    Check if one tensor is greater than or equal to another element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean tensor with True where x >= y
    """
    return torch.ge(convert_to_tensor(x), convert_to_tensor(y))


def logical_and(x: Any, y: Any) -> torch.Tensor:
    """
    Compute the logical AND of two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean tensor with True where x AND y
    """
    return torch.logical_and(convert_to_tensor(x), convert_to_tensor(y))


def logical_or(x: Any, y: Any) -> torch.Tensor:
    """
    Compute the logical OR of two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean tensor with True where x OR y
    """
    return torch.logical_or(convert_to_tensor(x), convert_to_tensor(y))


def logical_not(x: Any) -> torch.Tensor:
    """
    Compute the logical NOT of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Boolean tensor with True where NOT x
    """
    return torch.logical_not(convert_to_tensor(x))


def logical_xor(x: Any, y: Any) -> torch.Tensor:
    """
    Compute the logical XOR of two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean tensor with True where x XOR y
    """
    return torch.logical_xor(convert_to_tensor(x), convert_to_tensor(y))


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
    return torch.allclose(convert_to_tensor(x), convert_to_tensor(y), rtol=rtol, atol=atol)

def isclose(x: Any, y: Any, rtol: float = 1e-5, atol: float = 1e-8) -> torch.Tensor:
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
    return torch.isclose(convert_to_tensor(x), convert_to_tensor(y), rtol=rtol, atol=atol)


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
    x_tensor = convert_to_tensor(x)
    if axis is None:
        return torch.all(x_tensor)
    else:
        return torch.all(x_tensor, dim=axis)


def where(condition: Any, x: Any, y: Any) -> torch.Tensor:
    """
    Return elements chosen from x or y depending on condition.
    
    Args:
        condition: Boolean tensor
        x: Tensor with values to use where condition is True
        y: Tensor with values to use where condition is False
        
    Returns:
        Tensor with values from x where condition is True, and values from y elsewhere
    """
    return torch.where(
        convert_to_tensor(condition),
        convert_to_tensor(x),
        convert_to_tensor(y)
    )


class TorchComparisonOps:
    """PyTorch implementation of comparison operations."""
    
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