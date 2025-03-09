"""
MLX implementation of comparison operations.

This module provides MLX implementations of comparison operations.
"""

import mlx.core as mx
from typing import Union, Sequence, Optional, Tuple, Any, List

# Type aliases
ArrayLike = Union[mx.array, float, int, list, tuple]

# Import convert_to_tensor from tensor_ops
from ember_ml.backend.mlx.tensor_ops import convert_to_tensor

def equal(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Check if two MLX arrays are equal element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where elements are equal
    """
    return mx.equal(convert_to_tensor(x), convert_to_tensor(y))

def not_equal(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Check if two MLX arrays are not equal element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where elements are not equal
    """
    return mx.not_equal(convert_to_tensor(x), convert_to_tensor(y))

def less(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Check if elements of the first MLX array are less than the second element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where elements of x are less than y
    """
    return mx.less(convert_to_tensor(x), convert_to_tensor(y))

def less_equal(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Check if elements of the first MLX array are less than or equal to the second element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where elements of x are less than or equal to y
    """
    return mx.less_equal(convert_to_tensor(x), convert_to_tensor(y))

def greater(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Check if elements of the first MLX array are greater than the second element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where elements of x are greater than y
    """
    return mx.greater(convert_to_tensor(x), convert_to_tensor(y))

def greater_equal(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Check if elements of the first MLX array are greater than or equal to the second element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where elements of x are greater than or equal to y
    """
    return mx.greater_equal(convert_to_tensor(x), convert_to_tensor(y))

def logical_and(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Compute the logical AND of two MLX arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where both x and y are True
    """
    return mx.logical_and(convert_to_tensor(x), convert_to_tensor(y))

def logical_or(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Compute the logical OR of two MLX arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where either x or y is True
    """
    return mx.logical_or(convert_to_tensor(x), convert_to_tensor(y))

def logical_not(x: ArrayLike) -> mx.array:
    """
    Compute the logical NOT of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Boolean array with True where x is False
    """
    return mx.logical_not(convert_to_tensor(x))

def logical_xor(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Compute the logical XOR of two MLX arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where exactly one of x or y is True
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return mx.bitwise_xor(x_tensor, y_tensor)


def allclose(x: ArrayLike, y: ArrayLike, rtol: float = 1e-5, atol: float = 1e-8) -> mx.array:
    """
    Check if all elements of two MLX arrays are close within a tolerance.
    
    Args:
        x: First array
        y: Second array
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Boolean indicating if all elements are close
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return mx.allclose(x_tensor, y_tensor, rtol=rtol, atol=atol)


def isclose(x: ArrayLike, y: ArrayLike, rtol: float = 1e-5, atol: float = 1e-8) -> mx.array:
    """
    Check if elements of two MLX arrays are close within a tolerance element-wise.
    
    Args:
        x: First array
        y: Second array
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Boolean array with True where elements are close
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    # Implement isclose using the formula: |x - y| <= atol + rtol * |y|
    abs_diff = mx.abs(mx.subtract(x_tensor, y_tensor))
    tolerance = mx.add(atol, mx.multiply(rtol, mx.abs(y_tensor)))
    return mx.less_equal(abs_diff, tolerance)


def all(x: ArrayLike, axis: Any = None, keepdims: bool = False) -> mx.array:
    """
    Check if all elements in a tensor are True.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to perform the reduction.
            If None, reduce over all dimensions.
        keepdims: Keep reduced axes as singleton dimensions, defaults to False.
            
    Returns:
        Boolean tensor with True if all elements are True, False otherwise.
        If axis is specified, the result is a tensor with the specified
        axes reduced.
    """
    x_tensor = convert_to_tensor(x)
    return mx.all(x_tensor, axis=axis, keepdims=keepdims)


class MLXComparisonOps:
    """MLX implementation of comparison operations."""
    
    def equal(self, x, y):
        """Check if two tensors are equal element-wise."""
        return equal(x, y)
    
    def not_equal(self, x, y):
        """Check if two tensors are not equal element-wise."""
        return not_equal(x, y)
    
    def less(self, x, y):
        """Check if elements of the first tensor are less than the second element-wise."""
        return less(x, y)
    
    def less_equal(self, x, y):
        """Check if elements of the first tensor are less than or equal to the second element-wise."""
        return less_equal(x, y)
    
    def greater(self, x, y):
        """Check if elements of the first tensor are greater than the second element-wise."""
        return greater(x, y)
    
    def greater_equal(self, x, y):
        """Check if elements of the first tensor are greater than or equal to the second element-wise."""
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
    
    def all(self, x, axis=None, keepdims=False):
        """Check if all elements in a tensor are True."""
        return all(x, axis=axis, keepdims=keepdims)