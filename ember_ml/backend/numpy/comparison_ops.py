"""
NumPy implementation of comparison operations.

This module provides NumPy implementations of comparison operations.
"""

import numpy as np
from typing import Union, Any

# Import from tensor_ops
from ember_ml.backend.numpy.tensor import NumpyTensor


def equal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Check if two tensors are equal element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean NumPy array with True where x == y
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.equal(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


def not_equal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Check if two tensors are not equal element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean NumPy array with True where x != y
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.not_equal(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


def less(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Check if one tensor is less than another element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean NumPy array with True where x < y
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.less(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


def less_equal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Check if one tensor is less than or equal to another element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean NumPy array with True where x <= y
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.less_equal(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


def greater(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Check if one tensor is greater than another element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean NumPy array with True where x > y
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.greater(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


def greater_equal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Check if one tensor is greater than or equal to another element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean NumPy array with True where x >= y
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.greater_equal(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


def logical_and(x: Any, y: Any) -> np.ndarray:
    """
    Compute the logical AND of two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean NumPy array with True where x AND y
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.logical_and(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


def logical_or(x: Any, y: Any) -> np.ndarray:
    """
    Compute the logical OR of two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean NumPy array with True where x OR y
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.logical_or(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


def logical_not(x: Any) -> np.ndarray:
    """
    Compute the logical NOT of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Boolean NumPy array with True where NOT x
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.logical_not(Tensor.convert_to_tensor(x))


def logical_xor(x: Any, y: Any) -> np.ndarray:
    """
    Compute the logical XOR of two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean NumPy array with True where x XOR y
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.logical_xor(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


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
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.allclose(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y), rtol=rtol, atol=atol)

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
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.isclose(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y), rtol=rtol, atol=atol)


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
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.all(Tensor.convert_to_tensor(x), axis=axis)


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
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.where(condition, Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


class NumpyComparisonOps:
    """NumPy implementation of comparison operations."""
    
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
    
    def all(self, x, axis=None):
        """Check if all elements in a tensor are True."""
        return all(x, axis=axis)
        
    def where(self, condition, x, y):
        """Return elements chosen from x or y depending on condition."""
        return where(condition, x, y)