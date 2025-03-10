"""
Direct test script for NumPy tensor operations.

This script directly tests the NumPy tensor operations by copying the necessary
code from the tensor_ops.py file, without importing from the ember_ml framework.
"""

import numpy as np
from typing import Optional, Union, Tuple, List, Any, Sequence, Type

# Type aliases
ArrayLike = Union[np.ndarray, float, int, list, tuple]
Shape = Union[int, Sequence[int]]
DType = Union[np.dtype, str, None]

def convert_to_tensor(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
    """Convert input to a NumPy array."""
    return np.asarray(x, dtype=dtype)

def zeros(shape: Shape, dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
    """Create a NumPy array of zeros."""
    return np.zeros(shape, dtype=dtype)

def ones(shape: Shape, dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
    """Create a NumPy array of ones."""
    return np.ones(shape, dtype=dtype)

def eye(n: int, m: Optional[int] = None, dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
    """Create a NumPy identity matrix."""
    return np.eye(n, m, dtype=dtype)

def reshape(x: ArrayLike, shape: Shape) -> np.ndarray:
    """Reshape a NumPy array to a new shape."""
    return np.reshape(x, shape)

def transpose(x: ArrayLike, axes: Optional[Sequence[int]] = None) -> np.ndarray:
    """Permute the dimensions of a NumPy array."""
    return np.transpose(x, axes)

def stack(arrays: Sequence[ArrayLike], axis: int = 0) -> np.ndarray:
    """Stack NumPy arrays along a new axis."""
    return np.stack([convert_to_tensor(arr) for arr in arrays], axis=axis)

def concatenate(arrays: Sequence[ArrayLike], axis: int = 0) -> np.ndarray:
    """Concatenate NumPy arrays along a specified axis."""
    return np.concatenate([convert_to_tensor(arr) for arr in arrays], axis=axis)

def test_zeros():
    """Test zeros function."""
    result = zeros((2, 3))
    print("zeros((2, 3)):")
    print(result)
    print()

def test_ones():
    """Test ones function."""
    result = ones((2, 3))
    print("ones((2, 3)):")
    print(result)
    print()

def test_eye():
    """Test eye function."""
    result = eye(3)
    print("eye(3):")
    print(result)
    print()

def test_reshape():
    """Test reshape function."""
    x = ones((2, 3))
    result = reshape(x, (3, 2))
    print("reshape(ones((2, 3)), (3, 2)):")
    print(result)
    print()

def test_transpose():
    """Test transpose function."""
    x = ones((2, 3))
    result = transpose(x)
    print("transpose(ones((2, 3))):")
    print(result)
    print()

def test_stack():
    """Test stack function."""
    x = zeros((2, 2))
    y = ones((2, 2))
    result = stack([x, y])
    print("stack([zeros((2, 2)), ones((2, 2))]):")
    print(result)
    print()

def test_concatenate():
    """Test concatenate function."""
    x = zeros((2, 2))
    y = ones((2, 2))
    result = concatenate([x, y], axis=0)
    print("concatenate([zeros((2, 2)), ones((2, 2))], axis=0):")
    print(result)
    print()

if __name__ == "__main__":
    print("Testing NumPy tensor operations directly...\n")
    test_zeros()
    test_ones()
    test_eye()
    test_reshape()
    test_transpose()
    test_stack()
    test_concatenate()
    print("All tests completed successfully!")