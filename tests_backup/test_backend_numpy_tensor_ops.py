"""
Test script for NumPy tensor operations.

This script directly tests the NumPy tensor operations without going through
the ember_ml framework.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath('.'))

# Import NumPy tensor operations directly
from ember_ml.backend.numpy.tensor.ops.creation import zeros, ones, eye
from ember_ml.backend.numpy.tensor.ops.manipulation import reshape, transpose, stack, concatenate

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
    print("Testing NumPy tensor operations...\n")
    test_zeros()
    test_ones()
    test_eye()
    test_reshape()
    test_transpose()
    test_stack()
    test_concatenate()
    print("All tests completed successfully!")