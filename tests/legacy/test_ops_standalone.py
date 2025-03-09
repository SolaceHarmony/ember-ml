"""
Standalone test script for the ops module.

This script tests the basic functionality of the ops module without importing
the entire emberharmony package.
"""

import sys
import os
import numpy as np

# Add the parent directory to the path so we can import emberharmony
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the ops module
from ember_ml.ops import (
    get_ops, set_ops,
    tensor_ops, math_ops, device_ops, random_ops
)

def test_tensor_ops():
    """Test tensor operations."""
    set_ops('numpy')
    t_ops = tensor_ops()
    
    # Test zeros
    zeros = t_ops.zeros((2, 3))
    assert zeros.shape == (2, 3)
    assert np.all(zeros == 0)
    
    # Test ones
    ones = t_ops.ones((2, 3))
    assert ones.shape == (2, 3)
    assert np.all(ones == 1)
    
    # Test reshape
    x = t_ops.ones((6,))
    y = t_ops.reshape(x, (2, 3))
    assert y.shape == (2, 3)
    
    print("All tensor ops tests passed!")

def test_math_ops():
    """Test math operations."""
    set_ops('numpy')
    m_ops = math_ops()
    
    # Test add
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    z = m_ops.add(x, y)
    assert np.all(z == np.array([5, 7, 9]))
    
    # Test subtract
    z = m_ops.subtract(x, y)
    assert np.all(z == np.array([-3, -3, -3]))
    
    # Test multiply
    z = m_ops.multiply(x, y)
    assert np.all(z == np.array([4, 10, 18]))
    
    print("All math ops tests passed!")

def test_device_ops():
    """Test device operations."""
    set_ops('numpy')
    d_ops = device_ops()
    
    # Test get_device
    x = np.array([1, 2, 3])
    assert d_ops.get_device(x) == 'cpu'
    
    # Test to_device
    y = d_ops.to_device(x, 'cpu')
    assert np.all(y == x)
    
    print("All device ops tests passed!")

def test_random_ops():
    """Test random operations."""
    set_ops('numpy')
    r_ops = random_ops()
    
    # Test set_seed
    r_ops.set_seed(42)
    assert r_ops.get_seed() == 42
    
    # Test random_normal
    x = r_ops.random_normal((2, 3))
    assert x.shape == (2, 3)
    
    print("All random ops tests passed!")

if __name__ == "__main__":
    print("Testing tensor operations...")
    test_tensor_ops()
    
    print("\nTesting math operations...")
    test_math_ops()
    
    print("\nTesting device operations...")
    test_device_ops()
    
    print("\nTesting random operations...")
    test_random_ops()
    
    print("\nAll tests passed!")