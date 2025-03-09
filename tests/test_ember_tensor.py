"""
Test the EmberTensor class.

This module tests the EmberTensor class to ensure it works correctly
across different backends.
"""

import pytest
import numpy as np

from ember_ml import ops
from ember_ml.ops.tensor import EmberTensor
from ember_ml.backend import get_backend, set_backend

@pytest.fixture
def original_backend():
    """Fixture to save and restore the original backend."""
    original = get_backend()
    yield original
    set_backend(original)

class TestEmberTensor:
    """Test the EmberTensor class."""
    
    def test_creation(self):
        """Test creating an EmberTensor."""
        # Create a tensor from a list
        tensor = EmberTensor([1, 2, 3])
        assert tensor.shape == (3,)
        
        # Create a tensor with a specific shape
        tensor = EmberTensor.zeros((2, 3))
        assert tensor.shape == (2, 3)
        
        # Create a tensor with a specific dtype
        tensor = EmberTensor.ones((2, 3), dtype=ops.float32)
        assert tensor.dtype == ops.float32
    
    def test_numpy_conversion(self):
        """Test converting an EmberTensor to a NumPy array."""
        # Create a tensor from a list
        tensor = EmberTensor([1, 2, 3])
        
        # Convert to NumPy
        array = tensor.numpy()
        assert isinstance(array, np.ndarray)
        assert array.shape == (3,)
        np.testing.assert_array_equal(array, np.array([1, 2, 3]))
    
    def test_arithmetic_operations(self):
        """Test arithmetic operations on EmberTensor."""
        # Create tensors
        a = EmberTensor([1, 2, 3])
        b = EmberTensor([4, 5, 6])
        
        # Test addition
        c = a + b
        assert isinstance(c, EmberTensor)
        np.testing.assert_array_equal(c.numpy(), np.array([5, 7, 9]))
        
        # Test subtraction
        c = a - b
        assert isinstance(c, EmberTensor)
        np.testing.assert_array_equal(c.numpy(), np.array([-3, -3, -3]))
        
        # Test multiplication
        c = a * b
        assert isinstance(c, EmberTensor)
        np.testing.assert_array_equal(c.numpy(), np.array([4, 10, 18]))
        
        # Test division
        c = a / b
        assert isinstance(c, EmberTensor)
        np.testing.assert_array_almost_equal(c.numpy(), np.array([0.25, 0.4, 0.5]))
    
    def test_comparison_operations(self):
        """Test comparison operations on EmberTensor."""
        # Create tensors
        a = EmberTensor([1, 2, 3])
        b = EmberTensor([2, 2, 2])
        
        # Test equality
        c = a == b
        assert isinstance(c, EmberTensor)
        np.testing.assert_array_equal(c.numpy(), np.array([False, True, False]))
        
        # Test inequality
        c = a != b
        assert isinstance(c, EmberTensor)
        np.testing.assert_array_equal(c.numpy(), np.array([True, False, True]))
        
        # Test less than
        c = a < b
        assert isinstance(c, EmberTensor)
        np.testing.assert_array_equal(c.numpy(), np.array([True, False, False]))
        
        # Test greater than
        c = a > b
        assert isinstance(c, EmberTensor)
        np.testing.assert_array_equal(c.numpy(), np.array([False, False, True]))
    
    def test_shape_operations(self):
        """Test shape operations on EmberTensor."""
        # Create a tensor
        a = EmberTensor([[1, 2, 3], [4, 5, 6]])
        
        # Test reshape
        b = a.reshape((3, 2))
        assert isinstance(b, EmberTensor)
        assert b.shape == (3, 2)
        
        # Test transpose
        b = a.transpose()
        assert isinstance(b, EmberTensor)
        assert b.shape == (3, 2)
        
        # Test squeeze
        a = EmberTensor([[[1], [2]]])
        b = a.squeeze()
        assert isinstance(b, EmberTensor)
        assert b.shape == (2,)
        
        # Test unsqueeze
        a = EmberTensor([1, 2])
        b = a.unsqueeze(0)
        assert isinstance(b, EmberTensor)
        assert b.shape == (1, 2)
    
    def test_reduction_operations(self):
        """Test reduction operations on EmberTensor."""
        # Create a tensor
        a = EmberTensor([[1, 2, 3], [4, 5, 6]])
        
        # Test sum
        b = a.sum()
        assert isinstance(b, EmberTensor)
        assert b.numpy() == 21
        
        # Test mean
        b = a.mean()
        assert isinstance(b, EmberTensor)
        assert b.numpy() == 3.5
        
        # Test max
        b = a.max()
        assert isinstance(b, EmberTensor)
        assert b.numpy() == 6
        
        # Test min
        b = a.min()
        assert isinstance(b, EmberTensor)
        assert b.numpy() == 1
    
    def test_static_methods(self):
        """Test static methods for tensor creation."""
        # Test zeros
        a = EmberTensor.zeros((2, 3))
        assert isinstance(a, EmberTensor)
        assert a.shape == (2, 3)
        np.testing.assert_array_equal(a.numpy(), np.zeros((2, 3)))
        
        # Test ones
        a = EmberTensor.ones((2, 3))
        assert isinstance(a, EmberTensor)
        assert a.shape == (2, 3)
        np.testing.assert_array_equal(a.numpy(), np.ones((2, 3)))
        
        # Test full
        a = EmberTensor.full((2, 3), 5)
        assert isinstance(a, EmberTensor)
        assert a.shape == (2, 3)
        np.testing.assert_array_equal(a.numpy(), np.full((2, 3), 5))
        
        # Test arange
        a = EmberTensor.arange(0, 5)
        assert isinstance(a, EmberTensor)
        np.testing.assert_array_equal(a.numpy(), np.arange(0, 5))
        
        # Test linspace
        a = EmberTensor.linspace(0, 1, 5)
        assert isinstance(a, EmberTensor)
        np.testing.assert_array_almost_equal(a.numpy(), np.linspace(0, 1, 5))
        
        # Test eye
        a = EmberTensor.eye(3)
        assert isinstance(a, EmberTensor)
        np.testing.assert_array_equal(a.numpy(), np.eye(3))
    
    def test_backend_specific(self, original_backend):
        """Test backend-specific behavior."""
        # Test with NumPy backend
        set_backend('numpy')
        a = EmberTensor([1, 2, 3])
        assert a.backend == 'numpy'
        
        # Test with PyTorch backend if available
        try:
            import torch
            set_backend('torch')
            a = EmberTensor([1, 2, 3])
            assert a.backend == 'torch'
        except ImportError:
            pytest.skip("PyTorch not available")
        
        # Test with MLX backend if available
        try:
            import mlx.core
            set_backend('mlx')
            a = EmberTensor([1, 2, 3])
            assert a.backend == 'mlx'
        except ImportError:
            pytest.skip("MLX not available")