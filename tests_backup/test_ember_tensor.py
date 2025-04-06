"""
Test the EmberTensor class.

This module tests the EmberTensor class to ensure it works correctly
across different backends.
"""

import pytest

from ember_ml import ops
from ember_ml.nn.tensor import (
    EmberTensor, EmberDType, float32, array, convert_to_tensor,
    zeros, ones, eye, arange, linspace, full
)

@pytest.fixture
def original_backend():
    """Fixture to save and restore the original backend."""
    original = ops.get_backend()
    yield original
    # Ensure original is not None before setting it
    if original is not None:
        ops.set_backend(original)
    else:
        # Default to 'numpy' if original is None
        ops.set_backend('numpy')

class TestEmberTensor:
    """Test the EmberTensor class."""
    
    def test_creation(self):
        """Test creating an EmberTensor."""
        # Create a tensor from a list
        tensor = EmberTensor([1, 2, 3])
        assert tensor.shape == (3,)
        
        # Create a tensor with a specific shape
        tensor_obj = EmberTensor([0])  # Create a temporary tensor object
        tensor = tensor_obj.zeros((2, 3))
        assert tensor.shape == (2, 3)
        
        # Create a tensor with a specific dtype
        tensor_obj = EmberTensor([0])  # Create a temporary tensor object
        tensor = tensor_obj.ones((2, 3), dtype=float32)
        # Use string comparison for backend purity
        assert str(tensor.dtype).split('.')[-1] == 'float32'    
    def test_numpy_conversion(self):
        """Test converting an EmberTensor to a NumPy array."""
        # Create a tensor from a list
        tensor = EmberTensor([1, 2, 3])
        
        # Convert to NumPy
        array = tensor.numpy()
        # Check shape without direct NumPy references
        assert array.shape == (3,)
        # Check values by comparing with expected values
        expected = convert_to_tensor([1, 2, 3])
        # Use ops.equal and ops.all for comparison
        assert ops.all(ops.equal(tensor.to_backend_tensor(), expected))
    
    def test_arithmetic_operations(self):
        """Test arithmetic operations on EmberTensor."""
        # Create tensors
        a = EmberTensor([1, 2, 3])
        b = EmberTensor([4, 5, 6])
        
        # Test addition using ops functions
        c = EmberTensor(ops.add(a.to_backend_tensor(), b.to_backend_tensor()))
        assert isinstance(c, EmberTensor)
        expected = convert_to_tensor([5, 7, 9])
        assert ops.all(ops.equal(c.to_backend_tensor(), expected))
        
        # Test subtraction using ops functions
        c = EmberTensor(ops.subtract(a.to_backend_tensor(), b.to_backend_tensor()))
        assert isinstance(c, EmberTensor)
        expected = convert_to_tensor([-3, -3, -3])
        assert ops.all(ops.equal(c.to_backend_tensor(), expected))
        
        # Test multiplication using ops functions
        c = EmberTensor(ops.multiply(a.to_backend_tensor(), b.to_backend_tensor()))
        assert isinstance(c, EmberTensor)
        expected = convert_to_tensor([4, 10, 18])
        assert ops.all(ops.equal(c.to_backend_tensor(), expected))
        
        # Test division using ops functions
        c = EmberTensor(ops.divide(a.to_backend_tensor(), b.to_backend_tensor()))
        assert isinstance(c, EmberTensor)
        expected = convert_to_tensor([0.25, 0.4, 0.5])
        assert ops.allclose(c.to_backend_tensor(), expected, rtol=1e-5, atol=1e-8)
    
    def test_comparison_operations(self):
        """Test comparison operations on EmberTensor."""
        # Create tensors
        a = EmberTensor([1, 2, 3])
        b = EmberTensor([2, 2, 2])
        
        # Test equality using ops functions
        c = EmberTensor(ops.equal(a.to_backend_tensor(), b.to_backend_tensor()))
        assert isinstance(c, EmberTensor)
        expected = convert_to_tensor([False, True, False])
        assert ops.all(ops.equal(c.to_backend_tensor(), expected))
        
        # Test inequality using ops functions
        c = EmberTensor(ops.not_equal(a.to_backend_tensor(), b.to_backend_tensor()))
        assert isinstance(c, EmberTensor)
        expected = convert_to_tensor([True, False, True])
        assert ops.all(ops.equal(c.to_backend_tensor(), expected))
        
        # Test less than using ops functions
        c = EmberTensor(ops.less(a.to_backend_tensor(), b.to_backend_tensor()))
        assert isinstance(c, EmberTensor)
        expected = convert_to_tensor([True, False, False])
        assert ops.all(ops.equal(c.to_backend_tensor(), expected))
        
        # Test greater than using ops functions
        c = EmberTensor(ops.greater(a.to_backend_tensor(), b.to_backend_tensor()))
        assert isinstance(c, EmberTensor)
        expected = convert_to_tensor([False, False, True])
        assert ops.all(ops.equal(c.to_backend_tensor(), expected))
    
    def test_shape_operations(self):
        """Test shape operations on EmberTensor."""
        # Create a tensor
        a = EmberTensor([[1, 2, 3], [4, 5, 6]])
        
        # Test reshape
        tensor_obj = EmberTensor([0])  # Create a temporary tensor object
        b = tensor_obj.reshape(a.to_backend_tensor(), (3, 2))
        assert isinstance(b, EmberTensor)
        assert b.shape == (3, 2)
        
        # Test transpose
        tensor_obj = EmberTensor([0])  # Create a temporary tensor object
        b = tensor_obj.transpose(a.to_backend_tensor())
        assert isinstance(b, EmberTensor)
        assert b.shape == (3, 2)
        
        # Test squeeze
        a = EmberTensor([[[1], [2]]])
        tensor_obj = EmberTensor([0])  # Create a temporary tensor object
        b = tensor_obj.squeeze(a.to_backend_tensor())
        assert isinstance(b, EmberTensor)
        assert b.shape == (2,)
        
        # Test expand_dims (equivalent to unsqueeze)
        a = EmberTensor([1, 2])
        tensor_obj = EmberTensor([0])  # Create a temporary tensor object
        b = tensor_obj.expand_dims(a.to_backend_tensor(), 0)
        assert isinstance(b, EmberTensor)
        assert b.shape == (1, 2)
    
    def test_reduction_operations(self):
        """Test reduction operations on EmberTensor."""
        # Create a tensor
        a = EmberTensor([[1, 2, 3], [4, 5, 6]])
        
        # Test sum using ops functions
        b = EmberTensor(ops.sum(a.to_backend_tensor()))
        assert isinstance(b, EmberTensor)
        assert b.numpy() == 21
        
        # Test mean using ops functions
        b = EmberTensor(ops.mean(a.to_backend_tensor()))
        assert isinstance(b, EmberTensor)
        assert b.numpy() == 3.5
        
        # Test max using ops functions
        b = EmberTensor(ops.stats.max(a.to_backend_tensor()))
        assert isinstance(b, EmberTensor)
        assert b.numpy() == 6
        
        # Test min using ops functions
        b = EmberTensor(ops.stats.min(a.to_backend_tensor()))
        assert isinstance(b, EmberTensor)
        assert b.numpy() == 1
    
    def test_static_methods(self):
        """Test static methods for tensor creation."""
        # Create a temporary tensor object to use its methods
        tensor_obj = EmberTensor([0])
        
        # Test zeros
        a = tensor_obj.zeros((2, 3))
        assert isinstance(a, EmberTensor)
        assert a.shape == (2, 3)
        expected = zeros((2, 3))
        assert ops.all(ops.equal(a.to_backend_tensor(), expected))
        
        # Test ones
        a = tensor_obj.ones((2, 3))
        assert isinstance(a, EmberTensor)
        assert a.shape == (2, 3)
        expected = ones((2, 3))
        assert ops.all(ops.equal(a.to_backend_tensor(), expected))
        
        # Test full
        a = tensor_obj.full((2, 3), 5)
        assert isinstance(a, EmberTensor)
        assert a.shape == (2, 3)
        expected = full((2, 3), 5)
        assert ops.all(ops.equal(a.to_backend_tensor(), expected))
        
        # Test arange
        a = tensor_obj.arange(0, 5)
        assert isinstance(a, EmberTensor)
        expected = arange(0, 5)
        assert ops.all(ops.equal(a.to_backend_tensor(), expected))
        
        # Test linspace
        a = tensor_obj.linspace(0, 1, 5)
        assert isinstance(a, EmberTensor)
        expected = linspace(0, 1, 5)
        assert ops.allclose(a.to_backend_tensor(), expected, rtol=1e-5, atol=1e-8)
        
        # Test eye
        a = tensor_obj.eye(3)
        assert isinstance(a, EmberTensor)
        expected = eye(3)
        assert ops.all(ops.equal(a.to_backend_tensor(), expected))
    
    def test_backend_specific(self, original_backend):
        """Test backend-specific behavior."""
        # Test with NumPy backend
        ops.set_backend('numpy')
        a = EmberTensor([1, 2, 3])
        # Check the backend indirectly through the dtype string representation
        assert 'float' in str(a.dtype) or 'int' in str(a.dtype)
        
        # Test with PyTorch backend if available
        try:
            import torch
            ops.set_backend('torch')
            a = EmberTensor([1, 2, 3])
            # Check the backend indirectly through the dtype string representation
            assert 'float' in str(a.dtype) or 'int' in str(a.dtype)
        except ImportError:
            pytest.skip("PyTorch not available")
        
        # Test with MLX backend if available
        try:
            import mlx.core
            ops.set_backend('mlx')
            a = EmberTensor([1, 2, 3])
            # Check the backend indirectly through the dtype string representation
            assert 'float' in str(a.dtype) or 'int' in str(a.dtype)
        except ImportError:
            pytest.skip("MLX not available")