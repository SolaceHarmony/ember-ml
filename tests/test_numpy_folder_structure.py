"""
Unit tests for the NumPy backend with the new folder structure.

This module tests that the NumPy backend works correctly with the new folder structure,
ensuring that operations can be accessed through the ops interface.
"""

import pytest
import numpy as np
from ember_ml import ops
from ember_ml.backend import get_backend, set_backend

class TestNumPyFolderStructure:
    """Tests for the NumPy backend with the new folder structure."""

    @pytest.fixture(autouse=True)
    def setup_numpy_backend(self):
        """Set up the NumPy backend for testing."""
        prev_backend = get_backend()
        set_backend('numpy')
        ops.set_ops('numpy')
        yield
        set_backend(prev_backend)
        ops.set_ops(prev_backend)

    def test_tensor_creation(self):
        """Test tensor creation operations."""
        # Test zeros
        shape = (3, 4)
        x = tensor.zeros(shape)
        assert tensor.shape(x) == shape
        assert np.allclose(tensor.to_numpy(x), np.zeros(shape))

        # Test ones
        x = tensor.ones(shape)
        assert tensor.shape(x) == shape
        assert np.allclose(tensor.to_numpy(x), np.ones(shape))

        # Test eye
        n = 3
        x = tensor.eye(n)
        assert tensor.shape(x) == (n, n)
        assert np.allclose(tensor.to_numpy(x), np.eye(n))

        # Test arange
        x = tensor.arange(10)
        assert tensor.shape(x) == (10,)
        assert np.allclose(tensor.to_numpy(x), np.arange(10))

    def test_tensor_manipulation(self):
        """Test tensor manipulation operations."""
        # Test reshape
        original_shape = (2, 6)
        new_shape = (3, 4)
        x = tensor.ones(original_shape)
        y = tensor.reshape(x, new_shape)
        assert tensor.shape(y) == new_shape
        assert np.allclose(tensor.to_numpy(y), np.ones(new_shape))

        # Test transpose
        shape = (2, 3, 4)
        x = tensor.ones(shape)
        y = tensor.transpose(x)
        # The default transpose behavior might differ between backends
        # For NumPy, it's a complete transpose, not just the last two dimensions
        expected_shape = (4, 3, 2)
        assert tensor.shape(y) == expected_shape
        assert np.allclose(tensor.to_numpy(y), np.transpose(np.ones(shape)))

        # Test concatenate
        shape1 = (2, 3)
        shape2 = (2, 3)
        x1 = tensor.ones(shape1)
        x2 = tensor.zeros(shape2)
        y = tensor.concatenate([x1, x2], axis=0)
        assert tensor.shape(y) == (4, 3)
        assert np.allclose(tensor.to_numpy(y), np.concatenate([np.ones(shape1), np.zeros(shape2)], axis=0))

    def test_math_operations(self):
        """Test math operations."""
        # Test add
        x = tensor.ones((3, 4))
        y = tensor.ones((3, 4))
        z = ops.add(x, y)
        assert tensor.shape(z) == (3, 4)
        assert np.allclose(tensor.to_numpy(z), np.ones((3, 4)) + np.ones((3, 4)))

        # Test subtract
        z = ops.subtract(x, y)
        assert tensor.shape(z) == (3, 4)
        assert np.allclose(tensor.to_numpy(z), np.ones((3, 4)) - np.ones((3, 4)))

        # Test multiply
        z = ops.multiply(x, y)
        assert tensor.shape(z) == (3, 4)
        assert np.allclose(tensor.to_numpy(z), np.ones((3, 4)) * np.ones((3, 4)))

        # Test divide
        z = ops.divide(x, y)
        assert tensor.shape(z) == (3, 4)
        assert np.allclose(tensor.to_numpy(z), np.ones((3, 4)) / np.ones((3, 4)))

        # Test matmul
        x = tensor.ones((2, 3))
        y = tensor.ones((3, 4))
        z = ops.matmul(x, y)
        assert tensor.shape(z) == (2, 4)
        assert np.allclose(tensor.to_numpy(z), np.ones((2, 3)) @ np.ones((3, 4)))

    def test_reduction_operations(self):
        """Test reduction operations."""
        # Test mean
        x = tensor.arange(12).reshape((3, 4))
        y = ops.mean(x)
        assert np.isscalar(tensor.to_numpy(y)) or tensor.shape(y) == ()
        assert np.allclose(tensor.to_numpy(y), np.mean(np.arange(12).reshape((3, 4))))

        # Test sum
        y = ops.sum(x)
        assert np.isscalar(tensor.to_numpy(y)) or tensor.shape(y) == ()
        assert np.allclose(tensor.to_numpy(y), np.sum(np.arange(12).reshape((3, 4))))

    def test_element_wise_operations(self):
        """Test element-wise operations."""
        # Test exp
        x = tensor.ones((3, 4))
        y = ops.exp(x)
        assert tensor.shape(y) == (3, 4)
        assert np.allclose(tensor.to_numpy(y), np.exp(np.ones((3, 4))))

        # Test log
        y = ops.log(x)
        assert tensor.shape(y) == (3, 4)
        assert np.allclose(tensor.to_numpy(y), np.log(np.ones((3, 4))))

        # Test sqrt
        y = ops.sqrt(x)
        assert tensor.shape(y) == (3, 4)
        assert np.allclose(tensor.to_numpy(y), np.sqrt(np.ones((3, 4))))

    def test_activation_functions(self):
        """Test activation functions."""
        # Test sigmoid
        x = tensor.ones((3, 4))
        y = ops.sigmoid(x)
        assert tensor.shape(y) == (3, 4)
        assert np.allclose(tensor.to_numpy(y), 1 / (1 + np.exp(-np.ones((3, 4)))))

        # Test relu
        x = ops.subtract(tensor.arange(12).reshape((3, 4)), 5)
        y = ops.relu(x)
        assert tensor.shape(y) == (3, 4)
        assert np.allclose(tensor.to_numpy(y), np.maximum(0, np.arange(12).reshape((3, 4)) - 5))

        # Test tanh
        x = tensor.ones((3, 4))
        y = ops.tanh(x)
        assert tensor.shape(y) == (3, 4)
        assert np.allclose(tensor.to_numpy(y), np.tanh(np.ones((3, 4))))

    def test_random_operations(self):
        """Test random operations."""
        # Test random_normal
        shape = (3, 4)
        x = tensor.random_normal(shape)
        assert tensor.shape(x) == shape

        # Test random_uniform
        x = tensor.random_uniform(shape)
        assert tensor.shape(x) == shape

        # Test seed reproducibility using backend directly
        from ember_ml.backend import get_backend_module
        backend_module = get_backend_module()
        
        # Set seed using the backend module
        if hasattr(backend_module, 'set_seed'):
            backend_module.set_seed(42)
            x1 = tensor.random_normal(shape)
            backend_module.set_seed(42)
            x2 = tensor.random_normal(shape)
            assert np.allclose(tensor.to_numpy(x1), tensor.to_numpy(x2))

    def test_comparison_operations(self):
        """Test comparison operations."""
        # Test equal
        x = tensor.ones((3, 4))
        y = tensor.ones((3, 4))
        z = ops.equal(x, y)
        assert tensor.shape(z) == (3, 4)
        assert np.all(tensor.to_numpy(z))

        # Test not_equal
        y = tensor.zeros((3, 4))
        z = ops.not_equal(x, y)
        assert tensor.shape(z) == (3, 4)
        assert np.all(tensor.to_numpy(z))

        # Test greater
        z = ops.greater(x, y)
        assert tensor.shape(z) == (3, 4)
        assert np.all(tensor.to_numpy(z))

        # Test less
        z = ops.less(y, x)
        assert tensor.shape(z) == (3, 4)
        assert np.all(tensor.to_numpy(z))

    def test_device_operations(self):
        """Test device operations."""
        # Test to_device
        x = tensor.ones((3, 4))
        y = ops.to_device(x, 'cpu')
        assert tensor.shape(y) == (3, 4)
        assert np.allclose(tensor.to_numpy(y), np.ones((3, 4)))

        # Test get_device
        device = ops.get_device(x)
        assert device == 'cpu'

    def test_dtype_operations(self):
        """Test dtype operations."""
        # Test cast
        x = tensor.ones((3, 4))
        y = tensor.cast(x, np.float32)
        assert tensor.shape(y) == (3, 4)
        assert ops.dtype(y) == np.float32
        assert np.allclose(tensor.to_numpy(y), np.ones((3, 4), dtype=np.float32))

    def test_solver_operations(self):
        """Test solver operations."""
        # Test solve
        a = tensor.eye(3)
        b = tensor.ones((3, 1))
        x = ops.solve(a, b)
        assert tensor.shape(x) == (3, 1)
        assert np.allclose(tensor.to_numpy(x), np.ones((3, 1)))