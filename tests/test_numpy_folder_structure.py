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
        assert ops.shape(x) == shape
        assert np.allclose(ops.to_numpy(x), np.zeros(shape))

        # Test ones
        x = ops.ones(shape)
        assert ops.shape(x) == shape
        assert np.allclose(ops.to_numpy(x), np.ones(shape))

        # Test eye
        n = 3
        x = ops.eye(n)
        assert ops.shape(x) == (n, n)
        assert np.allclose(ops.to_numpy(x), np.eye(n))

        # Test arange
        x = ops.arange(10)
        assert ops.shape(x) == (10,)
        assert np.allclose(ops.to_numpy(x), np.arange(10))

    def test_tensor_manipulation(self):
        """Test tensor manipulation operations."""
        # Test reshape
        original_shape = (2, 6)
        new_shape = (3, 4)
        x = ops.ones(original_shape)
        y = ops.reshape(x, new_shape)
        assert ops.shape(y) == new_shape
        assert np.allclose(ops.to_numpy(y), np.ones(new_shape))

        # Test transpose
        shape = (2, 3, 4)
        x = ops.ones(shape)
        y = ops.transpose(x)
        # The default transpose behavior might differ between backends
        # For NumPy, it's a complete transpose, not just the last two dimensions
        expected_shape = (4, 3, 2)
        assert ops.shape(y) == expected_shape
        assert np.allclose(ops.to_numpy(y), np.transpose(np.ones(shape)))

        # Test concatenate
        shape1 = (2, 3)
        shape2 = (2, 3)
        x1 = ops.ones(shape1)
        x2 = tensor.zeros(shape2)
        y = ops.concatenate([x1, x2], axis=0)
        assert ops.shape(y) == (4, 3)
        assert np.allclose(ops.to_numpy(y), np.concatenate([np.ones(shape1), np.zeros(shape2)], axis=0))

    def test_math_operations(self):
        """Test math operations."""
        # Test add
        x = ops.ones((3, 4))
        y = ops.ones((3, 4))
        z = ops.add(x, y)
        assert ops.shape(z) == (3, 4)
        assert np.allclose(ops.to_numpy(z), np.ones((3, 4)) + np.ones((3, 4)))

        # Test subtract
        z = ops.subtract(x, y)
        assert ops.shape(z) == (3, 4)
        assert np.allclose(ops.to_numpy(z), np.ones((3, 4)) - np.ones((3, 4)))

        # Test multiply
        z = ops.multiply(x, y)
        assert ops.shape(z) == (3, 4)
        assert np.allclose(ops.to_numpy(z), np.ones((3, 4)) * np.ones((3, 4)))

        # Test divide
        z = ops.divide(x, y)
        assert ops.shape(z) == (3, 4)
        assert np.allclose(ops.to_numpy(z), np.ones((3, 4)) / np.ones((3, 4)))

        # Test matmul
        x = ops.ones((2, 3))
        y = ops.ones((3, 4))
        z = ops.matmul(x, y)
        assert ops.shape(z) == (2, 4)
        assert np.allclose(ops.to_numpy(z), np.ones((2, 3)) @ np.ones((3, 4)))

    def test_reduction_operations(self):
        """Test reduction operations."""
        # Test mean
        x = ops.arange(12).reshape((3, 4))
        y = ops.mean(x)
        assert np.isscalar(ops.to_numpy(y)) or ops.shape(y) == ()
        assert np.allclose(ops.to_numpy(y), np.mean(np.arange(12).reshape((3, 4))))

        # Test sum
        y = ops.sum(x)
        assert np.isscalar(ops.to_numpy(y)) or ops.shape(y) == ()
        assert np.allclose(ops.to_numpy(y), np.sum(np.arange(12).reshape((3, 4))))

    def test_element_wise_operations(self):
        """Test element-wise operations."""
        # Test exp
        x = ops.ones((3, 4))
        y = ops.exp(x)
        assert ops.shape(y) == (3, 4)
        assert np.allclose(ops.to_numpy(y), np.exp(np.ones((3, 4))))

        # Test log
        y = ops.log(x)
        assert ops.shape(y) == (3, 4)
        assert np.allclose(ops.to_numpy(y), np.log(np.ones((3, 4))))

        # Test sqrt
        y = ops.sqrt(x)
        assert ops.shape(y) == (3, 4)
        assert np.allclose(ops.to_numpy(y), np.sqrt(np.ones((3, 4))))

    def test_activation_functions(self):
        """Test activation functions."""
        # Test sigmoid
        x = ops.ones((3, 4))
        y = ops.sigmoid(x)
        assert ops.shape(y) == (3, 4)
        assert np.allclose(ops.to_numpy(y), 1 / (1 + np.exp(-np.ones((3, 4)))))

        # Test relu
        x = ops.subtract(ops.arange(12).reshape((3, 4)), 5)
        y = ops.relu(x)
        assert ops.shape(y) == (3, 4)
        assert np.allclose(ops.to_numpy(y), np.maximum(0, np.arange(12).reshape((3, 4)) - 5))

        # Test tanh
        x = ops.ones((3, 4))
        y = ops.tanh(x)
        assert ops.shape(y) == (3, 4)
        assert np.allclose(ops.to_numpy(y), np.tanh(np.ones((3, 4))))

    def test_random_operations(self):
        """Test random operations."""
        # Test random_normal
        shape = (3, 4)
        x = ops.random_normal(shape)
        assert ops.shape(x) == shape

        # Test random_uniform
        x = ops.random_uniform(shape)
        assert ops.shape(x) == shape

        # Test seed reproducibility using backend directly
        from ember_ml.backend import get_backend_module
        backend_module = get_backend_module()
        
        # Set seed using the backend module
        if hasattr(backend_module, 'set_seed'):
            backend_module.set_seed(42)
            x1 = ops.random_normal(shape)
            backend_module.set_seed(42)
            x2 = ops.random_normal(shape)
            assert np.allclose(ops.to_numpy(x1), ops.to_numpy(x2))

    def test_comparison_operations(self):
        """Test comparison operations."""
        # Test equal
        x = ops.ones((3, 4))
        y = ops.ones((3, 4))
        z = ops.equal(x, y)
        assert ops.shape(z) == (3, 4)
        assert np.all(ops.to_numpy(z))

        # Test not_equal
        y = tensor.zeros((3, 4))
        z = ops.not_equal(x, y)
        assert ops.shape(z) == (3, 4)
        assert np.all(ops.to_numpy(z))

        # Test greater
        z = ops.greater(x, y)
        assert ops.shape(z) == (3, 4)
        assert np.all(ops.to_numpy(z))

        # Test less
        z = ops.less(y, x)
        assert ops.shape(z) == (3, 4)
        assert np.all(ops.to_numpy(z))

    def test_device_operations(self):
        """Test device operations."""
        # Test to_device
        x = ops.ones((3, 4))
        y = ops.to_device(x, 'cpu')
        assert ops.shape(y) == (3, 4)
        assert np.allclose(ops.to_numpy(y), np.ones((3, 4)))

        # Test get_device
        device = ops.get_device(x)
        assert device == 'cpu'

    def test_dtype_operations(self):
        """Test dtype operations."""
        # Test cast
        x = ops.ones((3, 4))
        y = ops.cast(x, np.float32)
        assert ops.shape(y) == (3, 4)
        assert ops.dtype(y) == np.float32
        assert np.allclose(ops.to_numpy(y), np.ones((3, 4), dtype=np.float32))

    def test_solver_operations(self):
        """Test solver operations."""
        # Test solve
        a = ops.eye(3)
        b = ops.ones((3, 1))
        x = ops.solve(a, b)
        assert ops.shape(x) == (3, 1)
        assert np.allclose(ops.to_numpy(x), np.ones((3, 1)))