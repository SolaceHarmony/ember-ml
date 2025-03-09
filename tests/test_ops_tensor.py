"""
Unit tests for tensor operations across different backends.

This module contains pytest tests for the tensor operations in the ops module.
It tests each operation with different backends to ensure consistency.
"""

import pytest
import numpy as np
from ember_ml import ops
from ember_ml.backend import get_backend, set_backend

# List of backends to test
BACKENDS = ['numpy']
try:
    import torch
    BACKENDS.append('torch')
except ImportError:
    pass

try:
    import mlx.core
    BACKENDS.append('mlx')
except ImportError:
    pass

@pytest.fixture(params=BACKENDS)
def backend(request):
    """Fixture to test with different backends."""
    prev_backend = get_backend()
    set_backend(request.param)
    ops.set_ops(request.param)
    yield request.param
    set_backend(prev_backend)
    ops.set_ops(prev_backend)

class TestTensorCreation:
    """Tests for tensor creation operations."""

    def test_zeros(self, backend):
        """Test zeros operation."""
        # Test with 1D shape
        shape = (5,)
        x = ops.zeros(shape)
        assert ops.shape(x) == shape
        assert np.allclose(ops.to_numpy(x), np.zeros(shape))

        # Test with 2D shape
        shape = (3, 4)
        x = ops.zeros(shape)
        assert ops.shape(x) == shape
        assert np.allclose(ops.to_numpy(x), np.zeros(shape))

        # Test with dtype
        if backend == 'numpy':
            dtype = np.float32
        elif backend == 'torch':
            import torch
            dtype = torch.float32
        elif backend == 'mlx':
            import mlx.core
            dtype = mlx.core.float32
        else:
            pytest.skip(f"Unknown backend: {backend}")

        x = ops.zeros(shape, dtype=dtype)
        assert ops.dtype(x) == dtype
        assert np.allclose(ops.to_numpy(x), np.zeros(shape, dtype=np.float32))

    def test_ones(self, backend):
        """Test ones operation."""
        # Test with 1D shape
        shape = (5,)
        x = ops.ones(shape)
        assert ops.shape(x) == shape
        assert np.allclose(ops.to_numpy(x), np.ones(shape))

        # Test with 2D shape
        shape = (3, 4)
        x = ops.ones(shape)
        assert ops.shape(x) == shape
        assert np.allclose(ops.to_numpy(x), np.ones(shape))

        # Test with dtype
        if backend == 'numpy':
            dtype = np.float32
        elif backend == 'torch':
            import torch
            dtype = torch.float32
        elif backend == 'mlx':
            import mlx.core
            dtype = mlx.core.float32
        else:
            pytest.skip(f"Unknown backend: {backend}")

        x = ops.ones(shape, dtype=dtype)
        assert ops.dtype(x) == dtype
        assert np.allclose(ops.to_numpy(x), np.ones(shape, dtype=np.float32))

    def test_zeros_like(self, backend):
        """Test zeros_like operation."""
        # Create a tensor to use as reference
        shape = (3, 4)
        x_ref = ops.ones(shape)
        
        # Test zeros_like
        x = ops.zeros_like(x_ref)
        assert ops.shape(x) == shape
        assert np.allclose(ops.to_numpy(x), np.zeros(shape))
        
        # Test with different dtype
        if backend == 'numpy':
            dtype = np.float32
        elif backend == 'torch':
            import torch
            dtype = torch.float32
        elif backend == 'mlx':
            import mlx.core
            dtype = mlx.core.float32
        else:
            pytest.skip(f"Unknown backend: {backend}")
            
        x = ops.zeros_like(x_ref, dtype=dtype)
        assert ops.dtype(x) == dtype
        assert np.allclose(ops.to_numpy(x), np.zeros(shape, dtype=np.float32))

    def test_ones_like(self, backend):
        """Test ones_like operation."""
        # Create a tensor to use as reference
        shape = (3, 4)
        x_ref = ops.zeros(shape)
        
        # Test ones_like
        x = ops.ones_like(x_ref)
        assert ops.shape(x) == shape
        assert np.allclose(ops.to_numpy(x), np.ones(shape))
        
        # Test with different dtype
        if backend == 'numpy':
            dtype = np.float32
        elif backend == 'torch':
            import torch
            dtype = torch.float32
        elif backend == 'mlx':
            import mlx.core
            dtype = mlx.core.float32
        else:
            pytest.skip(f"Unknown backend: {backend}")
            
        x = ops.ones_like(x_ref, dtype=dtype)
        assert ops.dtype(x) == dtype
        assert np.allclose(ops.to_numpy(x), np.ones(shape, dtype=np.float32))

    def test_eye(self, backend):
        """Test eye operation."""
        # Test square matrix
        n = 3
        x = ops.eye(n)
        assert ops.shape(x) == (n, n)
        assert np.allclose(ops.to_numpy(x), np.eye(n))
        
        # Test rectangular matrix
        n, m = 3, 4
        x = ops.eye(n, m)
        assert ops.shape(x) == (n, m)
        assert np.allclose(ops.to_numpy(x), np.eye(n, m))
        
        # Test with dtype
        if backend == 'numpy':
            dtype = np.float32
        elif backend == 'torch':
            import torch
            dtype = torch.float32
        elif backend == 'mlx':
            import mlx.core
            dtype = mlx.core.float32
        else:
            pytest.skip(f"Unknown backend: {backend}")
            
        x = ops.eye(n, dtype=dtype)
        assert ops.dtype(x) == dtype
        assert np.allclose(ops.to_numpy(x), np.eye(n, dtype=np.float32))

class TestTensorManipulation:
    """Tests for tensor manipulation operations."""

    def test_reshape(self, backend):
        """Test reshape operation."""
        # Create a tensor
        original_shape = (2, 6)
        x = ops.ones(original_shape)
        
        # Test reshape
        new_shape = (3, 4)
        y = ops.reshape(x, new_shape)
        assert ops.shape(y) == new_shape
        assert np.allclose(ops.to_numpy(y), np.ones(new_shape))
        
        # Test reshape with -1 dimension
        new_shape = (3, -1)
        y = ops.reshape(x, new_shape)
        assert ops.shape(y) == (3, 4)
        assert np.allclose(ops.to_numpy(y), np.ones((3, 4)))

    def test_transpose(self, backend):
        """Test transpose operation."""
        # Create a tensor
        shape = (2, 3, 4)
        x = ops.ones(shape)
        
        # Test default transpose
        y = ops.transpose(x)
        
        # The default transpose behavior differs between backends
        # For NumPy, it's a complete transpose, for PyTorch and MLX it swaps the last two dimensions
        if backend == 'numpy':
            expected_shape = (4, 3, 2)
            expected_array = np.transpose(np.ones(shape))
        else:
            expected_shape = (2, 4, 3)
            expected_array = np.transpose(np.ones(shape), (0, 2, 1))
            
        assert ops.shape(y) == expected_shape
        assert np.allclose(ops.to_numpy(y), expected_array)
        
        # Test transpose with specified axes
        axes = (2, 0, 1)
        y = ops.transpose(x, axes)
        assert ops.shape(y) == (4, 2, 3)
        assert np.allclose(ops.to_numpy(y), np.transpose(np.ones(shape), axes))

    def test_concatenate(self, backend):
        """Test concatenate operation."""
        # Create tensors
        shape1 = (2, 3)
        shape2 = (2, 3)
        x1 = ops.ones(shape1)
        x2 = ops.zeros(shape2)
        
        # Test concatenate along axis 0
        y = ops.concatenate([x1, x2], axis=0)
        assert ops.shape(y) == (4, 3)
        assert np.allclose(ops.to_numpy(y), np.concatenate([np.ones(shape1), np.zeros(shape2)], axis=0))
        
        # Test concatenate along axis 1
        y = ops.concatenate([x1, x2], axis=1)
        assert ops.shape(y) == (2, 6)
        assert np.allclose(ops.to_numpy(y), np.concatenate([np.ones(shape1), np.zeros(shape2)], axis=1))

    def test_stack(self, backend):
        """Test stack operation."""
        # Create tensors
        shape = (2, 3)
        x1 = ops.ones(shape)
        x2 = ops.zeros(shape)
        
        # Test stack along axis 0
        y = ops.stack([x1, x2], axis=0)
        assert ops.shape(y) == (2, 2, 3)
        assert np.allclose(ops.to_numpy(y), np.stack([np.ones(shape), np.zeros(shape)], axis=0))
        
        # Test stack along axis 1
        y = ops.stack([x1, x2], axis=1)
        assert ops.shape(y) == (2, 2, 3)
        assert np.allclose(ops.to_numpy(y), np.stack([np.ones(shape), np.zeros(shape)], axis=1))
        
        # Test stack along axis 2
        y = ops.stack([x1, x2], axis=2)
        assert ops.shape(y) == (2, 3, 2)
        assert np.allclose(ops.to_numpy(y), np.stack([np.ones(shape), np.zeros(shape)], axis=2))

class TestTensorInfo:
    """Tests for tensor information operations."""

    def test_shape(self, backend):
        """Test shape operation."""
        # Test with 1D shape
        shape = (5,)
        x = ops.ones(shape)
        assert ops.shape(x) == shape
        
        # Test with 2D shape
        shape = (3, 4)
        x = ops.ones(shape)
        assert ops.shape(x) == shape
        
        # Test with 3D shape
        shape = (2, 3, 4)
        x = ops.ones(shape)
        assert ops.shape(x) == shape

    def test_dtype(self, backend):
        """Test dtype operation."""
        # Test with default dtype
        x = ops.ones((3, 4))
        
        # The default dtype depends on the backend
        if backend == 'numpy':
            assert ops.dtype(x) == np.float64
        elif backend == 'torch':
            import torch
            assert ops.dtype(x) == torch.float32
        elif backend == 'mlx':
            import mlx.core
            assert ops.dtype(x) == mlx.core.float32
        
        # Test with specified dtype
        if backend == 'numpy':
            dtype = np.float32
        elif backend == 'torch':
            import torch
            dtype = torch.float32
        elif backend == 'mlx':
            import mlx.core
            dtype = mlx.core.float32
        else:
            pytest.skip(f"Unknown backend: {backend}")
            
        x = ops.ones((3, 4), dtype=dtype)
        assert ops.dtype(x) == dtype

    def test_cast(self, backend):
        """Test cast operation."""
        # Create a tensor
        x = ops.ones((3, 4))
        
        # Test cast to different dtype
        if backend == 'numpy':
            dtype = np.int32
        elif backend == 'torch':
            import torch
            dtype = torch.int32
        elif backend == 'mlx':
            import mlx.core
            dtype = mlx.core.int32
        else:
            pytest.skip(f"Unknown backend: {backend}")
            
        y = ops.cast(x, dtype)
        assert ops.dtype(y) == dtype
        assert np.allclose(ops.to_numpy(y), np.ones((3, 4), dtype=np.int32))

    def test_copy(self, backend):
        """Test copy operation."""
        # Create a tensor
        x = ops.ones((3, 4))
        
        # Test copy
        y = ops.copy(x)
        assert ops.shape(y) == ops.shape(x)
        assert ops.dtype(y) == ops.dtype(x)
        assert np.allclose(ops.to_numpy(y), ops.to_numpy(x))
        
        # Verify that y is a copy, not a reference
        if backend == 'numpy':
            # For NumPy, we can modify the array directly
            x_np = ops.to_numpy(x)
            x_np[0, 0] = 0
            assert ops.to_numpy(x)[0, 0] == 0
            assert ops.to_numpy(y)[0, 0] == 1
        elif backend == 'torch':
            # For PyTorch, we need to modify the tensor
            import torch
            x_torch = x
            x_torch[0, 0] = 0
            assert ops.to_numpy(x)[0, 0] == 0
            assert ops.to_numpy(y)[0, 0] == 1
        elif backend == 'mlx':
            # MLX arrays are immutable, so we can't modify them in-place
            # Instead, we'll just verify that the copy works
            assert np.allclose(ops.to_numpy(y), ops.to_numpy(x))