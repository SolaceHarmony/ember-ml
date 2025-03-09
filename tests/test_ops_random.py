"""
Unit tests for random operations across different backends.

This module contains pytest tests for the random operations in the ops module.
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

class TestRandomGeneration:
    """Tests for random generation operations."""

    def test_random_normal(self, backend):
        """Test random_normal operation."""
        # Test with 1D shape
        shape = (1000,)
        mean = 0.0
        stddev = 1.0
        x = ops.random_normal(shape, mean, stddev)
        assert ops.shape(x) == shape
        
        # Check statistical properties
        x_np = ops.to_numpy(x)
        assert abs(np.mean(x_np) - mean) < 0.1
        assert abs(np.std(x_np) - stddev) < 0.1
        
        # Test with 2D shape
        shape = (100, 100)
        mean = 2.0
        stddev = 0.5
        x = ops.random_normal(shape, mean, stddev)
        assert ops.shape(x) == shape
        
        # Check statistical properties
        x_np = ops.to_numpy(x)
        assert abs(np.mean(x_np) - mean) < 0.1
        assert abs(np.std(x_np) - stddev) < 0.1
        
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
            
        x = ops.random_normal(shape, mean, stddev, dtype=dtype)
        assert ops.dtype(x) == dtype

    def test_random_uniform(self, backend):
        """Test random_uniform operation."""
        # Test with 1D shape
        shape = (1000,)
        minval = 0.0
        maxval = 1.0
        x = ops.random_uniform(shape, minval, maxval)
        assert ops.shape(x) == shape
        
        # Check statistical properties
        x_np = ops.to_numpy(x)
        assert np.min(x_np) >= minval
        assert np.max(x_np) <= maxval
        assert abs(np.mean(x_np) - (minval + maxval) / 2) < 0.1
        
        # Test with 2D shape
        shape = (100, 100)
        minval = -1.0
        maxval = 1.0
        x = ops.random_uniform(shape, minval, maxval)
        assert ops.shape(x) == shape
        
        # Check statistical properties
        x_np = ops.to_numpy(x)
        assert np.min(x_np) >= minval
        assert np.max(x_np) <= maxval
        assert abs(np.mean(x_np) - (minval + maxval) / 2) < 0.1
        
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
            
        x = ops.random_uniform(shape, minval, maxval, dtype=dtype)
        assert ops.dtype(x) == dtype

    def test_random_binomial(self, backend):
        """Test random_binomial operation."""
        try:
            # Test with 1D shape
            shape = (1000,)
            p = 0.5
            x = ops.random_binomial(shape, p)
            assert ops.shape(x) == shape
            
            # Check statistical properties
            x_np = ops.to_numpy(x)
            assert np.all(np.logical_or(x_np == 0, x_np == 1))
            assert abs(np.mean(x_np) - p) < 0.1
            
            # Test with 2D shape
            shape = (100, 100)
            p = 0.7
            x = ops.random_binomial(shape, p)
            assert ops.shape(x) == shape
            
            # Check statistical properties
            x_np = ops.to_numpy(x)
            assert np.all(np.logical_or(x_np == 0, x_np == 1))
            assert abs(np.mean(x_np) - p) < 0.1
            
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
                
            x = ops.random_binomial(shape, p, dtype=dtype)
            assert ops.dtype(x) == dtype
        except (NotImplementedError, AttributeError):
            pytest.skip(f"{backend} backend doesn't implement random_binomial")

class TestRandomSeed:
    """Tests for random seed management."""

    def test_set_random_seed(self, backend):
        """Test set_random_seed operation."""
        try:
            # Set random seed
            ops.set_random_seed(42)
            
            # Generate random tensor
            shape = (100, 100)
            x1 = ops.random_normal(shape)
            
            # Set the same random seed again
            ops.set_random_seed(42)
            
            # Generate another random tensor
            x2 = ops.random_normal(shape)
            
            # The two tensors should be identical
            assert np.allclose(ops.to_numpy(x1), ops.to_numpy(x2))
            
            # Set a different random seed
            ops.set_random_seed(43)
            
            # Generate another random tensor
            x3 = ops.random_normal(shape)
            
            # The third tensor should be different from the first two
            assert not np.allclose(ops.to_numpy(x1), ops.to_numpy(x3))
        except (NotImplementedError, AttributeError):
            pytest.skip(f"{backend} backend doesn't implement set_random_seed")

    def test_get_random_seed(self, backend):
        """Test get_random_seed operation."""
        try:
            # Set random seed
            ops.set_random_seed(42)
            
            # Get random seed
            seed = ops.random_ops().get_seed()
            
            # The seed should be 42
            assert seed == 42
            
            # Set a different random seed
            ops.set_random_seed(43)
            
            # Get random seed
            seed = ops.random_ops().get_seed()
            
            # The seed should be 43
            assert seed == 43
        except (NotImplementedError, AttributeError):
            pytest.skip(f"{backend} backend doesn't implement get_seed")

class TestRandomUtilities:
    """Tests for random utility operations."""

    def test_shuffle(self, backend):
        """Test shuffle operation."""
        try:
            # Create a tensor
            x = ops.arange(100)
            
            # Set random seed for reproducibility
            ops.set_random_seed(42)
            
            # Shuffle the tensor
            y = ops.random_ops().shuffle(x)
            
            # The shuffled tensor should have the same shape
            assert ops.shape(y) == ops.shape(x)
            
            # The shuffled tensor should have the same elements
            assert np.array_equal(np.sort(ops.to_numpy(y)), np.sort(ops.to_numpy(x)))
            
            # The shuffled tensor should be different from the original
            assert not np.array_equal(ops.to_numpy(y), ops.to_numpy(x))
            
            # Set the same random seed again
            ops.set_random_seed(42)
            
            # Shuffle the tensor again
            z = ops.random_ops().shuffle(x)
            
            # The two shuffled tensors should be identical
            assert np.array_equal(ops.to_numpy(y), ops.to_numpy(z))
        except (NotImplementedError, AttributeError):
            pytest.skip(f"{backend} backend doesn't implement shuffle")
            