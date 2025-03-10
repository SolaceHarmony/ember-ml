"""
Unit tests for data type operations across different backends.

This module contains pytest tests for the data type operations in the ops module.
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

class TestDTypeOperations:
    """Tests for data type operations."""

    def test_get_dtype(self, backend):
        """Test get_dtype operation."""
        # Test with default dtype names
        dtype_names = ['float32', 'float64', 'int32', 'int64', 'bool']
        
        for dtype_name in dtype_names:
            try:
                # Get the dtype
                dtype = ops.get_dtype(dtype_name)
                
                # Create a tensor with the dtype
                x = ops.ones((3, 4), dtype=dtype)
                
                # Verify that the tensor has the correct dtype
                if backend == 'numpy':
                    assert ops.dtype(x) == getattr(np, dtype_name)
                elif backend == 'torch':
                    import torch
                    assert ops.dtype(x) == getattr(torch, dtype_name)
                elif backend == 'mlx':
                    import mlx.core
                    assert ops.dtype(x) == getattr(mlx.core, dtype_name)
            except (ValueError, AttributeError):
                # Skip if the dtype is not supported by the backend
                pass

    def test_to_numpy_dtype(self, backend):
        """Test to_numpy_dtype operation."""
        # Create a tensor with default dtype
        x = ops.ones((3, 4))
        
        # Get the dtype
        dtype = ops.dtype(x)
        
        # Convert to NumPy dtype
        numpy_dtype = ops.to_numpy_dtype(dtype)
        
        # Verify that the NumPy dtype is correct
        if backend == 'numpy':
            assert numpy_dtype == dtype
        elif backend == 'torch':
            import torch
            if dtype == torch.float32:
                assert numpy_dtype == np.float32 or numpy_dtype == 'float32'
            elif dtype == torch.float64:
                assert numpy_dtype == np.float64 or numpy_dtype == 'float64'
            elif dtype == torch.int32:
                assert numpy_dtype == np.int32 or numpy_dtype == 'int32'
            elif dtype == torch.int64:
                assert numpy_dtype == np.int64 or numpy_dtype == 'int64'
            elif dtype == torch.bool:
                assert numpy_dtype == np.bool_ or numpy_dtype == 'bool'
        elif backend == 'mlx':
            import mlx.core
            if dtype == mlx.core.float32:
                assert numpy_dtype == np.float32 or numpy_dtype == 'float32'
            elif dtype == mlx.core.int32:
                assert numpy_dtype == np.int32 or numpy_dtype == 'int32'
            elif dtype == mlx.core.bool_:
                assert numpy_dtype == np.bool_ or numpy_dtype == 'bool'

    def test_from_numpy_dtype(self, backend):
        """Test from_numpy_dtype operation."""
        # Test with NumPy dtypes
        numpy_dtypes = [np.float32, np.float64, np.int32, np.int64, np.bool_]
        
        for numpy_dtype in numpy_dtypes:
            try:
                # Convert from NumPy dtype
                dtype = ops.from_numpy_dtype(numpy_dtype)
                
                # Create a tensor with the dtype
                x = ops.ones((3, 4), dtype=dtype)
                
                # Verify that the tensor has the correct dtype
                if backend == 'numpy':
                    assert ops.dtype(x) == numpy_dtype
                elif backend == 'torch':
                    import torch
                    if numpy_dtype == np.float32:
                        assert ops.dtype(x) == torch.float32
                    elif numpy_dtype == np.float64:
                        assert ops.dtype(x) == torch.float64
                    elif numpy_dtype == np.int32:
                        assert ops.dtype(x) == torch.int32
                    elif numpy_dtype == np.int64:
                        assert ops.dtype(x) == torch.int64
                    elif numpy_dtype == np.bool_:
                        assert ops.dtype(x) == torch.bool
                elif backend == 'mlx':
                    import mlx.core
                    if numpy_dtype == np.float32:
                        assert ops.dtype(x) == mlx.core.float32
                    elif numpy_dtype == np.int32:
                        assert ops.dtype(x) == mlx.core.int32
                    elif numpy_dtype == np.bool_:
                        assert ops.dtype(x) == mlx.core.bool_
            except (ValueError, AttributeError):
                # Skip if the dtype is not supported by the backend
                pass

class TestDTypeCompatibility:
    """Tests for data type compatibility."""

    def test_mixed_dtypes(self, backend):
        """Test operations with mixed dtypes."""
        # Create tensors with different dtypes
        if backend == 'numpy':
            x = ops.ones((3, 4), dtype=np.float32)
            y = ops.ones((3, 4), dtype=np.float64)
        elif backend == 'torch':
            import torch
            x = ops.ones((3, 4), dtype=torch.float32)
            y = ops.ones((3, 4), dtype=torch.float64)
        elif backend == 'mlx':
            import mlx.core
            x = ops.ones((3, 4), dtype=mlx.core.float32)
            # MLX doesn't support float64, so use int32 instead
            y = ops.ones((3, 4), dtype=mlx.core.int32)
        
        # Test addition with mixed dtypes
        z = ops.add(x, y)
        
        # The result should have the higher precision dtype
        if backend == 'numpy':
            assert ops.dtype(z) == np.float64
        elif backend == 'torch':
            assert ops.dtype(z) == torch.float64
        elif backend == 'mlx':
            # MLX follows different promotion rules
            assert ops.dtype(z) in [mlx.core.float32, mlx.core.int32]

    def test_dtype_promotion(self, backend):
        """Test dtype promotion rules."""
        # Create tensors with different dtypes
        if backend == 'numpy':
            dtypes = [np.float32, np.float64, np.int32, np.int64]
        elif backend == 'torch':
            import torch
            dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
        elif backend == 'mlx':
            import mlx.core
            dtypes = [mlx.core.float32, mlx.core.int32]
        
        # Test all combinations of dtypes
        for dtype1 in dtypes:
            for dtype2 in dtypes:
                # Create tensors
                x = ops.ones((3, 4), dtype=dtype1)
                y = ops.ones((3, 4), dtype=dtype2)
                
                # Test addition
                z = ops.add(x, y)
                
                # The result should have the correct dtype based on promotion rules
                if backend == 'numpy':
                    # NumPy has well-defined promotion rules
                    expected_dtype = np.result_type(dtype1, dtype2)
                    assert ops.dtype(z) == expected_dtype
                elif backend == 'torch':
                    # PyTorch has similar promotion rules to NumPy
                    if dtype1 == torch.float64 or dtype2 == torch.float64:
                        assert ops.dtype(z) == torch.float64
                    elif dtype1 == torch.float32 or dtype2 == torch.float32:
                        assert ops.dtype(z) == torch.float32
                    elif dtype1 == torch.int64 or dtype2 == torch.int64:
                        assert ops.dtype(z) == torch.int64
                    else:
                        assert ops.dtype(z) == torch.int32
                elif backend == 'mlx':
                    # MLX has its own promotion rules
                    if dtype1 == mlx.core.float32 or dtype2 == mlx.core.float32:
                        assert ops.dtype(z) == mlx.core.float32
                    else:
                        assert ops.dtype(z) == mlx.core.int32

    def test_dtype_precision(self, backend):
        """Test dtype precision preservation."""
        # Create a tensor with high precision
        if backend == 'numpy':
            x = ops.ones((3, 4), dtype=np.float64)
        elif backend == 'torch':
            import torch
            x = ops.ones((3, 4), dtype=torch.float64)
        elif backend == 'mlx':
            import mlx.core
            # MLX doesn't support float64, so use float32 instead
            x = ops.ones((3, 4), dtype=mlx.core.float32)
        
        # Perform operations that should preserve precision
        y = ops.multiply(x, 2.0)
        z = ops.add(y, 3.0)
        
        # The result should have the same precision as the input
        assert ops.dtype(z) == ops.dtype(x)
        
        # Cast to lower precision
        if backend == 'numpy':
            w = ops.cast(z, np.float32)
            assert ops.dtype(w) == np.float32
        elif backend == 'torch':
            w = ops.cast(z, torch.float32)
            assert ops.dtype(w) == torch.float32
        elif backend == 'mlx':
            # MLX doesn't support float64, so this test is not applicable
            pass
            