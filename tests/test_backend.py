"""
Unit tests for backend selection and switching.

This module contains pytest tests for the backend selection and switching
functionality in the emberharmony library.
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

class TestBackendSelection:
    """Tests for backend selection."""

    def test_default_backend(self):
        """Test default backend selection."""
        # Get the current backend
        current_backend = get_backend()
        
        # Reset to default backend
        set_backend('numpy')
        ops.set_ops('numpy')
        
        # Verify that the backend is set to numpy
        assert get_backend() == 'numpy'
        assert ops.get_ops() == 'numpy'
        
        # Restore the original backend
        set_backend(current_backend)
        ops.set_ops(current_backend)

    @pytest.mark.parametrize('backend_name', BACKENDS)
    def test_backend_switching(self, backend_name):
        """Test backend switching."""
        # Get the current backend
        current_backend = get_backend()
        
        # Switch to the specified backend
        set_backend(backend_name)
        ops.set_ops(backend_name)
        
        # Verify that the backend is set correctly
        assert get_backend() == backend_name
        assert ops.get_ops() == backend_name
        
        # Restore the original backend
        set_backend(current_backend)
        ops.set_ops(current_backend)

    def test_invalid_backend(self):
        """Test switching to an invalid backend."""
        # Get the current backend
        current_backend = get_backend()
        
        # Try to switch to an invalid backend
        with pytest.raises(ValueError):
            set_backend('invalid_backend')
        
        # Verify that the backend is still set to the original backend
        assert get_backend() == current_backend
        
        # Try to switch ops to an invalid backend
        with pytest.raises(ValueError):
            ops.set_ops('invalid_backend')
        
        # Verify that the ops backend is still set to the original backend
        assert ops.get_ops() == current_backend

class TestBackendPersistence:
    """Tests for backend persistence."""

    @pytest.mark.parametrize('backend_name', BACKENDS)
    def test_backend_persistence(self, backend_name):
        """Test that backend setting persists across module reloads."""
        # Get the current backend
        current_backend = get_backend()
        
        try:
            # Switch to the specified backend
            set_backend(backend_name)
            ops.set_ops(backend_name)
            
            # Verify that the backend is set correctly
            assert get_backend() == backend_name
            assert ops.get_ops() == backend_name
            
            # Reload the backend module
            import importlib
            importlib.reload(importlib.import_module('emberharmony.backend'))
            
            # Verify that the backend is still set correctly
            assert get_backend() == backend_name
        finally:
            # Restore the original backend
            set_backend(current_backend)
            ops.set_ops(current_backend)

class TestBackendCompatibility:
    """Tests for backend compatibility."""

    @pytest.mark.parametrize('backend1', BACKENDS)
    @pytest.mark.parametrize('backend2', BACKENDS)
    def test_tensor_conversion(self, backend1, backend2):
        """Test tensor conversion between backends."""
        # Skip if the backends are the same
        if backend1 == backend2:
            pytest.skip(f"Skipping conversion from {backend1} to {backend2}")
        
        # Get the current backend
        current_backend = get_backend()
        
        try:
            # Switch to the first backend
            set_backend(backend1)
            ops.set_ops(backend1)
            
            # Create a tensor
            x1 = ops.ones((3, 4))
            
            # Switch to the second backend
            set_backend(backend2)
            ops.set_ops(backend2)
            
            # Convert the tensor to the second backend
            x2 = ops.convert_to_tensor(ops.to_numpy(x1))
            
            # Verify that the tensor has the same values
            assert np.allclose(ops.to_numpy(x2), ops.to_numpy(x1))
        finally:
            # Restore the original backend
            set_backend(current_backend)
            ops.set_ops(current_backend)

    @pytest.mark.parametrize('backend_name', BACKENDS)
    def test_operation_consistency(self, backend_name):
        """Test operation results consistency across backends."""
        # Get the current backend
        current_backend = get_backend()
        
        try:
            # Switch to numpy backend
            set_backend('numpy')
            ops.set_ops('numpy')
            
            # Create tensors
            x_numpy = ops.ones((3, 4))
            y_numpy = ops.full((3, 4), 2)
            
            # Perform operations
            add_numpy = ops.add(x_numpy, y_numpy)
            subtract_numpy = ops.subtract(x_numpy, y_numpy)
            multiply_numpy = ops.multiply(x_numpy, y_numpy)
            divide_numpy = ops.divide(x_numpy, y_numpy)
            
            # Switch to the specified backend
            set_backend(backend_name)
            ops.set_ops(backend_name)
            
            # Create tensors
            x = ops.ones((3, 4))
            y = ops.full((3, 4), 2)
            
            # Perform operations
            add_result = ops.add(x, y)
            subtract_result = ops.subtract(x, y)
            multiply_result = ops.multiply(x, y)
            divide_result = ops.divide(x, y)
            
            # Verify that the results are the same
            assert np.allclose(ops.to_numpy(add_result), ops.to_numpy(add_numpy))
            assert np.allclose(ops.to_numpy(subtract_result), ops.to_numpy(subtract_numpy))
            assert np.allclose(ops.to_numpy(multiply_result), ops.to_numpy(multiply_numpy))
            assert np.allclose(ops.to_numpy(divide_result), ops.to_numpy(divide_numpy))
        finally:
            # Restore the original backend
            set_backend(current_backend)
            ops.set_ops(current_backend)

class TestBackendAutoDetection:
    """Tests for backend auto-detection."""

    def test_backend_auto_detection(self):
        """Test backend auto-detection."""
        # Get the current backend
        current_backend = get_backend()
        
        try:
            # Reset to default backend
            set_backend('numpy')
            ops.set_ops('numpy')
            
            # Verify that the backend is set to numpy
            assert get_backend() == 'numpy'
            assert ops.get_ops() == 'numpy'
            
            # Check if torch is available
            if 'torch' in BACKENDS:
                # Switch to torch backend
                set_backend('torch')
                ops.set_ops('torch')
                
                # Verify that the backend is set to torch
                assert get_backend() == 'torch'
                assert ops.get_ops() == 'torch'
            
            # Check if mlx is available
            if 'mlx' in BACKENDS:
                # Switch to mlx backend
                set_backend('mlx')
                ops.set_ops('mlx')
                
                # Verify that the backend is set to mlx
                assert get_backend() == 'mlx'
                assert ops.get_ops() == 'mlx'
        finally:
            # Restore the original backend
            set_backend(current_backend)
            ops.set_ops(current_backend)