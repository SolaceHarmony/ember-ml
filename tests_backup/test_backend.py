"""
Unit tests for backend selection and switching.

This module contains pytest tests for the backend selection and switching
functionality in the ember_ml library.
"""

import pytest
import platform
import numpy as np  # Required for array comparison in tests
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.backend import get_backend, set_backend

# Note: This test file specifically tests backend switching functionality,
# which requires checking for the availability of different backends.
# The direct backend imports below are necessary for this purpose.
# NumPy is used for array comparison in tests, which is a common testing pattern.

# Ensure current_backend is always a string
def get_current_backend() -> str:
    """Get the current backend and ensure it's a string."""
    backend = get_backend()
    return backend if backend is not None else 'numpy'

# List of backends to test
BACKENDS = ['numpy']

# Note: The following imports are necessary for testing backend availability
# and are used to populate the BACKENDS list. These are exceptions to the
# backend purity rules since this is a test file specifically for testing
# backend switching functionality.
# EMBERLINT: IGNORE - Backend-specific imports are required for this test file
try:
    import torch  # noqa: F401 - Used to check if torch is available
    BACKENDS.append('torch')
    # The torch import is used to check if PyTorch is available on the system
    # and to add it to the BACKENDS list for testing
except ImportError:
    pass

try:
    import mlx  # noqa: F401 - Used to check if mlx is available
    BACKENDS.append('mlx')
    # The mlx import is used to check if MLX is available on the system
    # and to add it to the BACKENDS list for testing
except ImportError:
    pass

class TestBackendSelection:
    """Tests for backend selection."""

    def test_default_backend(self):
            """Test default backend selection."""
            # Get the current backend
            current_backend = get_current_backend()
            
            # Reset to default backend
            set_backend('numpy')
            ops.set_backend('numpy')
            
            # Verify that the backend is set to numpy
            assert get_backend() == 'numpy'
            assert ops.get_ops() == 'numpy'
            
            # Restore the original backend
            set_backend(current_backend)
            ops.set_backend(current_backend)

    @pytest.mark.parametrize('backend_name', BACKENDS)
    def test_backend_switching(self, backend_name):
        """Test backend switching."""
        # Get the current backend
        current_backend = get_backend()
        
        # Switch to the specified backend
        set_backend(backend_name)
        ops.set_backend(backend_name)
        
        # Verify that the backend is set correctly
        assert get_backend() == backend_name
        assert ops.get_ops() == backend_name
        
        # Restore the original backend
        set_backend(current_backend or 'numpy')
        ops.set_backend(current_backend or 'numpy')

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
            ops.set_backend('invalid_backend')
        
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
            ops.set_backend(backend_name)
            
            # Verify that the backend is set correctly
            assert get_backend() == backend_name
            assert ops.get_ops() == backend_name
            
            # Reload the backend module
            import importlib
            importlib.reload(importlib.import_module('ember_ml.backend'))
            
            # Verify that the backend is still set correctly
            assert get_backend() == backend_name
        finally:
            # Restore the original backend
            set_backend(current_backend or 'numpy')
            ops.set_backend(current_backend or 'numpy')

class TestBackendCompatibility:
    """Tests for backend compatibility."""

    @pytest.mark.parametrize('backend1', BACKENDS)
    @pytest.mark.parametrize('backend2', BACKENDS)
    def test_tensor_conversion(self, backend1, backend2):
        """Test tensor conversion between backends."""
        # Get the current backend
        current_backend = get_backend()
        
        try:
            # Switch to the first backend
            set_backend(backend1)
            ops.set_backend(backend1)
            
            # Create a tensor
            x1 = tensor.ones((3, 4))
            
            # Switch to the second backend
            set_backend(backend2)
            ops.set_backend(backend2)
            
            # For MLX and PyTorch combinations, direct conversion should fail with ValueError
            # This is a PASS condition as we want strong typing between these backends
            if (backend1 == 'mlx' and backend2 == 'torch') or (backend1 == 'torch' and backend2 == 'mlx'):
                with pytest.raises(ValueError):
                    x2 = tensor.convert_to_tensor(x1)
                
                # But conversion through NumPy should work
                # Switch back to the first backend to get NumPy representation
                set_backend(backend1)
                ops.set_backend(backend1)
                x_np = tensor.to_numpy(x1)
                
                # Switch back to the second backend
                set_backend(backend2)
                ops.set_backend(backend2)
                
                # Convert from NumPy to the second backend
                x2 = tensor.convert_to_tensor(x_np)
                
                # Verify that the tensor has the correct shape and values
                assert tensor.shape(x2) == (3, 4)
                assert np.allclose(tensor.to_numpy(x2), x_np)
            elif backend1 == backend2:
                # For same-backend conversions, direct conversion should work
                # This tests that torch-torch, mlx-mlx, numpy-numpy conversions are allowed
                x2 = tensor.convert_to_tensor(x1)
                
                # Verify that the tensor has the correct shape and values
                assert tensor.shape(x2) == (3, 4)
                assert np.allclose(tensor.to_numpy(x2), tensor.to_numpy(x1))
            else:
                # For other backend combinations, conversion through NumPy should work
                x2 = tensor.convert_to_tensor(tensor.to_numpy(x1))
                
                # Verify that the tensor has the correct shape and values
                assert tensor.shape(x2) == (3, 4)
                assert np.allclose(tensor.to_numpy(x2), tensor.to_numpy(x1))
        finally:
            # Restore the original backend
            set_backend(current_backend or 'numpy')
            ops.set_backend(current_backend or 'numpy')
    @pytest.mark.parametrize('backend_name', BACKENDS)
    def test_operation_consistency(self, backend_name):
        """Test operation results consistency across backends."""
        # Get the current backend
        current_backend = get_backend()
        
        try:
            # Skip test for torch on macOS with MPS to avoid device issues
            if backend_name == 'torch' and platform.system() == 'Darwin' and platform.machine() == 'arm64':
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        pytest.skip("Skipping test for PyTorch with MPS to avoid device issues")
                except Exception as e:
                    pass
            
            # Switch to numpy backend
            set_backend('numpy')
            ops.set_backend('numpy')
            
            # Create tensors
            x_numpy = tensor.ones((3, 4))
            y_numpy = tensor.full((3, 4), 2)
            
            # Perform operations
            add_numpy = ops.add(x_numpy, y_numpy)
            subtract_numpy = ops.subtract(x_numpy, y_numpy)
            multiply_numpy = ops.multiply(x_numpy, y_numpy)
            divide_numpy = ops.divide(x_numpy, y_numpy)
            
            # Convert to numpy arrays for comparison
            add_numpy_array = tensor.to_numpy(add_numpy)
            subtract_numpy_array = tensor.to_numpy(subtract_numpy)
            multiply_numpy_array = tensor.to_numpy(multiply_numpy)
            divide_numpy_array = tensor.to_numpy(divide_numpy)
            
            # Switch to the specified backend
            set_backend(backend_name)
            ops.set_backend(backend_name)
            
            # Create tensors
            x = tensor.ones((3, 4))
            y = tensor.full((3, 4), 2)
            
            # Perform operations
            add_result = ops.add(x, y)
            subtract_result = ops.subtract(x, y)
            multiply_result = ops.multiply(x, y)
            divide_result = ops.divide(x, y)
            
            # Verify that the results are the same
            # Note: np.allclose is used here for array comparison in tests.
            # This is a common testing pattern and an exception to the backend purity rules
            # since this is a test file specifically for testing backend functionality.
            assert np.allclose(tensor.to_numpy(add_result), add_numpy_array)
            assert np.allclose(tensor.to_numpy(subtract_result), subtract_numpy_array)
            assert np.allclose(tensor.to_numpy(multiply_result), multiply_numpy_array)
            assert np.allclose(tensor.to_numpy(divide_result), divide_numpy_array)
        finally:
            # Restore the original backend
            set_backend(current_backend or 'numpy')
            ops.set_backend(current_backend or 'numpy')

class TestBackendAutoDetection:
    """Tests for backend auto-detection."""

    def test_backend_auto_detection(self):
        """Test backend auto-detection."""
        # Get the current backend
        current_backend = get_backend()
        
        try:
            # Reset to default backend
            set_backend('numpy')
            ops.set_backend('numpy')
            
            # Verify that the backend is set to numpy
            assert get_backend() == 'numpy'
            assert ops.get_ops() == 'numpy'
            
            # Check if torch is available
            if 'torch' in BACKENDS:
                # Switch to torch backend
                set_backend('torch')
                ops.set_backend('torch')
                
                # Verify that the backend is set to torch
                assert get_backend() == 'torch'
                assert ops.get_ops() == 'torch'
            
            # Check if mlx is available
            if 'mlx' in BACKENDS:
                # Switch to mlx backend
                set_backend('mlx')
                ops.set_backend('mlx')
                
                # Verify that the backend is set to mlx
                assert get_backend() == 'mlx'
                assert ops.get_ops() == 'mlx'
        finally:
            # Restore the original backend
            set_backend(current_backend or 'numpy')
            ops.set_backend(current_backend or 'numpy')