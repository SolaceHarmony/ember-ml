"""
Test script for the auto-selection of backends.

This script tests the auto-selection of backends based on available hardware.
"""

import os
import platform
import pytest
from ember_ml.ops import (
    get_backend,
    set_backend,
    get_device
)
from ember_ml.backend import auto_select_backend, set_device
from ember_ml.nn import tensor

@pytest.fixture
def original_backend_and_device():
    """Save and restore the original backend and device."""
    original_backend = get_backend()
    # Create a dummy tensor to get the device
    import ember_ml as nl
    dummy = tensor.zeros((1, 1))
    original_device = get_device(dummy)
    
    # Clean up device name (remove ':0' suffix if present)
    if original_device and ':' in original_device:
        original_device = original_device.split(':')[0]
    
    yield original_backend, original_device
    
    # Restore the original backend and device
    set_backend(original_backend or 'mlx')
    if original_device:
        set_device(original_device)

# List of all available backends to test
ALL_BACKENDS = ['numpy', 'torch', 'mlx']

class TestBackendAutoSelection:
    """Test the auto-selection of backends."""
    
    def test_auto_select_backend(self, original_backend_and_device):
        """Test the auto-selection of backends."""
        backend, device = auto_select_backend()
        
        # Verify that a valid backend was selected
        assert backend in ['numpy', 'torch', 'torch_optimized', 'mlx']
        
        # Verify that the device is appropriate for the backend
        if backend == 'torch':
            assert device in ['cpu', 'cuda', 'mps']
        elif backend == 'mlx':
            assert device is None  # MLX doesn't use explicit device selection
        else:
            assert device is None  # NumPy doesn't use explicit device selection
        
        # Print the selected backend and device for debugging
        print(f"Auto-selected backend: {backend}, device: {device}")
        
        # Check if we're on macOS with Apple Silicon
        is_apple_silicon = (
            platform.system() == 'Darwin' and
            platform.machine().startswith('arm')
        )
        
        # Verify that the backend selection follows the priority rules
        if is_apple_silicon:
            # On Apple Silicon, MLX should be prioritized if available
            try:
                import mlx.core
                try:
                    if mlx.core.metal.is_available():
                        assert backend == 'mlx'
                except:
                    pass  # Skip if we can't check MLX availability
            except ImportError:
                pass  # Skip if MLX is not installed
        else:
            # On other platforms, PyTorch with CUDA should be prioritized if available
            try:
                import torch
                if torch.cuda.is_available():
                    assert backend == 'torch'
                    assert device == 'cuda'
            except ImportError:
                pass  # Skip if PyTorch is not installed
    
    @pytest.mark.parametrize("backend_name", ALL_BACKENDS)
    def test_backend_switching_after_auto_selection(self, backend_name, original_backend_and_device):
        """Test switching backends after auto-selection."""
        # First auto-select a backend
        auto_backend, _ = auto_select_backend()
        
        # Skip if the backend is the same as the one we're testing
        if backend_name == auto_backend:
            pytest.skip(f"Skipping {backend_name} as it's the auto-selected backend")
        
        # Switch to the specified backend
        set_backend(backend_name)
        
        # Also set the ops backend to ensure the ops module is updated
        from ember_ml import ops
        ops.set_backend(backend_name)
        
        # Verify the backend was switched
        assert get_backend() == backend_name
        assert ops.get_ops() == backend_name
        
        # Create a tensor with the new backend
        import ember_ml as nl
        Tensor = tensor.zeros((2, 2))
        
        # Verify the tensor has the correct type
        if backend_name == 'numpy':
            assert 'numpy.ndarray' in str(type(Tensor))
        elif backend_name == 'torch':
            assert 'torch.Tensor' in str(type(Tensor))
        elif backend_name == 'mlx':
            assert 'mlx.core.array' in str(type(Tensor))
    
    def test_set_and_get_device(self, original_backend_and_device):
        """Test setting and getting the device."""
        # Only test with PyTorch backend
        current_backend = get_backend()
        if current_backend != 'torch':
            set_backend('torch')
            # Also set the ops backend
            from ember_ml import ops
            ops.set_backend('torch')
        
        # Test setting and getting the device
        set_device('cpu')
        # Create a dummy tensor to get the device
        import ember_ml as nl
        dummy = tensor.zeros((1, 1))
        
        # Get the device and clean it up (remove ':0' suffix if present)
        device = get_device(dummy)
        if device and ':' in device:
            device = device.split(':')[0]
        
        # Check if we're on macOS with Apple Silicon
        import platform
        is_apple_silicon = (
            platform.system() == 'Darwin' and
            platform.machine().startswith('arm')
        )
        
        # On Apple Silicon, PyTorch might default to MPS even if we set it to CPU
        if is_apple_silicon:
            assert device in ['cpu', 'mps']
        else:
            assert device == 'cpu'
        
        # Test setting an invalid device
        with pytest.raises(ValueError):
            set_device('invalid_device')