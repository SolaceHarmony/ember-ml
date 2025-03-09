"""
Test script for the auto-selection of backends.

This script tests the auto-selection of backends based on available hardware.
"""

import os
import platform
import pytest
from ember_ml.backend import auto_select_backend, get_backend, set_backend, get_device, set_device

@pytest.fixture
def original_backend_and_device():
    """Save and restore the original backend and device."""
    original_backend = get_backend()
    # Create a dummy tensor to get the device
    import ember_ml as nl
    dummy = nl.zeros((1, 1))
    original_device = get_device(dummy)
    
    yield original_backend, original_device
    
    # Restore the original backend and device
    set_backend(original_backend)
    if original_device:
        set_device(original_device)

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
    
    def test_set_and_get_device(self, original_backend_and_device):
        """Test setting and getting the device."""
        # Only test with PyTorch backend
        current_backend = get_backend()
        if current_backend != 'torch':
            set_backend('torch')
        
        # Test setting and getting the device
        set_device('cpu')
        # Create a dummy tensor to get the device
        import ember_ml as nl
        dummy = nl.zeros((1, 1))
        
        # Check if we're on macOS with Apple Silicon
        import platform
        is_apple_silicon = (
            platform.system() == 'Darwin' and
            platform.machine().startswith('arm')
        )
        
        # On Apple Silicon, PyTorch might default to MPS even if we set it to CPU
        if is_apple_silicon:
            assert get_device(dummy) in ['cpu', 'mps']
        else:
            assert get_device(dummy) == 'cpu'
        
        # Test setting an invalid device
        with pytest.raises(ValueError):
            set_device('invalid_device')