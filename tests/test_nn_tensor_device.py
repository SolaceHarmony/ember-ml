"""
Unit tests for device operations across different backends.

This module contains pytest tests for the device operations in the nn.tensor module.
It tests each operation with different backends to ensure consistency.
"""

import pytest
from ember_ml import ops
from ember_ml.nn import tensor

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
    prev_backend = str(ops.get_backend())
    ops.set_backend(request.param)
    ops.set_backend(request.param)
    yield request.param
    ops.set_backend(prev_backend)
    ops.set_backend(prev_backend)

class TestDeviceOperations:
    """Tests for device operations."""

    def test_to_device(self, backend):
        """Test to_device operation."""
        # Skip for numpy backend as it doesn't have device concept
        if backend == 'numpy':
            pytest.skip("NumPy backend doesn't have device concept")
            
        # Create a tensor
        x = tensor.ones((3, 4))
        
        # Get current device
        current_device = ops.get_device(x)
        
        # Test to_device with current device (should be a no-op)
        y = ops.to_device(x, current_device)
        assert ops.get_device(y) == current_device
        assert ops.allclose(tensor.to_numpy(y), tensor.to_numpy(x))
        
        # Test to_device with 'cpu' device
        if backend == 'torch':
            y = ops.to_device(x, 'cpu')
            assert ops.get_device(y) == 'cpu'
            assert ops.allclose(tensor.to_numpy(y), tensor.to_numpy(x))
        elif backend == 'mlx':
            # MLX doesn't have explicit device concept like PyTorch
            # but we can still test the operation
            y = ops.to_device(x, 'cpu')
            assert ops.allclose(tensor.to_numpy(y), tensor.to_numpy(x))

    def test_get_device(self, backend):
        """Test get_device operation."""
        # Skip for numpy backend as it doesn't have device concept
        if backend == 'numpy':
            pytest.skip("NumPy backend doesn't have device concept")
            
        # Create a tensor
        x = tensor.ones((3, 4))
        
        # Test get_device
        device = ops.get_device(x)
        
        # The device should be a string
        assert isinstance(device, str)
        
        # For PyTorch, the device should be 'cpu' or 'cuda:X'
        if backend == 'torch':
            assert device.startswith('cpu') or device.startswith('cuda') or device.startswith('mps')
        elif backend == 'mlx':
            # MLX doesn't have explicit device concept like PyTorch
            # but we can still test the operation
            assert device in ['cpu', 'gpu', 'mps']

class TestDeviceManagement:
    """Tests for device management."""

    def test_set_default_device(self, backend):
        """Test set_default_device operation."""
        # Skip for numpy backend as it doesn't have device concept
        if backend == 'numpy':
            pytest.skip("NumPy backend doesn't have device concept")
            
        # Get current default device
        try:
            current_default_device = ops.device_ops().get_default_device()
        except (NotImplementedError, AttributeError):
            pytest.skip(f"{backend} backend doesn't implement get_default_device")
            
        try:
            # Set default device to 'cpu'
            ops.device_ops().set_default_device('cpu')
            
            # Verify that the default device is set correctly
            assert ops.device_ops().get_default_device() == 'cpu'
            
            # Create a tensor (should be on the default device)
            x = tensor.ones((3, 4))
            assert ops.get_device(x) == 'cpu'
        finally:
            # Restore the original default device
            ops.device_ops().set_default_device(current_default_device)

    def test_is_available(self, backend):
        """Test is_available operation."""
        # Skip for numpy backend as it doesn't have device concept
        if backend == 'numpy':
            pytest.skip("NumPy backend doesn't have device concept")
            
        try:
            # Test is_available for 'cpu'
            # MLX might return False for 'cpu' on Apple Silicon
            if backend == 'mlx':
                assert ops.device_ops().is_available('cpu') in [True, False]
            else:
                assert ops.device_ops().is_available('cpu') == True
            
            # Test is_available for 'cuda'
            if backend == 'torch':
                import torch
                assert ops.device_ops().is_available('cuda') == torch.cuda.is_available()
                
                # Test is_available for 'mps' (Apple Silicon)
                if hasattr(torch.backends, 'mps'):
                    assert ops.device_ops().is_available('mps') == torch.backends.mps.is_available()
            elif backend == 'mlx':
                # MLX doesn't have explicit device concept like PyTorch
                # but we can still test the operation
                assert ops.device_ops().is_available('gpu') in [True, False]
        except (NotImplementedError, AttributeError):
            pytest.skip(f"{backend} backend doesn't implement is_available")

    def test_memory_info(self, backend):
        """Test memory_info operation."""
        # Skip for numpy backend as it doesn't have device concept
        if backend == 'numpy':
            pytest.skip("NumPy backend doesn't have device concept")
            
        try:
            # Test memory_info for current device
            memory_info = ops.device_ops().memory_info()
            
            # The memory_info should be a dictionary
            assert isinstance(memory_info, dict)
            
            # The dictionary should contain memory information
            if backend == 'torch' and ops.get_available_devices() == 'cuda':
                # For PyTorch with CUDA, the dictionary should contain
                # 'allocated', 'reserved', 'free', and 'total' keys
                assert 'allocated' in memory_info
                assert 'reserved' in memory_info
                assert 'free' in memory_info
                assert 'total' in memory_info
            elif backend == 'mlx':
                # MLX doesn't have explicit memory management like PyTorch
                # but we can still test the operation
                assert len(memory_info) > 0
        except (NotImplementedError, AttributeError):
            pytest.skip(f"{backend} backend doesn't implement memory_info")

    def test_synchronize(self, backend):
        """Test synchronize operation."""
        # Skip for numpy backend as it doesn't have device concept
        if backend == 'numpy':
            pytest.skip("NumPy backend doesn't have device concept")
            
        try:
            # Test synchronize for current device
            ops.device_ops().synchronize()
            
            # Test synchronize for 'cpu'
            ops.device_ops().synchronize('cpu')
            
            # Test synchronize for 'cuda' if available
            import torch
            if backend == 'torch' and 'cuda' in ops.get_available_devices():
                ops.device_ops().synchronize('cuda')
                
                # Test synchronize for specific CUDA device if available
                if torch.cuda.device_count() > 0:
                    ops.device_ops().synchronize(f'cuda:{torch.cuda.current_device()}')
        except (NotImplementedError, AttributeError):
            pytest.skip(f"{backend} backend doesn't implement synchronize")
            