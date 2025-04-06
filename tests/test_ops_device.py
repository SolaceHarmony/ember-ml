import pytest
from ember_ml import ops
from ember_ml.nn import tensor

# Assume conftest.py provides 'backend' fixture

def test_get_available_devices(backend):
    """Tests if ops.get_available_devices returns a list containing 'cpu'."""
    ops.set_backend(backend)
    available = ops.get_available_devices()
    assert isinstance(available, list), "Should return a list"
    assert 'cpu' in available, "CPU device should always be available"
    # Note: Presence of 'cuda' or 'mps' depends on the testing environment hardware.
    # We only assert 'cpu' for basic validity.

@pytest.mark.skipif(not ops.get_available_devices() or 'cuda' not in ops.get_available_devices(), reason="CUDA device not available")
def test_to_device_cuda(backend):
    """Tests moving a tensor to CUDA device (if available)."""
    ops.set_backend(backend)
    if backend == 'numpy':
         pytest.skip("NumPy backend does not support CUDA") # Skip numpy for cuda test

    initial_device = 'cpu'
    target_device = 'cuda'
    x_cpu = tensor.convert_to_tensor([1.0, 2.0], device=initial_device)
    assert ops.get_device(x_cpu) == initial_device, "Initial tensor not on CPU"

    x_cuda = ops.to_device(x_cpu, target_device)
    assert ops.get_device(x_cuda) == target_device, "Tensor not moved to CUDA"

    # Test moving back to CPU
    x_cpu_again = ops.to_device(x_cuda, initial_device)
    assert ops.get_device(x_cpu_again) == initial_device, "Tensor not moved back to CPU"

@pytest.mark.skipif(not ops.get_available_devices() or 'mps' not in ops.get_available_devices(), reason="MPS device not available")
def test_to_device_mps(backend):
    """Tests moving a tensor to MPS device (if available)."""
    ops.set_backend(backend)
    if backend == 'numpy':
         pytest.skip("NumPy backend does not support MPS") # Skip numpy for mps test

    initial_device = 'cpu'
    target_device = 'mps'
    x_cpu = tensor.convert_to_tensor([1.0, 2.0], device=initial_device)
    assert ops.get_device(x_cpu) == initial_device, "Initial tensor not on CPU"

    x_mps = ops.to_device(x_cpu, target_device)
    assert ops.get_device(x_mps) == target_device, "Tensor not moved to MPS"

     # Test moving back to CPU
    x_cpu_again = ops.to_device(x_mps, initial_device)
    assert ops.get_device(x_cpu_again) == initial_device, "Tensor not moved back to CPU"

def test_get_device(backend):
    """Tests ops.get_device returns the correct device string."""
    ops.set_backend(backend)
    x_cpu = tensor.convert_to_tensor([1.0, 2.0], device='cpu')
    assert ops.get_device(x_cpu) == 'cpu'

    # Add tests for other devices if available, similar to test_to_device_*

# TODO: Add tests for memory_usage and memory_info if feasible/deterministic asserts can be made.