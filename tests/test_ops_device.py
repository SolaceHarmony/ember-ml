import pytest
from pytest import mark
from ember_ml import ops
from ember_ml.nn import tensor

# Define the backend order: numpy -> torch -> mlx

# --- NumPy Backend Setup & Tests ---
@mark.run(order=1)
def test_setup_numpy():
    """Set up the NumPy backend."""
    print("\n=== Setting backend to NumPy ===")
    ops.set_backend('numpy')
    assert ops.get_backend() == 'numpy'

@mark.run(order=1)
def test_get_available_devices_numpy():
    """Tests ops.get_available_devices with NumPy backend."""
    available = ops.get_available_devices()
    assert isinstance(available, list), "Should return a list"
    assert 'cpu' in available, "CPU device should always be available"
    # NumPy usually only reports 'cpu'
    assert 'cuda' not in available, "NumPy should not list CUDA"
    assert 'mps' not in available, "NumPy should not list MPS"

@mark.run(order=1)
def test_to_device_cuda_numpy():
    """Tests moving a tensor to CUDA device (NumPy backend - should skip)."""
    pytest.skip("NumPy backend does not support CUDA")

@mark.run(order=1)
def test_to_device_mps_numpy():
    """Tests moving a tensor to MPS device (NumPy backend - should skip)."""
    pytest.skip("NumPy backend does not support MPS")

@mark.run(order=1)
def test_get_device_numpy():
    """Tests ops.get_device returns 'cpu' for NumPy backend."""
    x_cpu = tensor.convert_to_tensor([1.0, 2.0]) # Device is implicitly CPU
    assert ops.get_device(x_cpu) == 'cpu'

# --- PyTorch Backend Setup & Tests ---
@mark.run(order=2)
def test_setup_torch():
    """Set up the PyTorch backend."""
    print("\n=== Setting backend to PyTorch ===")
    try:
        import torch
        ops.set_backend('torch')
        assert ops.get_backend() == 'torch'
    except ImportError:
        pytest.skip("PyTorch not available")

@mark.run(order=2)
def test_get_available_devices_torch():
    """Tests ops.get_available_devices with PyTorch backend."""
    available = ops.get_available_devices()
    assert isinstance(available, list), "Should return a list"
    assert 'cpu' in available, "CPU device should always be available"
    # Check for others based on actual availability
    try:
        import torch
        if torch.cuda.is_available():
            assert any(x.startswith('cuda') for x in available), "CUDA missing when available"
        if torch.backends.mps.is_available():
             assert 'mps' in available, "MPS missing when available"
    except ImportError: pass # Already skipped if torch not installed

@mark.run(order=2)
def test_to_device_cuda_torch():
    """Tests moving a tensor to CUDA device with PyTorch backend (if available)."""
    if 'cuda' not in ops.get_available_devices():
        pytest.skip("CUDA device not available")
    initial_device = 'cpu'
    target_device = 'cuda' # Use generic 'cuda'
    x_cpu = tensor.convert_to_tensor([1.0, 2.0], device=initial_device)
    assert ops.get_device(x_cpu).startswith(initial_device), "Initial tensor not on CPU"
    x_cuda = ops.to_device(x_cpu, target_device)
    assert ops.get_device(x_cuda).startswith(target_device), "Tensor not moved to CUDA"
    x_cpu_again = ops.to_device(x_cuda, initial_device)
    assert ops.get_device(x_cpu_again).startswith(initial_device), "Tensor not moved back to CPU"

@mark.run(order=2)
def test_to_device_mps_torch():
    """Tests moving a tensor to MPS device with PyTorch backend (if available)."""
    if 'mps' not in ops.get_available_devices():
        pytest.skip("MPS device not available")
    initial_device = 'cpu'
    target_device = 'mps'
    x_cpu = tensor.convert_to_tensor([1.0, 2.0], device=initial_device)
    assert ops.get_device(x_cpu).startswith(initial_device), "Initial tensor not on CPU"
    x_mps = ops.to_device(x_cpu, target_device)
    assert ops.get_device(x_mps).startswith(target_device), "Tensor not moved to MPS"
    x_cpu_again = ops.to_device(x_mps, initial_device)
    assert ops.get_device(x_cpu_again).startswith(initial_device), "Tensor not moved back to CPU"

@mark.run(order=2)
def test_get_device_torch():
    """Tests ops.get_device returns the correct device string for PyTorch."""
    x_cpu = tensor.convert_to_tensor([1.0, 2.0], device='cpu')
    assert ops.get_device(x_cpu).startswith('cpu')
    if 'cuda' in ops.get_available_devices():
        x_cuda = ops.to_device(x_cpu, 'cuda')
        assert ops.get_device(x_cuda).startswith('cuda')
    if 'mps' in ops.get_available_devices():
        x_mps = ops.to_device(x_cpu, 'mps')
        assert ops.get_device(x_mps).startswith('mps')

# --- MLX Backend Setup & Tests ---
@mark.run(order=3)
def test_setup_mlx():
    """Set up the MLX backend."""
    print("\n=== Setting backend to MLX ===")
    try:
        import mlx.core
        ops.set_backend('mlx')
        assert ops.get_backend() == 'mlx'
    except ImportError:
        pytest.skip("MLX not available")

@mark.run(order=3)
def test_get_available_devices_mlx():
    """Tests ops.get_available_devices with MLX backend."""
    available = ops.get_available_devices()
    assert isinstance(available, list), "Should return a list"
    assert 'cpu' in available, "CPU device should always be available"
    # MLX primarily uses 'gpu' on compatible hardware (like Apple Silicon MPS)
    import platform
    if platform.system() == "Darwin":
        # MLX might report 'gpu' or 'mps'
         assert 'gpu' in available or 'mps' in available, "GPU/MPS device should be available for mlx on macOS"

@mark.run(order=3)
def test_to_device_cuda_mlx():
    """Tests moving a tensor to CUDA device (MLX backend - should skip)."""
    pytest.skip("MLX backend does not support CUDA")

@mark.run(order=3)
def test_to_device_mps_mlx():
    """Tests moving a tensor to MPS/GPU device with MLX backend (if available)."""
    # MLX uses 'gpu' as the general term, maps to MPS on Apple Silicon
    if 'gpu' not in ops.get_available_devices() and 'mps' not in ops.get_available_devices():
        pytest.skip("MPS/GPU device not available for MLX")

    initial_device = 'cpu'
    # Use 'gpu' as the target, as MLX abstracts MPS under this usually
    target_device = 'gpu' if 'gpu' in ops.get_available_devices() else 'mps'

    x_cpu = tensor.convert_to_tensor([1.0, 2.0], device=initial_device)
    assert ops.get_device(x_cpu).startswith(initial_device), "Initial tensor not on CPU"

    x_gpu = ops.to_device(x_cpu, target_device)
    # MLX get_device might return 'gpu' even if specifically 'mps' was available
    assert ops.get_device(x_gpu) == target_device , f"Tensor not moved to {target_device}"

    x_cpu_again = ops.to_device(x_gpu, initial_device)
    assert ops.get_device(x_cpu_again).startswith(initial_device), "Tensor not moved back to CPU"


@mark.run(order=3)
def test_get_device_mlx():
    """Tests ops.get_device returns the correct device string for MLX."""
    x_cpu = tensor.convert_to_tensor([1.0, 2.0], device='cpu')
    assert ops.get_device(x_cpu).startswith('cpu')
    if 'gpu' in ops.get_available_devices() or 'mps' in ops.get_available_devices():
         target = 'gpu' if 'gpu' in ops.get_available_devices() else 'mps'
         x_gpu = ops.to_device(x_cpu, target)
         assert ops.get_device(x_gpu) == target

# TODO: Add tests for memory_usage and memory_info