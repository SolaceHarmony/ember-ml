"""
Test to check if tensor(...) returns an EmberTensor-like object.
"""

import pytest
from ember_ml import ops, tensor


@pytest.fixture
def original_backend():
    """Fixture to save and restore the original backend."""
    original = ops.get_backend()
    yield original
    # Ensure original is not None before setting it
    if original is not None:
        ops.set_backend(original)
    else:
        # Default to 'numpy' if original is None
        ops.set_backend('numpy')


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_tensor_returns_backend_tensor(backend_name, original_backend):
    """Test if tensor(...) returns a backend tensor wrapper with expected attributes."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Create a tensor using tensor(...)
        data = [1, 2, 3]
        t = tensor(data)
        
        # Check if it's an EmberTensor object
        print(f"Type of t: {type(t)}")
        # Check expected attributes exist
        print(f"Has shape attribute: {hasattr(t, 'shape')}")
        if hasattr(t, 'shape'):
            print(f"Shape: {t.shape}")
        
        print(f"Has dtype attribute: {hasattr(t, 'dtype')}")
        if hasattr(t, 'dtype'):
            print(f"Dtype: {t.dtype}")
        
        # Basic sanity
        assert hasattr(t, 'shape') and hasattr(t, 'dtype')
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")