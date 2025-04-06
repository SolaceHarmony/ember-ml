import pytest
from ember_ml.backend import set_backend, get_backend # Keep backend functions
from ember_ml import ops # Import ops module for device functions

# Assuming conftest.py provides a 'backend' fixture that parametrizes tests
# over ['numpy', 'torch', 'mlx']

def test_set_and_get_backend(backend):
    """
    Tests if setting the backend correctly updates the global state
    and if get_backend retrieves the currently set backend.
    """
    original_backend = get_backend() # Store original backend
    try:
        set_backend(backend)
        assert get_backend() == backend, f"Failed to set backend to {backend}"
    finally:
        set_backend(original_backend) # Restore original backend

def test_get_available_devices(): # Renamed test function for clarity
    """
    Tests if ops.get_available_devices returns a list of expected device names.
    """
    available = ops.get_available_devices() # Corrected function call using ops module
    assert isinstance(available, list), "Should return a list"
    assert 'cpu' in available, "CPU device should always be available"
    # Add checks for torch and mlx if they are expected to be installed/available
    # For now, we just check the type and numpy presence.
    # More robust checks could verify installation status if needed.