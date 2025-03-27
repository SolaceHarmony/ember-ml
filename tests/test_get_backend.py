import pytest
import os
from ember_ml.backend import get_backend, set_backend

# List of all available backends to test
ALL_BACKENDS = ['numpy', 'torch', 'mlx']

def _clear_backend_cache():
    """Clear the backend cache."""
    from ember_ml.backend import _CURRENT_BACKEND, _CURRENT_BACKEND_MODULE
    _CURRENT_BACKEND = None
    _CURRENT_BACKEND_MODULE = None

@pytest.fixture
def original_backend():
    """Save and restore the original backend."""
    original = get_backend()
    yield original
    if original is not None:
        set_backend(original)

@pytest.mark.parametrize("backend_name", ALL_BACKENDS)
def test_get_backend_with_env_var(backend_name, original_backend):
    """Test get_backend with environment variable for all backends."""
    # Clear the backend cache
    _clear_backend_cache()

    # Set the environment variable
    os.environ['EMBER_ML_BACKEND'] = backend_name

    # Clear any existing backend file
    from ember_ml.backend import EMBER_BACKEND_FILE
    if EMBER_BACKEND_FILE.exists():
        EMBER_BACKEND_FILE.unlink()

    # Get the backend
    backend = get_backend()

    # Also set the ops backend to ensure the ops module is updated
    from ember_ml import ops
    ops.set_backend(backend)

    # Assert that the backend is set correctly
    assert backend == backend_name

@pytest.mark.parametrize("backend_name", ALL_BACKENDS)
def test_get_backend_after_set(backend_name, original_backend):
    """Test get_backend after setting the backend for all backends."""
    # Set the backend
    set_backend(backend_name)
    
    # Get the backend
    backend = get_backend()
    
    # Assert that the backend is set correctly
    assert backend == backend_name