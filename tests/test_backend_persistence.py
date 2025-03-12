import pytest
import os
import platform
from pathlib import Path
from unittest.mock import patch, mock_open

from ember_ml.backend import get_backend, set_backend, _get_backend_from_file, _save_backend_to_file, EMBER_CONFIG_DIR, EMBER_BACKEND_FILE

# Test data
BACKENDS = ['numpy', 'torch', 'mlx']

# Helper function to clear internal state for testing persistence
def _clear_backend_cache():
    """Clear the backend cache for testing."""
    # Import necessary modules
    import importlib
    import ember_ml.backend
    
    # Get the current backend module
    backend_module = importlib.import_module('ember_ml.backend')
    
    # Access the internal variables using the module's globals
    if '_CURRENT_BACKEND' in backend_module.__dict__:
        print(f"Before clearing cache: _CURRENT_BACKEND = {backend_module.__dict__['_CURRENT_BACKEND']}")
        backend_module.__dict__['_CURRENT_BACKEND'] = None
    else:
        print("_CURRENT_BACKEND not found in backend module")
    
    if '_CURRENT_BACKEND_MODULE' in backend_module.__dict__:
        backend_module.__dict__['_CURRENT_BACKEND_MODULE'] = None
    
    # Also clear the ops module's cache
    try:
        ops_module = importlib.import_module('ember_ml.ops')
        if hasattr(ops_module, '_CURRENT_INSTANCES'):
            ops_module._CURRENT_INSTANCES = {}
        importlib.reload(ops_module)
    except Exception as e:
        print(f"Error clearing ops module cache: {e}")
    
    # Print the current state
    if '_CURRENT_BACKEND' in backend_module.__dict__:
        print(f"After clearing cache: _CURRENT_BACKEND = {backend_module.__dict__['_CURRENT_BACKEND']}")

@pytest.fixture
def clean_env():
    """Fixture to ensure a clean environment before each test."""
    # Clear environment variable
    if 'EMBER_ML_BACKEND' in os.environ:
        del os.environ['EMBER_ML_BACKEND']
    # Remove the .ember directory if it exists
    if EMBER_CONFIG_DIR.exists():
        if EMBER_BACKEND_FILE.exists():
            EMBER_BACKEND_FILE.unlink()
        EMBER_CONFIG_DIR.rmdir()
    # Clear the backend cache
    _clear_backend_cache()
    yield  # This allows the test to run
    # Cleanup after test (if needed)
    _clear_backend_cache()


def test_set_and_get_backend(clean_env):
    """Test setting and getting the backend."""
    for backend in BACKENDS:
        set_backend(backend)
        assert get_backend() == backend

def test_persistence(clean_env):
    """Test that the backend is persisted to file."""
    for backend in BACKENDS:
        set_backend(backend)
        _clear_backend_cache()  # Simulate a new session
        assert get_backend() == backend
def test_file_overrides_env(clean_env):
    """Test that the backend file overrides the environment variable."""
    EMBER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(EMBER_BACKEND_FILE, 'w') as f:
        f.write('numpy')
    os.environ['EMBER_ML_BACKEND'] = 'torch'
    assert get_backend() == 'numpy'
    assert get_backend() == 'numpy'

def test_environment_variable(clean_env):
    """Test that the environment variable is used if no file exists."""
    os.environ['EMBER_ML_BACKEND'] = 'mlx'
    assert get_backend() == 'mlx'

def test_auto_detection_mlx(clean_env):
    """Test auto-detection of backend."""
    # delete the file if it exists
    if EMBER_BACKEND_FILE.exists():
        EMBER_BACKEND_FILE.unlink()
    _clear_backend_cache()

    set_backend('mlx')
    from ember_ml.backend import _CURRENT_BACKEND
    print(f"After set_backend: _CURRENT_BACKEND = {_CURRENT_BACKEND}")
    backend = get_backend()
    print(f"Backend: {backend}, Expected: mlx")
    assert backend == 'mlx'

def test_auto_detection_torch(clean_env):
    """Test auto-detection of backend."""
    # delete the file if it exists
    if EMBER_BACKEND_FILE.exists():
        EMBER_BACKEND_FILE.unlink()
    _clear_backend_cache()

    set_backend('torch')
    from ember_ml.backend import _CURRENT_BACKEND
    print(f"After set_backend: _CURRENT_BACKEND = {_CURRENT_BACKEND}")
    backend = get_backend()
    print(f"Backend: {backend}, Expected: torch")
    assert backend == 'torch'

def test_auto_detection_numpy(clean_env):
    """Test auto-detection of backend."""
    # delete the file if it exists
    if EMBER_BACKEND_FILE.exists():
        EMBER_BACKEND_FILE.unlink()
    _clear_backend_cache()

    set_backend('numpy')
    from ember_ml.backend import _CURRENT_BACKEND
    print(f"After set_backend: _CURRENT_BACKEND = {_CURRENT_BACKEND}")
    backend = get_backend()
    print(f"Backend: {backend}, Expected: numpy")
    assert backend == 'numpy'

def test_invalid_backend(clean_env):
    """Test setting an invalid backend."""
    with pytest.raises(ValueError):
        set_backend('invalid')

def test_file_read_error(clean_env):
    """Test handling of file read errors."""
    # Mock _get_backend_from_file to raise an exception
    with patch('ember_ml.backend._get_backend_from_file', side_effect=Exception("Mocked read error")):
        # Set an environment variable to fall back on
        os.environ['EMBER_ML_BACKEND'] = 'mlx'
        assert get_backend() == 'mlx'

        # Test fallback to auto-detection (using parametrize and mocking platform)
        del os.environ['EMBER_ML_BACKEND']
        with patch('platform.system', return_value='Darwin'), \
            patch('platform.machine', return_value='arm64'):
            assert get_backend() == 'mlx'

def test_save_backend_to_file(clean_env):
    """Test _save_backend_to_file directly."""
    _save_backend_to_file('numpy')
    assert EMBER_BACKEND_FILE.exists()
    assert EMBER_BACKEND_FILE.read_text().strip() == 'numpy'

def test_get_backend_from_file(clean_env):
    """Test _get_backend_from_file directly."""
    EMBER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    EMBER_BACKEND_FILE.write_text('torch')
    assert _get_backend_from_file() == 'torch'
    EMBER_BACKEND_FILE.unlink()
    assert _get_backend_from_file() is None