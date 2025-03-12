"""
Test module for backend switching in Ember ML.

This module tests that the backend switching mechanism works correctly,
ensuring that tensor operations use the correct backend after switching.
"""

import pytest
import importlib
from ember_ml.backend import get_backend, set_backend
from ember_ml import ops

@pytest.fixture
def original_backend():
    """Save and restore the original backend."""
    original = get_backend()
    yield original
    if original is not None:
        set_backend(original)

def test_backend_switching(original_backend):
    """Test that backend switching works correctly."""
    # Get all available backends
    backends = ['numpy', 'torch', 'mlx']
    
    # Test switching to each backend
    for backend in backends:
        if backend == original_backend:
            continue
        
        # Switch to the backend
        set_backend(backend)
        
        # Verify the backend was switched
        assert get_backend() == backend
        
        # Create a tensor with the new backend
        tensor = ops.zeros((2, 2))
        
        # Verify the tensor has the correct type
        if backend == 'numpy':
            assert 'numpy.ndarray' in str(type(tensor))
        elif backend == 'torch':
            assert 'torch.Tensor' in str(type(tensor))
        elif backend == 'mlx':
            assert 'mlx.core.array' in str(type(tensor))
        
        # Perform a simple operation
        result = ops.add(tensor, tensor)
        
        # Verify the result has the correct type
        if backend == 'numpy':
            assert 'numpy.ndarray' in str(type(result))
        elif backend == 'torch':
            assert 'torch.Tensor' in str(type(result))
        elif backend == 'mlx':
            assert 'mlx.core.array' in str(type(result))

def test_backend_persistence(original_backend):
    """Test that the backend persists after switching."""
    # Get a backend different from the original
    backends = ['numpy', 'torch', 'mlx']
    test_backend = next((b for b in backends if b != original_backend), 'numpy')
    
    # Switch to the test backend
    set_backend(test_backend)
    
    # Verify the backend was switched
    assert get_backend() == test_backend
    
    # Create a tensor with the new backend
    tensor1 = ops.zeros((2, 2))
    
    # Create another tensor without switching backends
    tensor2 = ops.zeros((2, 2))
    
    # Verify both tensors have the same type
    assert type(tensor1) == type(tensor2)
    
    # Perform operations with both tensors
    result1 = ops.add(tensor1, tensor1)
    result2 = ops.add(tensor2, tensor2)
    
    # Verify both results have the same type
    assert type(result1) == type(result2)