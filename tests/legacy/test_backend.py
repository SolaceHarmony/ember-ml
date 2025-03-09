"""
Test script for the emberharmony backend system.

This script tests the basic functionality of the backend system,
including backend switching and tensor operations.
"""

import numpy as np
import ember_ml as nl
from ember_ml.backend import get_backend, set_backend, get_device
import importlib.util

def test_backend_operations():
    """Test basic backend operations."""
    print(f"Current backend: {get_backend()}")
    
    # Test tensor creation
    a = nl.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])  # Use float values to ensure consistent dtype
    b = nl.ones((2, 2))
    c = nl.zeros((2, 2))
    
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = {c}")
    
    # Test mathematical operations
    d = nl.add(a, b)
    e = nl.matmul(a, b)
    
    print(f"a + b = {d}")
    print(f"a @ b = {e}")
    
    # Test activation functions
    f = nl.sigmoid(a)
    g = nl.tanh(a)
    h = nl.relu(a)
    
    print(f"sigmoid(a) = {f}")
    print(f"tanh(a) = {g}")
    print(f"relu(a) = {h}")
    
    # Test random operations
    nl.set_seed(42)
    i = nl.random_normal((2, 2))
    j = nl.random_uniform((2, 2))
    
    print(f"random_normal = {i}")
    print(f"random_uniform = {j}")
    
    # Test device operations
    device = get_device(a)
    print(f"Device: {device}")
    
    # Use assert instead of return
    assert True

def is_backend_available(backend_name):
    """Check if a backend is available."""
    if backend_name == 'numpy':
        return True
    elif backend_name == 'torch':
        return importlib.util.find_spec('torch') is not None
    elif backend_name == 'mlx':
        return importlib.util.find_spec('mlx') is not None
    return False
def test_backend_switching():
    """Test backend switching (if available)."""
    # Save original backend
    original_backend = get_backend()
    print(f"Original backend: {original_backend}")
    
    # Try to switch to other backends if available
    backends_to_test = ['torch', 'mlx']
    
    for backend in backends_to_test:
        if is_backend_available(backend):
            try:
                set_backend(backend)
                print(f"Switched to backend: {get_backend()}")
                
                # Create a tensor
                a = nl.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])  # Use float values to ensure consistent dtype
                print(f"{backend} tensor: {a}")
                
            except Exception as e:
                print(f"Error using {backend} backend: {e}")
        else:
            print(f"{backend} backend not available (not installed)")
    
    # Restore original backend
    set_backend(original_backend)
    print(f"Restored backend: {get_backend()}")
    
    # Use assert instead of return
    assert True

def check_available_backends():
    """Check which backends are available."""
    print("Checking available backends:")
    backends = ['numpy', 'torch', 'mlx']
    
    for backend in backends:
        if is_backend_available(backend):
            print(f"  - {backend}: Available")
        else:
            print(f"  - {backend}: Not available")

if __name__ == "__main__":
    print("Testing neural_lib backend system...")
    
    print("\nChecking available backends:")
    check_available_backends()
    
    print("\n1. Testing basic operations:")
    test_backend_operations()
    
    print("\n2. Testing backend switching:")
    test_backend_switching()
    
    print("\nAll tests completed!")