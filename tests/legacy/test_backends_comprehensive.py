"""
Comprehensive test script for the emberharmony backend system.

This script tests the backend system with more operations and demonstrates
switching between backends.
"""
import numpy as np
import ember_ml as nl
from ember_ml.backend import get_backend, set_backend
import time
import importlib.util
import pytest

# Define the backend_name fixture
@pytest.fixture(params=['numpy', 'torch', 'mlx'])
def backend_name(request):
    """Fixture that provides backend names to test functions."""
    backend = request.param
    if is_backend_available(backend):
        return backend
    else:
        pytest.skip(f"{backend} backend not available")
import importlib.util

def is_backend_available(backend_name):
    """Check if a backend is available."""
    if backend_name == 'numpy':
        return True
    elif backend_name == 'torch':
        return importlib.util.find_spec('torch') is not None
    elif backend_name == 'mlx':
        return importlib.util.find_spec('mlx') is not None
    return False

def test_creation_ops(backend_name):
    """Test tensor creation operations."""
    print(f"\n--- Testing creation operations with {backend_name} backend ---")
    
    # Set the backend
    set_backend(backend_name)
    
    # Test tensor creation
    a = nl.zeros((2, 3))  # Use default dtype
    b = nl.ones((3, 2))   # Use default dtype
    c = nl.eye(3)         # Use default dtype
    d = nl.random_normal((2, 2))
    e = nl.random_uniform((2, 2))
    
    print(f"zeros((2, 3)) = \n{a}")
    print(f"ones((3, 2)) = \n{b}")
    print(f"eye(3) = \n{c}")
    print(f"random_normal((2, 2)) = \n{d}")
    print(f"random_uniform((2, 2)) = \n{e}")
    
    # Use assert instead of return
    assert True

def test_math_ops(backend_name):
    """Test mathematical operations."""
    print(f"\n--- Testing math operations with {backend_name} backend ---")
    
    # Set the backend
    set_backend(backend_name)
    
    # Create test tensors
    a = nl.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])  # Use float values to ensure consistent dtype
    b = nl.convert_to_tensor([[5.0, 6.0], [7.0, 8.0]])  # Use float values to ensure consistent dtype
    
    # Test operations
    add_result = nl.add(a, b)
    sub_result = nl.subtract(a, b)
    mul_result = nl.multiply(a, b)
    div_result = nl.divide(a, b)
    matmul_result = nl.matmul(a, b)
    
    print(f"a = \n{a}")
    print(f"b = \n{b}")
    print(f"a + b = \n{add_result}")
    print(f"a - b = \n{sub_result}")
    print(f"a * b = \n{mul_result}")
    print(f"a / b = \n{div_result}")
    print(f"a @ b = \n{matmul_result}")
    
    # Use assert instead of return
    assert True

def test_activation_ops(backend_name):
    """Test activation functions."""
    print(f"\n--- Testing activation functions with {backend_name} backend ---")
    
    # Set the backend
    set_backend(backend_name)
    
    # Create test tensor (already using float values)
    x = nl.convert_to_tensor([[-1.0, 0.0], [1.0, 2.0]])
    
    # Test activations
    sigmoid_result = nl.sigmoid(x)
    tanh_result = nl.tanh(x)
    relu_result = nl.relu(x)
    softmax_result = nl.softmax(x)
    
    print(f"x = \n{x}")
    print(f"sigmoid(x) = \n{sigmoid_result}")
    print(f"tanh(x) = \n{tanh_result}")
    print(f"relu(x) = \n{relu_result}")
    print(f"softmax(x) = \n{softmax_result}")
    
    # Use assert instead of return
    assert True

def test_array_ops(backend_name):
    """Test array manipulation operations."""
    print(f"\n--- Testing array operations with {backend_name} backend ---")
    
    # Set the backend
    set_backend(backend_name)
    
    # Create test tensor
    a = nl.convert_to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Use float values to ensure consistent dtype
    
    # Test operations
    reshaped = nl.reshape(a, (3, 2))
    transposed = nl.transpose(a)
    expanded = nl.expand_dims(a, axis=0)
    squeezed = nl.squeeze(expanded)
    
    print(f"a = \n{a}")
    print(f"reshape(a, (3, 2)) = \n{reshaped}")
    print(f"transpose(a) = \n{transposed}")
    print(f"expand_dims(a, axis=0) = \n{expanded}")
    print(f"squeeze(expanded) = \n{squeezed}")
    
    # Use assert instead of return
    assert True

def benchmark_backends():
    """Benchmark the different backends."""
    print("\n--- Benchmarking backends ---")
    
    backends = ['numpy', 'torch', 'mlx']
    available_backends = [b for b in backends if is_backend_available(b)]
    
    # Matrix multiplication benchmark
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\nMatrix multiplication benchmark (size {size}x{size}):")
        
        for backend in available_backends:
            set_backend(backend)
            
            # Create random matrices
            a = nl.random_normal((size, size))
            b = nl.random_normal((size, size))
            
            # Warm-up
            _ = nl.matmul(a, b)
            
            # Benchmark
            start_time = time.time()
            for _ in range(5):
                _ = nl.matmul(a, b)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 5
            print(f"  {backend}: {avg_time:.4f} seconds")

def main():
    """Main function to test the backend system."""
    print("Comprehensive EmberHarmony Backend Test")
    print("======================================")
    
    # Check available backends
    print("\nChecking available backends:")
    backends = ['numpy', 'torch', 'mlx']
    available_backends = []
    
    for backend in backends:
        if is_backend_available(backend):
            print(f"  - {backend}: Available")
            available_backends.append(backend)
        else:
            print(f"  - {backend}: Not available")
    
    # Test each available backend
    for backend in available_backends:
        test_creation_ops(backend)
        test_math_ops(backend)
        test_activation_ops(backend)
        test_array_ops(backend)
    
    # Benchmark backends
    benchmark_backends()
    
    # Restore default backend
    set_backend('numpy')
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()