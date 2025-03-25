"""
Test script for scatter operations.

This script tests the scatter operations across all backends
to ensure they work correctly.
"""

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor import EmberTensor
# First test with MLX backend
ops.set_backend("mlx")
print("Testing with MLX backend")
def test_scatter_add():
    """Test the scatter addition operation."""
    print("Testing scatter add operation")
    
    # Create test tensors
    src = tensor.convert_to_tensor([1.0, 2.0, 3.0],dtype=tensor.float32)
    index = tensor.convert_to_tensor([0, 2, 0],dtype=tensor.int32)
    dim_size = 3
    
    # Test scatter add operation
    result = tensor.scatter(src, index, dim_size, aggr="add")
    print("Result of scatter add operation:")
    print(result)
    
    # Debug print to understand what's happening
    print("src shape:", tensor.shape(src))
    print("index shape:", tensor.shape(index))
    
    # Verify the result
    expected = EmberTensor([4.0, 0.0, 2.0], dtype=tensor.float32)  # 1+3 at index 0, 0 at index 1, 2 at index 2
    is_equal = ops.allclose(result, expected)
    print("Result matches expected output:", is_equal)
    
    return is_equal

def test_scatter_max():
    """Test the scatter max operation."""
    print("\nTesting scatter max operation")
    
    # Create test tensors
    src = tensor.convert_to_tensor([1.0, 5.0, 3.0])
    index = tensor.convert_to_tensor([0, 0, 1])
    dim_size = 2
    
    # Test scatter max operation
    result = tensor.scatter(src, index, dim_size, aggr="max")
    print("Result of scatter max operation:")
    print(result)
    
    # Verify the result
    expected = EmberTensor([5.0, 3.0], dtype=tensor.float32)  # max(1,5) at index 0, 3 at index 1
    is_equal = ops.allclose(result, expected)
    print("Result matches expected output:", is_equal)
    
    return is_equal

def test_scatter_mean():
    """Test the scatter mean operation."""
    print("\nTesting scatter mean operation")
    
    # Create test tensors
    src = tensor.convert_to_tensor([1.0, 5.0, 3.0], dtype=tensor.float32)
    index = tensor.convert_to_tensor([0, 0, 1], dtype=tensor.int32)
    dim_size = 2
    
    # Print src and index for debug
    print("src values:", src)
    print("index values:", index)
    
    # Test scatter mean operation
    print("\nRunning with backend:", ops.get_backend())
    print("scatter_mean test: src values =", [1.0, 5.0, 3.0])
    print("scatter_mean test: indices =", [0, 0, 1])
    print("scatter_mean test: dim_size =", 2)
    result = tensor.scatter(src, index, dim_size, aggr="mean")
    print("Result of scatter mean operation:")
    print(result)
    
    # Verify the result
    expected = EmberTensor([3.0, 3.0], dtype=tensor.float32)  # mean(1,5) at index 0, 3 at index 1
    is_equal = ops.allclose(result, expected)
    print("Result matches expected output:", is_equal)
    
    return is_equal

def test_scatter_multi_dimensional():
    """Test scatter with multi-dimensional tensors."""
    print("\nTesting scatter with multi-dimensional tensors")
    
    # Create test tensors
    src = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = tensor.convert_to_tensor([0, 2, 0])
    dim_size = 3
    
    # Test scatter add operation with axis=0
    result = tensor.scatter(src, index, dim_size, aggr="add", axis=0)
    print("Result of scatter add operation with axis=0:")
    print(result)
    
    # Verify the result
    expected = EmberTensor([[6.0, 8.0], [0.0, 0.0], [3.0, 4.0]], dtype=tensor.float32)
    is_equal = ops.allclose(result, expected)
    print("Result matches expected output:", is_equal)
    
    return is_equal

def test_all():
    """Run all tests and report results."""
    tests = [
        test_scatter_add,
        test_scatter_max,
        test_scatter_mean,
        test_scatter_multi_dimensional
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\nSummary:")
    for i, test in enumerate(tests):
        print(f"{test.__name__}: {'PASS' if results[i] else 'FAIL'}")
    
    return all(results)

def test_all_backends():
    """Test all backends and report results."""
    backends = ["mlx", "torch", "numpy"]
    results = {}
    
    for backend in backends:
        print(f"\n{'-'*50}")
        print(f"Testing with {backend.upper()} backend")
        print(f"{'-'*50}")
        ops.set_backend(backend)
        results[backend] = test_all()
    
    print("\n\nSummary of all backends:")
    for backend, result in results.items():
        print(f"{backend.upper()} tests: {'PASS' if result else 'FAIL'}")
    
    return all(results.values())

if __name__ == "__main__":
    test_all_backends()