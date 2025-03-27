"""
Test script for the gather operation.

This script tests the gather operation across all backends
to ensure it works correctly for PCA feature extraction.
"""

from ember_ml import ops
from ember_ml.nn import tensor
import numpy as np

def test_gather():
    """Test the gather operation across all backends."""
    print("Testing gather operation across backends")
    
    # Create test tensors
    x = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indices = tensor.convert_to_tensor([0, 2])
    
    # Test gather operation
    result = tensor.gather(x, indices, axis=0)
    print("Result of gather operation:")
    print(result)
    
    # Verify the result
    expected = np.array([[1, 2, 3], [7, 8, 9]])
    expected_tensor = tensor.convert_to_tensor(expected)
    is_equal = ops.allclose(result, expected_tensor)
    print("Result matches expected output:", is_equal)
    
    return is_equal

if __name__ == "__main__":
    test_gather()