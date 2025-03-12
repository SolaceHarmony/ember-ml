"""
Test for the EmberTensor.size() method.
"""

import pytest
from ember_ml.ops.tensor import EmberTensor
from ember_ml import ops


def test_ember_tensor_size():
    """Test that the size() method returns the correct number of elements."""
    # Test with 1D tensor
    tensor_1d = EmberTensor([1, 2, 3, 4, 5])
    assert tensor_1d.size() == 5
    
    # Test with 2D tensor
    tensor_2d = EmberTensor([[1, 2, 3], [4, 5, 6]])
    assert tensor_2d.size() == 6
    
    # Test with 3D tensor
    tensor_3d = EmberTensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert tensor_3d.size() == 8
    
    # Test with empty tensor
    tensor_empty = EmberTensor([])
    assert tensor_empty.size() == 0
    
    # Test with scalar tensor
    tensor_scalar = EmberTensor(5)
    assert tensor_scalar.size() == 1
    
    # Test with zeros
    tensor_zeros = EmberTensor.zeros((2, 3, 4))
    assert tensor_zeros.size() == 24
    
    # Test with ones
    tensor_ones = EmberTensor.ones((3, 2, 1))
    assert tensor_ones.size() == 6


if __name__ == "__main__":
    test_ember_tensor_size()
    print("All tests passed!")