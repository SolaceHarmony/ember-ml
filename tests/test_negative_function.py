"""
Test for the ops.negative function.
"""

import pytest
from ember_ml import ops


def test_negative_function():
    """Test that the negative function works correctly."""
    # Test with scalar
    x = ops.convert_to_tensor(5)
    result = ops.negative(x)
    assert ops.to_numpy(result) == -5
    
    # Test with 1D tensor
    x = ops.convert_to_tensor([1, 2, 3])
    result = ops.negative(x)
    expected = [-1, -2, -3]
    result_np = ops.to_numpy(result)
    for i in range(len(expected)):
        assert result_np[i] == expected[i]
    
    # Test with 2D tensor
    x = ops.convert_to_tensor([[1, 2], [3, 4]])
    result = ops.negative(x)
    expected = [[-1, -2], [-3, -4]]
    result_np = ops.to_numpy(result)
    for i in range(len(expected)):
        for j in range(len(expected[i])):
            assert result_np[i][j] == expected[i][j]


if __name__ == "__main__":
    test_negative_function()
    print("All tests passed!")