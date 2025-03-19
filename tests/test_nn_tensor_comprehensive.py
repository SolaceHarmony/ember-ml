"""
Comprehensive tests for the EmberTensor class.

This module provides comprehensive tests for the EmberTensor class,
covering all its methods and properties.
"""

import pytest
from ember_ml import ops
from ember_ml.nn.tensor import EmberTensor


def test_initialization():
    """Test initialization of EmberTensor."""
    # Test with scalar
    t1 = EmberTensor(5)
    assert t1.shape == ()
    assert t1.size() == 1

    # Test with list
    t2 = EmberTensor([1, 2, 3])
    assert t2.shape == (3,)
    assert t2.size() == 3

    # Test with nested list
    t3 = EmberTensor([[1, 2], [3, 4]])
    assert t3.shape == (2, 2)
    assert t3.size() == 4

    # Test with name
    t4 = EmberTensor(5, name="test_tensor")
    assert t4.name == "test_tensor"


def test_shape_methods():
    """Test shape-related methods."""
    t = EmberTensor([[1, 2, 3], [4, 5, 6]])

    # Test shape property
    assert t.shape == (2, 3)

    # Test shape_as_list
    assert t.shape_as_list() == [2, 3]

    # Test shape_as_tuple
    assert t.shape_as_tuple() == (2, 3)

    # Test shape_at
    assert t.shape_at(0) == 2
    assert t.shape_at(1) == 3

    # Test size
    assert t.size() == 6


def test_size_method():
    """Test the size method specifically."""
    # Test with scalar
    t1 = EmberTensor(5)
    assert t1.size() == 1

    # Test with 1D tensor
    t2 = EmberTensor([1, 2, 3, 4, 5])
    assert t2.size() == 5

    # Test with 2D tensor
    t3 = EmberTensor([[1, 2, 3], [4, 5, 6]])
    assert t3.size() == 6

    # Test with 3D tensor
    t4 = EmberTensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert t4.size() == 8

    # Test with empty tensor
    t5 = EmberTensor([])
    assert t5.size() == 0


def test_dtype_and_device():
    """Test dtype and device properties."""
    # Test dtype
    t = EmberTensor([1, 2, 3])
    assert t.dtype is not None

    # Test device
    assert t.device is None  # Default is None


def test_to_method():
    """Test the to method for dtype conversion."""
    # Create a tensor with integer values
    t = EmberTensor([1, 2, 3])
    
    # Define common dtypes as strings
    dtypes = ['float32', 'float64', 'int32', 'int64']
    
    # Convert to a different dtype
    new_dtype = dtypes[0]  # Using 'float32'
    t2 = t.to(dtype=new_dtype)
    
    # Verify that the shape is preserved
    assert t2.shape == t.shape
    
    # Verify that the values are preserved
    # Use ops.equal to compare tensors in a backend-agnostic way
    comparison = ops.equal(t2.data, t.data)
    # Use ops.all to check if all elements are equal
    all_equal = ops.all(comparison)
    # Convert to a Python value using ops.item
    assert ops.item(all_equal)


def test_data_and_backend():
    """Test data and backend properties."""
    t = EmberTensor([1, 2, 3])

    # Test data property
    assert t.data is not None

    # Test backend property
    assert t.backend is not None
    assert isinstance(t.backend, str)


def test_string_representation():
    """Test string representation methods."""
    t = EmberTensor([1, 2, 3])
    
    # Test __repr__
    repr_str = repr(t)
    assert "EmberTensor" in repr_str
    assert "shape" in repr_str
    
    # Test __str__
    str_str = str(t)
    assert "EmberTensor" in str_str


def test_arithmetic_operations():
    """Test arithmetic operations."""
    t1 = EmberTensor([1, 2, 3])
    t2 = EmberTensor([4, 5, 6])
    
    # Test addition
    t_add = t1 + t2
    assert t_add.shape == (3,)
    assert ops.item(t_add[0]) == 5
    assert ops.item(t_add[1]) == 7
    assert ops.item(t_add[2]) == 9
    
    # Test subtraction
    t_sub = t2 - t1
    assert t_sub.shape == (3,)
    assert ops.item(t_sub[0]) == 3
    assert ops.item(t_sub[1]) == 3
    assert ops.item(t_sub[2]) == 3
    
    # Test multiplication
    t_mul = t1 * t2
    assert t_mul.shape == (3,)
    assert ops.item(t_mul[0]) == 4
    assert ops.item(t_mul[1]) == 10
    assert ops.item(t_mul[2]) == 18
    
    # Test division
    t_div = t2 / t1
    assert t_div.shape == (3,)
    assert ops.item(t_div[0]) == 4
    assert abs(ops.item(t_div[1]) - 2.5) < 1e-6
    assert ops.item(t_div[2]) == 2
    
    # Test negation
    t_neg = -t1
    assert t_neg.shape == (3,)
    assert ops.item(t_neg[0]) == -1
    assert ops.item(t_neg[1]) == -2
    assert ops.item(t_neg[2]) == -3
    
    # Test absolute value
    t_abs = abs(-t1)
    assert t_abs.shape == (3,)
    assert ops.item(t_abs[0]) == 1
    assert ops.item(t_abs[1]) == 2
    assert ops.item(t_abs[2]) == 3


def test_comparison_operations():
    """Test comparison operations."""
    t1 = EmberTensor([1, 2, 3])
    t2 = EmberTensor([1, 2, 3])
    t3 = EmberTensor([4, 5, 6])
    
    # Test equality
    assert t1 == t2
    assert not (t1 == t3)
    
    # Test inequality
    assert not (t1 != t2)
    assert t1 != t3


def test_shape_operations():
    """Test shape operations."""
    t = EmberTensor([[1, 2, 3], [4, 5, 6]])
    
    # Test reshape
    t_reshaped = t.reshape((3, 2))
    assert t_reshaped.shape == (3, 2)
    
    # Test transpose
    t_transposed = t.transpose()
    assert t_transposed.shape == (3, 2)
    
    # Test squeeze and unsqueeze
    t_unsqueezed = t.unsqueeze(0)
    assert t_unsqueezed.shape == (1, 2, 3)
    
    t_squeezed = t_unsqueezed.squeeze(0)
    assert t_squeezed.shape == (2, 3)


def test_reduction_operations():
    """Test reduction operations."""
    t = EmberTensor([[1, 2, 3], [4, 5, 6]])
    
    # Test sum
    t_sum = t.sum()
    assert t_sum.size() == 1
    assert ops.item(t_sum) == 21
    
    t_sum_axis0 = t.sum(axis=0)
    assert t_sum_axis0.shape == (3,)
    assert ops.item(t_sum_axis0[0]) == 5
    assert ops.item(t_sum_axis0[1]) == 7
    assert ops.item(t_sum_axis0[2]) == 9
    
    # Test mean
    t_mean = t.mean()
    assert t_mean.size() == 1
    assert ops.item(t_mean) == 3.5
    
    # Test max
    t_max = t.max()
    assert t_max.size() == 1
    assert ops.item(t_max) == 6
    
    # Test min
    t_min = t.min()
    assert t_min.size() == 1
    assert ops.item(t_min) == 1


def test_static_creation_methods():
    """Test static methods for tensor creation."""
    # Test zeros
    t_zeros = EmberTensor.zeros((2, 3))
    assert t_zeros.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            assert ops.item(t_zeros[i, j]) == 0
    
    # Test ones
    t_ones = EmberTensor.ones((2, 3))
    assert t_ones.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            assert ops.item(t_ones[i, j]) == 1
    
    # Test full
    t_full = EmberTensor.full((2, 3), 7)
    assert t_full.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            assert ops.item(t_full[i, j]) == 7
    
    # Test arange
    t_arange = EmberTensor.arange(0, 5)
    assert t_arange.shape == (5,)
    for i in range(5):
        assert ops.item(t_arange[i]) == i
    
    # Test linspace
    t_linspace = EmberTensor.linspace(0, 1, 5)
    assert t_linspace.shape == (5,)
    assert ops.item(t_linspace[0]) == 0
    assert ops.item(t_linspace[-1]) == 1
    
    # Test eye
    t_eye = EmberTensor.eye(3)
    assert t_eye.shape == (3, 3)
    for i in range(3):
        for j in range(3):
            expected = 1 if i == j else 0
            assert ops.item(t_eye[i, j]) == expected
    
    # Test zeros_like and ones_like
    t = EmberTensor([[1, 2], [3, 4]])
    t_zeros_like = EmberTensor.zeros_like(t)
    assert t_zeros_like.shape == (2, 2)
    for i in range(2):
        for j in range(2):
            assert ops.item(t_zeros_like[i, j]) == 0
    
    t_ones_like = EmberTensor.ones_like(t)
    assert t_ones_like.shape == (2, 2)
    for i in range(2):
        for j in range(2):
            assert ops.item(t_ones_like[i, j]) == 1


def test_conversion_methods():
    """Test conversion methods."""
    # Create a list
    data = [1, 2, 3]
    
    # Test from_tensor
    t1 = EmberTensor(data)
    t_from_tensor = EmberTensor.from_tensor(t1)
    assert t_from_tensor.shape == (3,)
    for i in range(3):
        assert ops.item(t_from_tensor[i]) == data[i]


# Add a new test specifically for NumPy conversion
def test_numpy_conversion():
    """Test conversion to NumPy arrays."""
    t = EmberTensor([[1, 2, 3], [4, 5, 6]])
    np_arr = ops.to_numpy(t.data)
    assert np_arr.shape == (2, 3)