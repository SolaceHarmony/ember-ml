"""
Tests for EmberTensor operations.

This module tests the functionality of the EmberTensor class and its operations.
"""

import pytest
from ember_ml.ops.tensor import EmberTensor
from ember_ml import ops
from ember_ml.backend import set_backend, get_backend

class TestEmberTensorOperations:
    """Test cases for EmberTensor operations."""

    @pytest.fixture(autouse=True)
    def setup_test(self):
        """Set up the test."""
        # Store the original backend
        self.original_backend = get_backend()
        # Set the backend to numpy for testing
        set_backend('numpy')
        yield
        # Restore the original backend
        set_backend(self.original_backend)

    def test_creation(self):
        """Test EmberTensor creation."""
        # Create a tensor from a list
        tensor = EmberTensor([1, 2, 3, 4, 5])
        assert tensor.shape == (5,)
        assert str(tensor.dtype).endswith('int32') or str(tensor.dtype).endswith('int64')
        
        # Create a tensor from a tensor
        tensor2d = ops.reshape(ops.convert_to_tensor([[1, 2, 3], [4, 5, 6]]), (2, 3))
        tensor = EmberTensor(tensor2d)
        assert tensor.shape == (2, 3)
        assert str(tensor.dtype).endswith('int32') or str(tensor.dtype).endswith('int64')
        
        # Create a tensor with a specific dtype
        tensor = EmberTensor([1, 2, 3, 4, 5], dtype='float32')
        assert tensor.shape == (5,)
        assert str(tensor.dtype).endswith('float32')

    def test_static_methods(self):
        """Test EmberTensor static methods."""
        # Test zeros
        zeros = EmberTensor.zeros((3, 3))
        assert zeros.shape == (3, 3)
        assert ops.all(ops.equal(zeros.data, 0))
        
        # Test ones
        ones = EmberTensor.ones((3, 3))
        assert ones.shape == (3, 3)
        assert ops.all(ops.equal(ones.data, 1))
        
        # Test full
        full = EmberTensor.full((3, 3), 5)
        assert full.shape == (3, 3)
        assert ops.all(ops.equal(full.data, 5))
        
        # Test arange
        arange = EmberTensor.arange(0, 10, 2)
        assert arange.shape == (5,)
        expected = ops.convert_to_tensor([0, 2, 4, 6, 8])
        assert ops.all(ops.equal(arange.data, expected))
        
        # Test linspace
        linspace = EmberTensor.linspace(0, 1, 5)
        assert linspace.shape == (5,)
        expected = ops.convert_to_tensor([0, 0.25, 0.5, 0.75, 1])
        assert ops.all(ops.less(ops.abs(ops.subtract(linspace.data, expected)), 1e-5))
        
        # Test eye
        eye = EmberTensor.eye(3)
        assert eye.shape == (3, 3)
        # Check diagonal is 1
        for i in range(3):
            assert ops.equal(eye.data[i, i], 1)
        # Check upper triangle is 0
        for i in range(3):
            for j in range(3):
                if ops.greater(j, i):  # Upper triangle
                    assert ops.equal(eye.data[i, j], 0)
        # Check lower triangle is 0
        for i in range(3):
            for j in range(3):
                if ops.greater(i, j):  # Lower triangle
                    assert ops.equal(eye.data[i, j], 0)
        
        # Test random_normal
        random_normal = EmberTensor.random_normal((3, 3))
        assert random_normal.shape == (3, 3)
        
        # Test random_uniform
        random_uniform = EmberTensor.random_uniform((3, 3))
        assert random_uniform.shape == (3, 3)
        
        # Test zeros_like
        tensor = EmberTensor(ops.convert_to_tensor([[1, 2, 3], [4, 5, 6]]))
        zeros_like = EmberTensor.zeros_like(tensor)
        assert zeros_like.shape == (2, 3)
        assert ops.all(ops.equal(zeros_like.data, 0))
        
        # Test ones_like
        ones_like = EmberTensor.ones_like(tensor)
        assert ones_like.shape == (2, 3)
        assert ops.all(ops.equal(ones_like.data, 1))

    def test_properties(self):
        """Test EmberTensor properties."""
        tensor = EmberTensor(ops.convert_to_tensor([[1, 2, 3], [4, 5, 6]]))
        
        # Test shape
        assert tensor.shape == (2, 3)
        
        # Test dtype
        assert str(tensor.dtype).endswith('int32') or str(tensor.dtype).endswith('int64')
        
        # Test data
        assert tensor.data is not None
        
        # Test backend
        assert tensor.backend == 'numpy'

    def test_conversion(self):
        """Test EmberTensor conversion methods."""
        tensor = EmberTensor(ops.convert_to_tensor([[1, 2, 3], [4, 5, 6]]))
        
        # Test to_numpy
        numpy_data = ops.to_numpy(tensor.data)
        assert numpy_data.shape == (2, 3)
        
        # Test to
        float_tensor = tensor.to(dtype='float32')
        assert str(float_tensor.dtype).endswith('float32')
        assert float_tensor.shape == (2, 3)
        
        # Check values
        for i in range(2):
            for j in range(3):
                expected_value = ops.add(ops.add(ops.multiply(i, 3), j), 1)
                assert ops.equal(float_tensor.data[i, j], expected_value)

    def test_arithmetic_operations(self):
        """Test EmberTensor arithmetic operations."""
        tensor1 = EmberTensor([1, 2, 3, 4, 5])
        tensor2 = EmberTensor([5, 4, 3, 2, 1])
        
        # Test add
        result = ops.add(tensor1.data, tensor2.data)
        result = EmberTensor(result)
        assert isinstance(result, EmberTensor)
        assert result.shape == (5,)
        assert ops.all(ops.equal(result.data, ops.convert_to_tensor([6, 6, 6, 6, 6])))
        
        # Test subtract
        result = ops.subtract(tensor1.data, tensor2.data)
        result = EmberTensor(result)
        assert isinstance(result, EmberTensor)
        assert result.shape == (5,)
        assert ops.all(ops.equal(result.data, ops.convert_to_tensor([-4, -2, 0, 2, 4])))
        
        # Test multiply
        result = ops.multiply(tensor1.data, tensor2.data)
        result = EmberTensor(result)
        assert isinstance(result, EmberTensor)
        assert result.shape == (5,)
        assert ops.all(ops.equal(result.data, ops.convert_to_tensor([5, 8, 9, 8, 5])))
        
        # Test divide
        result = ops.divide(tensor1.data, tensor2.data)
        result = EmberTensor(result)
        assert isinstance(result, EmberTensor)
        assert result.shape == (5,)
        expected = ops.convert_to_tensor([0.2, 0.5, 1.0, 2.0, 5.0])
        assert ops.all(ops.less(ops.abs(ops.subtract(result.data, expected)), 1e-5))
        
        # Test negative (using multiply by -1 instead of unary minus)
        result = ops.multiply(tensor1.data, -1)
        result = EmberTensor(result)
        assert isinstance(result, EmberTensor)
        assert result.shape == (5,)
        assert ops.all(ops.equal(result.data, ops.convert_to_tensor([-1, -2, -3, -4, -5])))
        
        # Test absolute
        result = ops.abs(ops.multiply(tensor1.data, -1))
        result = EmberTensor(result)
        assert isinstance(result, EmberTensor)
        assert result.shape == (5,)
        assert ops.all(ops.equal(result.data, ops.convert_to_tensor([1, 2, 3, 4, 5])))

    def test_comparison_operations(self):
        """Test EmberTensor comparison operations."""
        tensor1 = EmberTensor(ops.convert_to_tensor([1, 2, 3, 4, 5]))
        tensor2 = EmberTensor(ops.convert_to_tensor([5, 4, 3, 2, 1]))
        
        # Test equal
        result = ops.equal(tensor1.data, tensor2.data)
        result_tensor = EmberTensor(result)
        assert isinstance(result_tensor, EmberTensor)
        assert result_tensor.shape == (5,)
        expected = ops.convert_to_tensor([False, False, True, False, False])
        assert ops.all(ops.equal(result, expected))
        
        # Test not equal
        result = ops.not_equal(tensor1.data, tensor2.data)
        result_tensor = EmberTensor(result)
        assert isinstance(result_tensor, EmberTensor)
        assert result_tensor.shape == (5,)
        expected = ops.convert_to_tensor([True, True, False, True, True])
        assert ops.all(ops.equal(result, expected))
        
        # Test less than
        result = ops.less(tensor1.data, tensor2.data)
        result_tensor = EmberTensor(result)
        assert isinstance(result_tensor, EmberTensor)
        assert result_tensor.shape == (5,)
        expected = ops.convert_to_tensor([True, True, False, False, False])
        assert ops.all(ops.equal(result, expected))
        
        # Test less than or equal
        result = ops.less_equal(tensor1.data, tensor2.data)
        result_tensor = EmberTensor(result)
        assert isinstance(result_tensor, EmberTensor)
        assert result_tensor.shape == (5,)
        expected = ops.convert_to_tensor([True, True, True, False, False])
        assert ops.all(ops.equal(result, expected))
        
        # Test greater than
        result = ops.greater(tensor1.data, tensor2.data)
        result_tensor = EmberTensor(result)
        assert isinstance(result_tensor, EmberTensor)
        assert result_tensor.shape == (5,)
        expected = ops.convert_to_tensor([False, False, False, True, True])
        assert ops.all(ops.equal(result, expected))
        
        # Test greater than or equal
        result = ops.greater_equal(tensor1.data, tensor2.data)
        result_tensor = EmberTensor(result)
        assert isinstance(result_tensor, EmberTensor)
        assert result_tensor.shape == (5,)
        expected = ops.convert_to_tensor([False, False, True, True, True])
        assert ops.all(ops.equal(result, expected))

    def test_shape_operations(self):
        """Test EmberTensor shape operations."""
        tensor = EmberTensor(ops.convert_to_tensor([[1, 2, 3], [4, 5, 6]]))
        
        # Test reshape
        reshaped = ops.reshape(tensor.data, (3, 2))
        reshaped_tensor = EmberTensor(reshaped)
        assert isinstance(reshaped_tensor, EmberTensor)
        assert reshaped_tensor.shape == (3, 2)
        expected = ops.convert_to_tensor([[1, 2], [3, 4], [5, 6]])
        assert ops.all(ops.equal(reshaped, expected))
        
        # Test transpose
        transposed = ops.transpose(tensor.data)
        transposed_tensor = EmberTensor(transposed)
        assert isinstance(transposed_tensor, EmberTensor)
        assert transposed_tensor.shape == (3, 2)
        expected = ops.convert_to_tensor([[1, 4], [2, 5], [3, 6]])
        assert ops.all(ops.equal(transposed, expected))
        
        # Test squeeze
        tensor3d = EmberTensor(ops.reshape(ops.convert_to_tensor([1, 2, 3]), (1, 1, 3)))
        squeezed = ops.squeeze(tensor3d.data)
        squeezed_tensor = EmberTensor(squeezed)
        assert isinstance(squeezed_tensor, EmberTensor)
        assert len(squeezed_tensor.shape) == 1
        assert squeezed_tensor.shape[0] == 3
        expected = ops.convert_to_tensor([1, 2, 3])
        assert ops.all(ops.equal(squeezed, expected))
        
        # Test unsqueeze (expand_dims)
        unsqueezed = ops.expand_dims(tensor.data, 0)
        unsqueezed_tensor = EmberTensor(unsqueezed)
        assert isinstance(unsqueezed_tensor, EmberTensor)
        assert unsqueezed_tensor.shape == (1, 2, 3)
        expected_shape = (1, 2, 3)
        assert unsqueezed_tensor.shape == expected_shape

    def test_reduction_operations(self):
        """Test EmberTensor reduction operations."""
        tensor = EmberTensor(ops.convert_to_tensor([[1, 2, 3], [4, 5, 6]]))
        
        # Test sum
        sum_result = ops.sum(tensor.data)
        sum_result_tensor = EmberTensor(sum_result)
        assert isinstance(sum_result_tensor, EmberTensor)
        assert ops.equal(sum_result, 21)
        
        # Test sum along axis
        sum_axis0 = ops.sum(tensor.data, axis=0)
        sum_axis0_tensor = EmberTensor(sum_axis0)
        assert isinstance(sum_axis0_tensor, EmberTensor)
        assert sum_axis0_tensor.shape == (3,)
        expected = ops.convert_to_tensor([5, 7, 9])
        assert ops.all(ops.equal(sum_axis0, expected))
        
        sum_axis1 = ops.sum(tensor.data, axis=1)
        sum_axis1_tensor = EmberTensor(sum_axis1)
        assert isinstance(sum_axis1_tensor, EmberTensor)
        assert sum_axis1_tensor.shape == (2,)
        expected = ops.convert_to_tensor([6, 15])
        assert ops.all(ops.equal(sum_axis1, expected))
        
        # Test mean
        mean_result = ops.mean(tensor.data)
        mean_result_tensor = EmberTensor(mean_result)
        assert isinstance(mean_result_tensor, EmberTensor)
        assert ops.equal(mean_result, 3.5)
        
        # Test mean along axis
        mean_axis0 = ops.mean(tensor.data, axis=0)
        mean_axis0_tensor = EmberTensor(mean_axis0)
        assert isinstance(mean_axis0_tensor, EmberTensor)
        assert mean_axis0_tensor.shape == (3,)
        expected = ops.convert_to_tensor([2.5, 3.5, 4.5])
        assert ops.all(ops.less(ops.abs(ops.subtract(mean_axis0, expected)), 1e-5))
        
        mean_axis1 = ops.mean(tensor.data, axis=1)
        mean_axis1_tensor = EmberTensor(mean_axis1)
        assert isinstance(mean_axis1_tensor, EmberTensor)
        assert mean_axis1_tensor.shape == (2,)
        expected = ops.convert_to_tensor([2.0, 5.0])
        assert ops.all(ops.less(ops.abs(ops.subtract(mean_axis1, expected)), 1e-5))
        
        # Test max
        max_result = ops.max(tensor.data)
        max_result_tensor = EmberTensor(max_result)
        assert isinstance(max_result_tensor, EmberTensor)
        assert ops.equal(max_result, 6)
        
        # Test max along axis
        max_axis0 = ops.max(tensor.data, axis=0)
        max_axis0_tensor = EmberTensor(max_axis0)
        assert isinstance(max_axis0_tensor, EmberTensor)
        assert max_axis0_tensor.shape == (3,)
        expected = ops.convert_to_tensor([4, 5, 6])
        assert ops.all(ops.equal(max_axis0, expected))
        
        max_axis1 = ops.max(tensor.data, axis=1)
        max_axis1_tensor = EmberTensor(max_axis1)
        assert isinstance(max_axis1_tensor, EmberTensor)
        assert max_axis1_tensor.shape == (2,)
        expected = ops.convert_to_tensor([3, 6])
        assert ops.all(ops.equal(max_axis1, expected))
        
        # Test min
        min_result = ops.min(tensor.data)
        min_result_tensor = EmberTensor(min_result)
        assert isinstance(min_result_tensor, EmberTensor)
        assert ops.equal(min_result, 1)
        
        # Test min along axis
        min_axis0 = ops.min(tensor.data, axis=0)
        min_axis0_tensor = EmberTensor(min_axis0)
        assert isinstance(min_axis0_tensor, EmberTensor)
        assert min_axis0_tensor.shape == (3,)
        expected = ops.convert_to_tensor([1, 2, 3])
        assert ops.all(ops.equal(min_axis0, expected))
        
        min_axis1 = ops.min(tensor.data, axis=1)
        min_axis1_tensor = EmberTensor(min_axis1)
        assert isinstance(min_axis1_tensor, EmberTensor)
        assert min_axis1_tensor.shape == (2,)
        expected = ops.convert_to_tensor([1, 4])
        assert ops.all(ops.equal(min_axis1, expected))

    def test_conversion_to_ops_tensor(self):
        """Test conversion of EmberTensor to ops tensor."""
        # Create an EmberTensor
        ember_tensor = EmberTensor([1, 2, 3, 4, 5])
        
        # Convert to ops tensor
        ops_tensor = ops.convert_to_tensor(ember_tensor.data)
        
        # Check that the conversion was successful
        assert ops_tensor is not None
        assert ops.shape(ops_tensor) == (5,)
        expected = ops.convert_to_tensor([1, 2, 3, 4, 5])
        assert ops.all(ops.equal(ops_tensor, expected))