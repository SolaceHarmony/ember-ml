"""
Test script for the ops module.

This script tests the basic functionality of the ops module using pytest.
"""

import numpy as np
import pytest

from ember_ml.ops import (
    get_ops, set_ops,
    tensor_ops, math_ops, device_ops, random_ops
)

# Setup and teardown
@pytest.fixture
def setup_numpy_backend():
    """Set up the NumPy backend for testing."""
    set_ops('numpy')
    yield
    # No teardown needed

# Tests for ops module functions
def test_get_ops():
    """Test get_ops function."""
    set_ops('numpy')
    assert get_ops() == 'numpy'

def test_set_ops():
    """Test set_ops function."""
    set_ops('numpy')
    assert get_ops() == 'numpy'
    
    # Test invalid ops name
    with pytest.raises(ValueError):
        set_ops('invalid')

# Tests for tensor operations
class TestTensorOps:
    """Test case for tensor operations."""
    
    def test_zeros(self, setup_numpy_backend):
        """Test zeros function."""
        t_ops = tensor_ops()
        zeros = t_ops.zeros((2, 3))
        assert zeros.shape == (2, 3)
        assert np.all(zeros == 0)
    
    def test_ones(self, setup_numpy_backend):
        """Test ones function."""
        t_ops = tensor_ops()
        ones = t_ops.ones((2, 3))
        assert ones.shape == (2, 3)
        assert np.all(ones == 1)
    
    def test_reshape(self, setup_numpy_backend):
        """Test reshape function."""
        t_ops = tensor_ops()
        x = t_ops.ones((6,))
        y = t_ops.reshape(x, (2, 3))
        assert y.shape == (2, 3)
    
    def test_convert_to_tensor(self, setup_numpy_backend):
        """Test convert_to_tensor function."""
        t_ops = tensor_ops()
        x = t_ops.convert_to_tensor([1, 2, 3])
        assert x.shape == (3,)
        assert np.all(x == np.array([1, 2, 3]))

# Tests for math operations
class TestMathOps:
    """Test case for math operations."""
    
    def test_add(self, setup_numpy_backend):
        """Test add function."""
        m_ops = math_ops()
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        z = m_ops.add(x, y)
        assert np.all(z == np.array([5, 7, 9]))
    
    def test_subtract(self, setup_numpy_backend):
        """Test subtract function."""
        m_ops = math_ops()
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        z = m_ops.subtract(x, y)
        assert np.all(z == np.array([-3, -3, -3]))
    
    def test_multiply(self, setup_numpy_backend):
        """Test multiply function."""
        m_ops = math_ops()
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        z = m_ops.multiply(x, y)
        assert np.all(z == np.array([4, 10, 18]))
    
    def test_divide(self, setup_numpy_backend):
        """Test divide function."""
        m_ops = math_ops()
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        z = m_ops.divide(x, y)
        assert np.allclose(z, np.array([0.25, 0.4, 0.5]))
    
    def test_exp(self, setup_numpy_backend):
        """Test exp function."""
        m_ops = math_ops()
        x = np.array([1, 2, 3])
        z = m_ops.exp(x)
        assert np.allclose(z, np.exp(x))
    
    def test_log(self, setup_numpy_backend):
        """Test log function."""
        m_ops = math_ops()
        x = np.array([1, 2, 3])
        z = m_ops.log(x)
        assert np.allclose(z, np.log(x))

    def test_log10(self, setup_numpy_backend):
        """Test log10 function."""
        m_ops = math_ops()
        x = np.array([1, 10, 100])
        z = m_ops.log10(x)
        assert np.allclose(z, np.log10(x))

    def test_log2(self, setup_numpy_backend):
        """Test log2 function."""
        m_ops = math_ops()
        x = np.array([1, 2, 4])
        z = m_ops.log2(x)
        assert np.allclose(z, np.log2(x))

    def test_pow(self, setup_numpy_backend):
        """Test pow function."""
        m_ops = math_ops()
        x = np.array([1, 2, 3])
        y = np.array([2, 3, 4])
        z = m_ops.pow(x, y)
        assert np.allclose(z, np.power(x, y))

    def test_sqrt(self, setup_numpy_backend):
        """Test sqrt function."""
        m_ops = math_ops()
        x = np.array([1, 4, 9])
        z = m_ops.sqrt(x)
        assert np.allclose(z, np.sqrt(x))

    def test_square(self, setup_numpy_backend):
        """Test square function."""
        m_ops = math_ops()
        x = np.array([1, 2, 3])
        z = m_ops.square(x)
        assert np.allclose(z, np.square(x))

    def test_abs(self, setup_numpy_backend):
        """Test abs function."""
        m_ops = math_ops()
        x = np.array([-1, -2, -3])
        z = m_ops.abs(x)
        assert np.allclose(z, np.abs(x))

    def test_negative(self, setup_numpy_backend):
        """Test negative function."""
        m_ops = math_ops()
        x = np.array([1, 2, 3])
        z = m_ops.negative(x)
        assert np.allclose(z, np.negative(x))

    def test_sign(self, setup_numpy_backend):
        """Test sign function."""
        m_ops = math_ops()
        x = np.array([-1, 0, 1])
        z = m_ops.sign(x)
        assert np.allclose(z, np.sign(x))

    def test_clip(self, setup_numpy_backend):
        """Test clip function."""
        m_ops = math_ops()
        x = np.array([1, 2, 3, 4, 5])
        z = m_ops.clip(x, 2, 4)
        assert np.allclose(z, np.clip(x, 2, 4))

    def test_sin(self, setup_numpy_backend):
        """Test sin function."""
        m_ops = math_ops()
        x = np.array([0, np.pi / 2, np.pi])
        z = m_ops.sin(x)
        assert np.allclose(z, np.sin(x))

    def test_cos(self, setup_numpy_backend):
        """Test cos function."""
        m_ops = math_ops()
        x = np.array([0, np.pi / 2, np.pi])
        z = m_ops.cos(x)
        assert np.allclose(z, np.cos(x))

    def test_tan(self, setup_numpy_backend):
        """Test tan function."""
        m_ops = math_ops()
        x = np.array([0, np.pi / 4, np.pi / 2])
        z = m_ops.tan(x)
        assert np.allclose(z, np.tan(x))

    def test_sinh(self, setup_numpy_backend):
        """Test sinh function."""
        m_ops = math_ops()
        x = np.array([0, 1, 2])
        z = m_ops.sinh(x)
        assert np.allclose(z, np.sinh(x))

    def test_cosh(self, setup_numpy_backend):
        """Test cosh function."""
        m_ops = math_ops()
        x = np.array([0, 1, 2])
        z = m_ops.cosh(x)
        assert np.allclose(z, np.cosh(x))

    def test_tanh(self, setup_numpy_backend):
        """Test tanh function."""
        m_ops = math_ops()
        x = np.array([0, 1, 2])
        z = m_ops.tanh(x)
        assert np.allclose(z, np.tanh(x))

    def test_sigmoid(self, setup_numpy_backend):
        """Test sigmoid function."""
        m_ops = math_ops()
        x = np.array([0, 1, 2])
        z = m_ops.sigmoid(x)
        assert np.allclose(z, 1 / (1 + np.exp(-x)))

    def test_softplus(self, setup_numpy_backend):
        """Test softplus function."""
        m_ops = math_ops()
        x = np.array([0, 1, 2])
        z = m_ops.softplus(x)
        assert np.allclose(z, np.log(1 + np.exp(x)))

    def test_relu(self, setup_numpy_backend):
        """Test relu function."""
        m_ops = math_ops()
        x = np.array([-1, 0, 1])
        z = m_ops.relu(x)
        assert np.allclose(z, np.maximum(0, x))

    def test_softmax(self, setup_numpy_backend):
        """Test softmax function."""
        m_ops = math_ops()
        x = np.array([1, 2, 3])
        z = m_ops.softmax(x)
        exp_x = np.exp(x - np.max(x))
        expected = exp_x / np.sum(exp_x)
        assert np.allclose(z, expected)

    def test_gradient(self, setup_numpy_backend):
        """Test gradient function."""
        m_ops = math_ops()
        x = np.array([1, 2, 4, 7, 11])
        z = m_ops.gradient(x)
        expected = np.gradient(x)
        assert np.allclose(z, expected)

    def test_cumsum(self, setup_numpy_backend):
        """Test cumsum function."""
        m_ops = math_ops()
        x = np.array([1, 2, 3])
        z = m_ops.cumsum(x)
        assert np.allclose(z, np.cumsum(x))

    def test_eigh(self, setup_numpy_backend):
        """Test eigh function."""
        m_ops = math_ops()
        x = np.array([[1, 2], [2, 3]])
        w, v = m_ops.eigh(x)
        expected_w, expected_v = np.linalg.eigh(x)
        assert np.allclose(w, expected_w)
        assert np.allclose(v, expected_v)

    def test_mod(self, setup_numpy_backend):
        """Test mod function."""
        m_ops = math_ops()
        x = np.array([1, 2, 3])
        y = np.array([2, 2, 2])
        z = m_ops.mod(x, y)
        assert np.allclose(z, np.mod(x, y))

    def test_floor_divide(self, setup_numpy_backend):
        """Test floor_divide function."""
        m_ops = math_ops()
        x = np.array([1, 2, 3])
        y = np.array([2, 2, 2])
        z = m_ops.floor_divide(x, y)
        assert np.allclose(z, np.floor_divide(x, y))

    def test_sort(self, setup_numpy_backend):
        """Test sort function."""
        m_ops = math_ops()
        x = np.array([3, 1, 2])
        z = m_ops.sort(x)
        assert np.allclose(z, np.sort(x))

# Tests for device operations
class TestDeviceOps:
    """Test case for device operations."""
    
    def test_get_device(self, setup_numpy_backend):
        """Test get_device function."""
        d_ops = device_ops()
        x = np.array([1, 2, 3])
        assert d_ops.get_device(x) == 'cpu'
    
    def test_to_device(self, setup_numpy_backend):
        """Test to_device function."""
        d_ops = device_ops()
        x = np.array([1, 2, 3])
        y = d_ops.to_device(x, 'cpu')
        assert np.all(y == x)
    
    def test_set_default_device(self, setup_numpy_backend):
        """Test set_default_device function."""
        d_ops = device_ops()
        d_ops.set_default_device('cpu')
        assert d_ops.get_default_device() == 'cpu'
    
    def test_is_available(self, setup_numpy_backend):
        """Test is_available function."""
        d_ops = device_ops()
        assert d_ops.is_available('cpu')
        assert not d_ops.is_available('gpu')

# Tests for random operations
class TestRandomOps:
    """Test case for random operations."""
    
    def test_set_seed(self, setup_numpy_backend):
        """Test set_seed function."""
        r_ops = random_ops()
        r_ops.set_seed(42)
        assert r_ops.get_seed() == 42
    
    def test_random_normal(self, setup_numpy_backend):
        """Test random_normal function."""
        r_ops = random_ops()
        x = r_ops.random_normal((2, 3))
        assert x.shape == (2, 3)
    
    def test_random_uniform(self, setup_numpy_backend):
        """Test random_uniform function."""
        r_ops = random_ops()
        x = r_ops.random_uniform((2, 3))
        assert x.shape == (2, 3)
        assert np.all(x >= 0) and np.all(x < 1)
    
    def test_shuffle(self, setup_numpy_backend):
        """Test shuffle function."""
        r_ops = random_ops()
        x = np.array([1, 2, 3, 4, 5])
        y = r_ops.shuffle(x)
        assert y.shape == (5,)
        assert set(y.tolist()) == set(x.tolist())
