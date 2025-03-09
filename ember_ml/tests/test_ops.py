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