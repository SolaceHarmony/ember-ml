import pytest
from ember_ml import ops
from ember_ml.nn import tensor

# Assume conftest.py provides 'backend' fixture

@pytest.fixture
def comparison_tensors(backend):
    """Fixture to create sample tensors for comparison testing."""
    ops.set_backend(backend)
    t1 = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    t2 = tensor.convert_to_tensor([[1.0, 0.0], [5.0, 4.0]])
    t_equal = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]]) # Equal to t1
    scalar = tensor.convert_to_tensor(3.0)
    return t1, t2, t_equal, scalar

def test_ops_equal(comparison_tensors, backend):
    """Tests ops.equal."""
    ops.set_backend(backend)
    t1, t2, t_equal, scalar = comparison_tensors
    
    result_t1_t_equal = ops.equal(t1, t_equal)
    expected_t1_t_equal = tensor.convert_to_tensor([[True, True], [True, True]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_t1_t_equal, expected_t1_t_equal)), f"{backend}: Equal (T==T) failed"

    result_t1_t2 = ops.equal(t1, t2)
    expected_t1_t2 = tensor.convert_to_tensor([[True, False], [False, True]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_t1_t2, expected_t1_t2)), f"{backend}: Equal (T!=T) failed"
    
    result_t1_scalar = ops.equal(t1, scalar)
    expected_t1_scalar = tensor.convert_to_tensor([[False, False], [True, False]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_t1_scalar, expected_t1_scalar)), f"{backend}: Equal (T==S) failed"

def test_ops_not_equal(comparison_tensors, backend):
    """Tests ops.not_equal."""
    ops.set_backend(backend)
    t1, t2, t_equal, scalar = comparison_tensors

    result_t1_t_equal = ops.not_equal(t1, t_equal)
    expected_t1_t_equal = tensor.convert_to_tensor([[False, False], [False, False]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_t1_t_equal, expected_t1_t_equal)), f"{backend}: Not Equal (T==T) failed"

    result_t1_t2 = ops.not_equal(t1, t2)
    expected_t1_t2 = tensor.convert_to_tensor([[False, True], [True, False]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_t1_t2, expected_t1_t2)), f"{backend}: Not Equal (T!=T) failed"

    result_t1_scalar = ops.not_equal(t1, scalar)
    expected_t1_scalar = tensor.convert_to_tensor([[True, True], [False, True]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_t1_scalar, expected_t1_scalar)), f"{backend}: Not Equal (T==S) failed"


def test_ops_less(comparison_tensors, backend):
    """Tests ops.less."""
    ops.set_backend(backend)
    t1, t2, _, scalar = comparison_tensors

    result_t1_t2 = ops.less(t1, t2)
    expected_t1_t2 = tensor.convert_to_tensor([[False, False], [True, False]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_t1_t2, expected_t1_t2)), f"{backend}: Less (T<T) failed"

    result_t1_scalar = ops.less(t1, scalar)
    expected_t1_scalar = tensor.convert_to_tensor([[True, True], [False, False]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_t1_scalar, expected_t1_scalar)), f"{backend}: Less (T<S) failed"

def test_ops_less_equal(comparison_tensors, backend):
    """Tests ops.less_equal."""
    ops.set_backend(backend)
    t1, t2, t_equal, scalar = comparison_tensors
    
    result_t1_t_equal = ops.less_equal(t1, t_equal)
    expected_t1_t_equal = tensor.convert_to_tensor([[True, True], [True, True]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_t1_t_equal, expected_t1_t_equal)), f"{backend}: Less Equal (T<=T) failed"

    result_t1_t2 = ops.less_equal(t1, t2)
    expected_t1_t2 = tensor.convert_to_tensor([[True, False], [True, True]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_t1_t2, expected_t1_t2)), f"{backend}: Less Equal (T<=T) failed"

    result_t1_scalar = ops.less_equal(t1, scalar)
    expected_t1_scalar = tensor.convert_to_tensor([[True, True], [True, False]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_t1_scalar, expected_t1_scalar)), f"{backend}: Less Equal (T<=S) failed"

def test_ops_greater(comparison_tensors, backend):
    """Tests ops.greater."""
    ops.set_backend(backend)
    t1, t2, _, scalar = comparison_tensors

    result_t1_t2 = ops.greater(t1, t2)
    expected_t1_t2 = tensor.convert_to_tensor([[False, True], [False, False]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_t1_t2, expected_t1_t2)), f"{backend}: Greater (T>T) failed"

    result_t1_scalar = ops.greater(t1, scalar)
    expected_t1_scalar = tensor.convert_to_tensor([[False, False], [False, True]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_t1_scalar, expected_t1_scalar)), f"{backend}: Greater (T>S) failed"

def test_ops_greater_equal(comparison_tensors, backend):
    """Tests ops.greater_equal."""
    ops.set_backend(backend)
    t1, t2, t_equal, scalar = comparison_tensors

    result_t1_t_equal = ops.greater_equal(t1, t_equal)
    expected_t1_t_equal = tensor.convert_to_tensor([[True, True], [True, True]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_t1_t_equal, expected_t1_t_equal)), f"{backend}: Greater Equal (T>=T) failed"

    result_t1_t2 = ops.greater_equal(t1, t2)
    expected_t1_t2 = tensor.convert_to_tensor([[True, True], [False, True]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_t1_t2, expected_t1_t2)), f"{backend}: Greater Equal (T>=T) failed"

    result_t1_scalar = ops.greater_equal(t1, scalar)
    expected_t1_scalar = tensor.convert_to_tensor([[False, False], [True, True]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_t1_scalar, expected_t1_scalar)), f"{backend}: Greater Equal (T>=S) failed"

def test_ops_allclose(backend):
    """Tests ops.allclose."""
    ops.set_backend(backend)
    t1 = tensor.convert_to_tensor([1.0, 2.0, 3.0])
    t2 = tensor.convert_to_tensor([1.000001, 2.000002, 3.000003])
    t3 = tensor.convert_to_tensor([1.1, 2.1, 3.1])

    assert ops.allclose(t1, t1), f"{backend}: allclose self failed"
    assert ops.allclose(t1, t2), f"{backend}: allclose slightly different failed"
    assert not ops.allclose(t1, t3), f"{backend}: allclose significantly different failed"
    assert ops.allclose(t1, t3, atol=0.2), f"{backend}: allclose with higher tolerance failed"

def test_ops_isclose(backend):
    """Tests ops.isclose."""
    ops.set_backend(backend)
    t1 = tensor.convert_to_tensor([1.0, 2.0, 3.0])
    t2 = tensor.convert_to_tensor([1.000001, 2.1, 3.000003]) # Mixed closeness
    
    result = ops.isclose(t1, t2)
    expected = tensor.convert_to_tensor([True, False, True], dtype=tensor.bool_)
    assert ops.all(ops.equal(result, expected)), f"{backend}: isclose basic failed"

def test_ops_logical_ops(backend):
    """Tests logical_and, logical_or, logical_not, logical_xor."""
    ops.set_backend(backend)
    t_bool1 = tensor.convert_to_tensor([[True, True], [False, False]], dtype=tensor.bool_)
    t_bool2 = tensor.convert_to_tensor([[True, False], [True, False]], dtype=tensor.bool_)

    # Logical AND
    result_and = ops.logical_and(t_bool1, t_bool2)
    expected_and = tensor.convert_to_tensor([[True, False], [False, False]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_and, expected_and)), f"{backend}: Logical AND failed"

    # Logical OR
    result_or = ops.logical_or(t_bool1, t_bool2)
    expected_or = tensor.convert_to_tensor([[True, True], [True, False]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_or, expected_or)), f"{backend}: Logical OR failed"

    # Logical NOT
    result_not = ops.logical_not(t_bool1)
    expected_not = tensor.convert_to_tensor([[False, False], [True, True]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_not, expected_not)), f"{backend}: Logical NOT failed"

    # Logical XOR
    result_xor = ops.logical_xor(t_bool1, t_bool2)
    expected_xor = tensor.convert_to_tensor([[False, True], [True, False]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_xor, expected_xor)), f"{backend}: Logical XOR failed"

def test_ops_all(backend):
    """Tests ops.all."""
    ops.set_backend(backend)
    t_true = tensor.convert_to_tensor([[True, True], [True, True]], dtype=tensor.bool_)
    t_mixed = tensor.convert_to_tensor([[True, False], [True, True]], dtype=tensor.bool_)
    t_false = tensor.convert_to_tensor([[False, False], [False, False]], dtype=tensor.bool_)

    # All True
    assert tensor.item(ops.all(t_true)) is True, f"{backend}: ops.all (all true) failed"

    # Mixed
    assert tensor.item(ops.all(t_mixed)) is False, f"{backend}: ops.all (mixed) failed"

    # All False
    assert tensor.item(ops.all(t_false)) is False, f"{backend}: ops.all (all false) failed"

    # Test with axis=0
    result_axis0 = ops.all(t_mixed, axis=0)
    expected_axis0 = tensor.convert_to_tensor([True, False], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_axis0, expected_axis0)), f"{backend}: ops.all (axis=0) failed"

    # Test with axis=1
    result_axis1 = ops.all(t_mixed, axis=1)
    expected_axis1 = tensor.convert_to_tensor([False, True], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_axis1, expected_axis1)), f"{backend}: ops.all (axis=1) failed"

def test_ops_where(backend):
    """Tests ops.where."""
    ops.set_backend(backend)
    condition = tensor.convert_to_tensor([[True, False], [False, True]], dtype=tensor.bool_)
    x = tensor.convert_to_tensor([[1, 2], [3, 4]])
    y = tensor.convert_to_tensor([[10, 20], [30, 40]])

    result = ops.where(condition, x, y)
    expected = tensor.convert_to_tensor([[1, 20], [30, 4]]) # Selects from x where True, from y where False
    assert ops.allclose(result, expected), f"{backend}: ops.where failed"

def test_ops_isnan(backend):
    """Tests ops.isnan."""
    ops.set_backend(backend)
    # Create NaN using Ember operations (e.g., inf * 0, 0/0)
    # Note: Behavior might vary slightly across backends
    zero = tensor.convert_to_tensor(0.0)
    inf = tensor.convert_to_tensor(float('inf'))
    # nan_val = ops.divide(zero, zero) # 0/0 often results in NaN
    # Alternative: inf * 0 might be NaN in some backends
    nan_val = ops.multiply(inf, zero)

    # Create a tensor containing NaN and regular numbers
    # Need to construct carefully to place nan_val
    val1 = tensor.convert_to_tensor(1.0)
    val3 = tensor.convert_to_tensor(3.0)
    val4 = tensor.convert_to_tensor(4.0)
    # Create rows and stack them
    row1 = tensor.stack([val1, nan_val])
    row2 = tensor.stack([val3, val4])
    t_nan = tensor.stack([row1, row2])
    
    t_no_nan = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])

    # Test with NaNs
    result_nan = ops.isnan(t_nan)
    expected_nan = tensor.convert_to_tensor([[False, True], [False, False]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_nan, expected_nan)), f"{backend}: ops.isnan (with NaN) failed"

    # Test without NaNs
    result_no_nan = ops.isnan(t_no_nan)
    expected_no_nan = tensor.convert_to_tensor([[False, False], [False, False]], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_no_nan, expected_no_nan)), f"{backend}: ops.isnan (no NaN) failed"



    result_tol = ops.isclose(t1, t2, atol=0.2)
    expected_tol = tensor.convert_to_tensor([True, True, True], dtype=tensor.bool_)
    assert ops.all(ops.equal(result_tol, expected_tol)), f"{backend}: isclose with higher tolerance failed"

# TODO: Add tests for logical_and, logical_or, logical_not, logical_xor, all, where, isnan