import pytest
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn import modules # Import modules namespace for activations
from ember_ml.ops import stats # Import stats for dropout check

# Assume conftest.py provides 'backend' fixture

@pytest.fixture
def activation_input(backend):
    """Fixture for activation function tests."""
    ops.set_backend(backend)
    t = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    return t

# Test individual activation modules
def test_activation_relu(activation_input, backend):
    """Tests modules.ReLU activation."""
    ops.set_backend(backend)
    activation = modules.ReLU()
    output = activation(activation_input)
    expected = ops.relu(activation_input) # Compare against ops version
    assert ops.allclose(output, expected), f"{backend}: ReLU module failed"

def test_activation_tanh(activation_input, backend):
    """Tests modules.Tanh activation."""
    ops.set_backend(backend)
    activation = modules.Tanh()
    output = activation(activation_input)
    expected = ops.tanh(activation_input)
    assert ops.allclose(output, expected), f"{backend}: Tanh module failed"

def test_activation_sigmoid(activation_input, backend):
    """Tests modules.Sigmoid activation."""
    ops.set_backend(backend)
    activation = modules.Sigmoid()
    output = activation(activation_input)
    expected = ops.sigmoid(activation_input)
    assert ops.allclose(output, expected), f"{backend}: Sigmoid module failed"

def test_activation_softmax(backend):
    """Tests modules.Softmax activation."""
    ops.set_backend(backend)
    t_matrix = tensor.convert_to_tensor([[1.0, 2.0, 1.0], [-1.0, 0.0, 1.0]])
    # Default axis is -1
    activation = modules.Softmax() 
    output = activation(t_matrix)
    expected = ops.softmax(t_matrix, axis=-1) # Compare against ops version
    assert ops.allclose(output, expected), f"{backend}: Softmax module failed (axis=-1)"

    # Test different axis
    activation_ax0 = modules.Softmax(axis=0)
    output_ax0 = activation_ax0(t_matrix)
    expected_ax0 = ops.softmax(t_matrix, axis=0)
    assert ops.allclose(output_ax0, expected_ax0), f"{backend}: Softmax module failed (axis=0)"

def test_activation_softplus(activation_input, backend):
    """Tests modules.Softplus activation."""
    ops.set_backend(backend)
    activation = modules.Softplus()
    output = activation(activation_input)
    expected = ops.softplus(activation_input)
    assert ops.allclose(output, expected), f"{backend}: Softplus module failed"

def test_activation_lecun_tanh(activation_input, backend):
    """Tests modules.LeCunTanh activation."""
    ops.set_backend(backend)
    activation = modules.LeCunTanh()
    output = activation(activation_input)
    # Calculate expected using ops
    # Expected: 1.7159 * tanh(0.666 * x)
    expected = ops.multiply(1.7159, ops.tanh(ops.multiply(0.666, activation_input)))
    # Cast expected value to match output dtype for comparison if needed
    expected = tensor.cast(expected, output.dtype)
    assert ops.allclose(output, expected, atol=1e-6), f"{backend}: LeCunTanh module failed"

def test_activation_dropout(activation_input, backend):
    """Tests modules.Dropout activation."""
    ops.set_backend(backend)
    rate = 0.5
    activation = modules.Dropout(rate=rate)
    
    # Test in training mode (should drop some elements)
    tensor.set_seed(42) # For reproducibility if possible
    output_train = activation(activation_input, training=True)
    assert tensor.shape(output_train) == tensor.shape(activation_input), "Dropout training shape mismatch"
    
    # Check if some elements are zero (dropped) - might not always happen with small input
    # Use a larger input size for a more reliable check
    large_input = tensor.ones(1000)
    tensor.set_seed(43)
    large_output_train = activation(large_input, training=True)
    num_zeros = stats.sum(tensor.cast(ops.equal(large_output_train, 0.0), tensor.int32))
    assert tensor.item(num_zeros) > 0, "Dropout training did not zero out any elements on larger input"
    
    # Check scaling: non-zero elements should be scaled by 1/(1-rate)
    # Using the original small input for easier manual verification of scaling
    tensor.set_seed(44) # Use a different seed to potentially get non-zero outputs
    output_train_scaled = activation(activation_input, training=True)
    non_zero_mask = ops.not_equal(output_train_scaled, 0.0)
    
    # Handle case where all elements might be dropped by chance (unlikely but possible)
    if tensor.item(stats.sum(tensor.cast(non_zero_mask, tensor.int32))) > 0:
        input_non_zero = activation_input[non_zero_mask]
        output_non_zero = output_train_scaled[non_zero_mask]
        one_tensor = tensor.convert_to_tensor(1.0)
        rate_tensor = tensor.convert_to_tensor(rate)
        denominator = ops.subtract(one_tensor, rate_tensor)
        # Avoid division by zero if rate is 1.0
        # Although dropout with rate 1.0 is unlikely to leave non-zero elements, handle defensively.
        # Add a small epsilon or use safe divide if available, otherwise rely on backend handling.
        # For now, assume rate != 1.0 based on the check in line 103, or let the backend handle potential 0 division.
        scale_factor = ops.divide(one_tensor, denominator)
        expected_scaled = ops.multiply(input_non_zero, scale_factor)
        # Use a tolerance for float comparison
        assert ops.allclose(output_non_zero, expected_scaled, atol=1e-6), "Dropout scaling incorrect"
    else:
        # If all elements were dropped, the test technically passes regarding scaling (vacuously true)
        pass 

    # Test in eval mode (should be identity)
    output_eval = activation(activation_input, training=False)
    assert ops.allclose(output_eval, activation_input), "Dropout eval mode is not identity"