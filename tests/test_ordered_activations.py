"""Tests for the activation functions with ordered execution by backend."""

import pytest
from pytest import mark
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn import modules
from ember_ml.ops import stats

# Define the backend order: numpy -> torch -> mlx
# Each test with the same order number will run together

# NUMPY BACKEND TESTS - Run first
@mark.run(order=1)
def test_setup_numpy():
    """Set up the NumPy backend for all numpy tests."""
    print("\n=== Running tests with NumPy backend ===")
    ops.set_backend('numpy')
    assert ops.get_backend() == 'numpy'

@mark.run(order=1)
def test_activation_relu_numpy():
    """Tests modules.ReLU activation with NumPy backend."""
    # Backend should already be set by test_setup_numpy
    input_tensor = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    activation = modules.ReLU()
    output = activation(input_tensor)
    # Call the activation function via the nn.modules.activations module
    from ember_ml.nn.modules.activations import relu
    expected = relu(input_tensor)
    assert ops.allclose(output, expected), "ReLU module failed"

@mark.run(order=1)
def test_activation_sigmoid_numpy():
    """Tests modules.Sigmoid activation with NumPy backend."""
    input_tensor = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    activation = modules.Sigmoid()
    output = activation(input_tensor)
    from ember_ml.nn.modules.activations import sigmoid
    expected = sigmoid(input_tensor)
    assert ops.allclose(output, expected), "Sigmoid module failed"


# --- NumPy Tanh Test ---
@mark.run(order=1)
def test_activation_tanh_numpy():
    """Tests modules.Tanh activation with NumPy backend."""
    input_tensor = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    activation = modules.Tanh()
    output = activation(input_tensor)
    from ember_ml.nn.modules.activations import tanh
    expected = tanh(input_tensor)
    assert ops.allclose(output, expected), "Tanh module failed"

# --- NumPy Softmax Test ---
@mark.run(order=1)
def test_activation_softmax_numpy():
    """Tests modules.Softmax activation with NumPy backend."""
    t_matrix = tensor.convert_to_tensor([[1.0, 2.0, 1.0], [-1.0, 0.0, 1.0]])
    # Default axis is -1
    activation = modules.Softmax()
    output = activation(t_matrix)
    from ember_ml.nn.modules.activations import softmax
    expected = softmax(t_matrix, axis=-1)
    assert ops.allclose(output, expected), "Softmax module (axis=-1) failed"
    # Test with axis=0
    activation_ax0 = modules.Softmax(axis=0)
    output_ax0 = activation_ax0(t_matrix)
    expected_ax0 = softmax(t_matrix, axis=0)
    assert ops.allclose(output_ax0, expected_ax0), "Softmax module (axis=0) failed"

# --- NumPy Softplus Test ---
@mark.run(order=1)
def test_activation_softplus_numpy():
    """Tests modules.Softplus activation with NumPy backend."""
    input_tensor = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    activation = modules.Softplus()
    output = activation(input_tensor)
    from ember_ml.nn.modules.activations import softplus
    expected = softplus(input_tensor)
    assert ops.allclose(output, expected), "Softplus module failed"

# --- NumPy LeCunTanh Test ---
@mark.run(order=1)
def test_activation_lecun_tanh_numpy():
    """Tests modules.LeCunTanh activation with NumPy backend."""
    activation_input = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    activation = modules.LeCunTanh()
    output = activation(activation_input)
    # Calculate expected output manually
    scale_factor = tensor.convert_to_tensor(0.66666667)
    amplitude = tensor.convert_to_tensor(1.7159)
    from ember_ml.nn.modules.activations import tanh
    expected = ops.multiply(amplitude, tanh(ops.multiply(scale_factor, activation_input)))
    assert ops.allclose(output, expected), "LeCunTanh module failed"

# --- NumPy Dropout Test ---
@mark.run(order=1)
def test_activation_dropout_numpy():
    """Tests modules.Dropout activation with NumPy backend."""
    activation_input = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    rate = 0.5
    activation = modules.Dropout(rate=rate, seed=42)
    # Test in training mode
    output_train = activation(activation_input, training=True)
    assert tensor.shape(output_train) == tensor.shape(activation_input), "Dropout training shape mismatch"
    # Test in inference mode (should be identity)
    output_eval = activation(activation_input, training=False)
    assert ops.allclose(output_eval, activation_input), "Dropout evaluation failed"
    # Check non-determinism (or determinism with seed)
    output_train_2 = activation(activation_input, training=True) # Same seed
    assert ops.allclose(output_train, output_train_2), "Dropout not deterministic with seed"

# TORCH BACKEND TESTS - Run second
@mark.run(order=2)
def test_setup_torch():
    """Set up the PyTorch backend for all torch tests."""
    print("\n=== Running tests with PyTorch backend ===")
    ops.set_backend('torch')
    assert ops.get_backend() == 'torch'

@mark.run(order=2)
def test_activation_relu_torch():
    """Tests modules.ReLU activation with PyTorch backend."""
    # Backend should already be set by test_setup_torch
    input_tensor = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    activation = modules.ReLU()
    output = activation(input_tensor)
    # Call the activation function via the nn.modules.activations module
    from ember_ml.nn.modules.activations import relu
    expected = relu(input_tensor)
    assert ops.allclose(output, expected), "ReLU module failed"

@mark.run(order=2)
def test_activation_sigmoid_torch():
    """Tests modules.Sigmoid activation with PyTorch backend."""
    input_tensor = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    activation = modules.Sigmoid()
    output = activation(input_tensor)
    from ember_ml.nn.modules.activations import sigmoid
    expected = sigmoid(input_tensor)
    assert ops.allclose(output, expected), "Sigmoid module failed"


# --- PyTorch Tanh Test ---
@mark.run(order=2)
def test_activation_tanh_torch():
    """Tests modules.Tanh activation with PyTorch backend."""
    input_tensor = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    activation = modules.Tanh()
    output = activation(input_tensor)
    from ember_ml.nn.modules.activations import tanh
    expected = tanh(input_tensor)
    assert ops.allclose(output, expected), "Tanh module failed"

# --- PyTorch Softmax Test ---
@mark.run(order=2)
def test_activation_softmax_torch():
    """Tests modules.Softmax activation with PyTorch backend."""
    t_matrix = tensor.convert_to_tensor([[1.0, 2.0, 1.0], [-1.0, 0.0, 1.0]])
    activation = modules.Softmax()
    output = activation(t_matrix)
    from ember_ml.nn.modules.activations import softmax
    expected = softmax(t_matrix, axis=-1)
    assert ops.allclose(output, expected), "Softmax module (axis=-1) failed"
    activation_ax0 = modules.Softmax(axis=0)
    output_ax0 = activation_ax0(t_matrix)
    expected_ax0 = softmax(t_matrix, axis=0)
    assert ops.allclose(output_ax0, expected_ax0), "Softmax module (axis=0) failed"

# --- PyTorch Softplus Test ---
@mark.run(order=2)
def test_activation_softplus_torch():
    """Tests modules.Softplus activation with PyTorch backend."""
    input_tensor = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    activation = modules.Softplus()
    output = activation(input_tensor)
    from ember_ml.nn.modules.activations import softplus
    expected = softplus(input_tensor)
    assert ops.allclose(output, expected), "Softplus module failed"

# --- PyTorch LeCunTanh Test ---
@mark.run(order=2)
def test_activation_lecun_tanh_torch():
    """Tests modules.LeCunTanh activation with PyTorch backend."""
    activation_input = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    activation = modules.LeCunTanh()
    output = activation(activation_input)
    scale_factor = tensor.convert_to_tensor(0.66666667)
    amplitude = tensor.convert_to_tensor(1.7159)
    from ember_ml.nn.modules.activations import tanh
    expected = ops.multiply(amplitude, tanh(ops.multiply(scale_factor, activation_input)))
    assert ops.allclose(output, expected), "LeCunTanh module failed"

# --- PyTorch Dropout Test ---
@mark.run(order=2)
def test_activation_dropout_torch():
    """Tests modules.Dropout activation with PyTorch backend."""
    activation_input = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    rate = 0.5
    activation = modules.Dropout(rate=rate, seed=42)
    output_train = activation(activation_input, training=True)
    assert tensor.shape(output_train) == tensor.shape(activation_input), "Dropout training shape mismatch"
    output_eval = activation(activation_input, training=False)
    assert ops.allclose(output_eval, activation_input), "Dropout evaluation failed"
    output_train_2 = activation(activation_input, training=True)
    assert ops.allclose(output_train, output_train_2), "Dropout not deterministic with seed"

# MLX BACKEND TESTS - Run last
@mark.run(order=3)
def test_setup_mlx():
    """Set up the MLX backend for all mlx tests."""
    print("\n=== Running tests with MLX backend ===")
    ops.set_backend('mlx')
    assert ops.get_backend() == 'mlx'

@mark.run(order=3)
def test_activation_relu_mlx():
    """Tests modules.ReLU activation with MLX backend."""
    # Backend should already be set by test_setup_mlx
    input_tensor = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    activation = modules.ReLU()
    output = activation(input_tensor)
    # Call the activation function via the nn.modules.activations module
    from ember_ml.nn.modules.activations import relu
    expected = relu(input_tensor)
    assert ops.allclose(output, expected), "ReLU module failed"

@mark.run(order=3)
def test_activation_sigmoid_mlx():
    """Tests modules.Sigmoid activation with MLX backend."""
    input_tensor = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    activation = modules.Sigmoid()
    output = activation(input_tensor)
    from ember_ml.nn.modules.activations import sigmoid
    expected = sigmoid(input_tensor)
    assert ops.allclose(output, expected), "Sigmoid module failed"


# --- MLX Tanh Test ---
@mark.run(order=3)
def test_activation_tanh_mlx():
    """Tests modules.Tanh activation with MLX backend."""
    input_tensor = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    activation = modules.Tanh()
    output = activation(input_tensor)
    from ember_ml.nn.modules.activations import tanh
    expected = tanh(input_tensor)
    assert ops.allclose(output, expected), "Tanh module failed"

# --- MLX Softmax Test ---
@mark.run(order=3)
def test_activation_softmax_mlx():
    """Tests modules.Softmax activation with MLX backend."""
    t_matrix = tensor.convert_to_tensor([[1.0, 2.0, 1.0], [-1.0, 0.0, 1.0]])
    activation = modules.Softmax()
    output = activation(t_matrix)
    from ember_ml.nn.modules.activations import softmax
    expected = softmax(t_matrix, axis=-1)
    assert ops.allclose(output, expected), "Softmax module (axis=-1) failed"
    activation_ax0 = modules.Softmax(axis=0)
    output_ax0 = activation_ax0(t_matrix)
    expected_ax0 = softmax(t_matrix, axis=0)
    assert ops.allclose(output_ax0, expected_ax0), "Softmax module (axis=0) failed"

# --- MLX Softplus Test ---
@mark.run(order=3)
def test_activation_softplus_mlx():
    """Tests modules.Softplus activation with MLX backend."""
    input_tensor = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    activation = modules.Softplus()
    output = activation(input_tensor)
    from ember_ml.nn.modules.activations import softplus
    expected = softplus(input_tensor)
    assert ops.allclose(output, expected), "Softplus module failed"

# --- MLX LeCunTanh Test ---
@mark.run(order=3)
def test_activation_lecun_tanh_mlx():
    """Tests modules.LeCunTanh activation with MLX backend."""
    activation_input = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    activation = modules.LeCunTanh()
    output = activation(activation_input)
    scale_factor = tensor.convert_to_tensor(0.66666667)
    amplitude = tensor.convert_to_tensor(1.7159)
    from ember_ml.nn.modules.activations import tanh
    expected = ops.multiply(amplitude, tanh(ops.multiply(scale_factor, activation_input)))
    assert ops.allclose(output, expected), "LeCunTanh module failed"

# --- MLX Dropout Test ---
@mark.run(order=3)
def test_activation_dropout_mlx():
    """Tests modules.Dropout activation with MLX backend."""
    activation_input = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    rate = 0.5
    activation = modules.Dropout(rate=rate, seed=42)
    output_train = activation(activation_input, training=True)
    assert tensor.shape(output_train) == tensor.shape(activation_input), "Dropout training shape mismatch"
    output_eval = activation(activation_input, training=False)
    assert ops.allclose(output_eval, activation_input), "Dropout evaluation failed"
    output_train_2 = activation(activation_input, training=True)
    assert ops.allclose(output_train, output_train_2), "Dropout not deterministic with seed"