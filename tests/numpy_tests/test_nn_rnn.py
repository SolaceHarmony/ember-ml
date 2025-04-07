# tests/numpy_tests/test_nn_rnn.py
import pytest
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn import modules

# Note: Assumes conftest.py provides the numpy_backend fixture

# Helper function to get RNN parameters
def _get_rnn_params():
    input_size = 5
    hidden_size = 10
    batch_size = 4
    seq_len = 3
    return input_size, hidden_size, batch_size, seq_len

def test_rnn_cell_forward_numpy(numpy_backend): # Use fixture
    """Tests RNNCell forward pass shape with NumPy backend."""
    input_size, hidden_size, batch_size, _ = _get_rnn_params()
    cell = modules.RNNCell(input_size, hidden_size)
    x_t = tensor.random_normal((batch_size, input_size))
    h_prev = tensor.random_normal((batch_size, hidden_size))
    h_next = cell(x_t, h_prev)
    assert isinstance(h_next, tensor.EmberTensor), "Output not EmberTensor"
    assert tensor.shape(h_next) == (batch_size, hidden_size), "Shape mismatch"

def test_rnn_layer_forward_numpy(numpy_backend): # Use fixture
    """Tests RNN layer forward pass shape with NumPy backend."""
    input_size, hidden_size, batch_size, seq_len = _get_rnn_params()
    layer = modules.RNN(input_size, hidden_size)
    x = tensor.random_normal((batch_size, seq_len, input_size))
    y, h_final = layer(x)
    assert isinstance(y, tensor.EmberTensor), "y not EmberTensor"
    assert isinstance(h_final, tensor.EmberTensor), "h_final not EmberTensor"
    assert tensor.shape(y) == (batch_size, seq_len, hidden_size), "y shape mismatch"
    assert tensor.shape(h_final) == (batch_size, hidden_size), "h_final shape mismatch"

def test_lstm_cell_forward_numpy(numpy_backend): # Use fixture
    """Tests LSTMCell forward pass shape with NumPy backend."""
    input_size, hidden_size, batch_size, _ = _get_rnn_params()
    cell = modules.LSTMCell(input_size, hidden_size)
    x_t = tensor.random_normal((batch_size, input_size))
    h_prev = tensor.random_normal((batch_size, hidden_size))
    c_prev = tensor.random_normal((batch_size, hidden_size))
    h_next, c_next = cell(x_t, (h_prev, c_prev))
    assert isinstance(h_next, tensor.EmberTensor), "h_next not EmberTensor"
    assert isinstance(c_next, tensor.EmberTensor), "c_next not EmberTensor"
    assert tensor.shape(h_next) == (batch_size, hidden_size), "h_next shape mismatch"
    assert tensor.shape(c_next) == (batch_size, hidden_size), "c_next shape mismatch"

def test_lstm_layer_forward_numpy(numpy_backend): # Use fixture
    """Tests LSTM layer forward pass shape with NumPy backend."""
    input_size, hidden_size, batch_size, seq_len = _get_rnn_params()
    layer = modules.LSTM(input_size, hidden_size)
    x = tensor.random_normal((batch_size, seq_len, input_size))
    y, (h_final, c_final) = layer(x)
    assert isinstance(y, tensor.EmberTensor), "y not EmberTensor"
    assert isinstance(h_final, tensor.EmberTensor), "h_final not EmberTensor"
    assert isinstance(c_final, tensor.EmberTensor), "c_final not EmberTensor"
    assert tensor.shape(y) == (batch_size, seq_len, hidden_size), "y shape mismatch"
    assert tensor.shape(h_final) == (batch_size, hidden_size), "h_final shape mismatch"
    assert tensor.shape(c_final) == (batch_size, hidden_size), "c_final shape mismatch"

def test_gru_cell_forward_numpy(numpy_backend): # Use fixture
    """Tests GRUCell forward pass shape with NumPy backend."""
    input_size, hidden_size, batch_size, _ = _get_rnn_params()
    cell = modules.GRUCell(input_size, hidden_size)
    x_t = tensor.random_normal((batch_size, input_size))
    h_prev = tensor.random_normal((batch_size, hidden_size))
    h_next = cell(x_t, h_prev)
    assert isinstance(h_next, tensor.EmberTensor), "Output not EmberTensor"
    assert tensor.shape(h_next) == (batch_size, hidden_size), "Shape mismatch"

def test_gru_layer_forward_numpy(numpy_backend): # Use fixture
    """Tests GRU layer forward pass shape with NumPy backend."""
    input_size, hidden_size, batch_size, seq_len = _get_rnn_params()
    layer = modules.GRU(input_size, hidden_size)
    x = tensor.random_normal((batch_size, seq_len, input_size))
    y, h_final = layer(x)
    assert isinstance(y, tensor.EmberTensor), "y not EmberTensor"
    assert isinstance(h_final, tensor.EmberTensor), "h_final not EmberTensor"
    assert tensor.shape(y) == (batch_size, seq_len, hidden_size), "y shape mismatch"
    assert tensor.shape(h_final) == (batch_size, hidden_size), "h_final shape mismatch"

def test_cfc_cell_forward_numpy(numpy_backend): # Use fixture
    """Tests CfCCell forward pass shape with NumPy backend."""
    input_size, hidden_size, batch_size, _ = _get_rnn_params()
    cell = modules.CfCCell(input_size=input_size, hidden_size=hidden_size)
    x_t = tensor.random_normal((batch_size, input_size))
    h_prev = tensor.random_normal((batch_size, hidden_size))
    h_next = cell(x_t, h_prev)
    assert isinstance(h_next, tensor.EmberTensor), "Output not EmberTensor"
    assert tensor.shape(h_next) == (batch_size, hidden_size), "Shape mismatch"

def test_cfc_layer_forward_numpy(numpy_backend): # Use fixture
    """Tests CfC layer forward pass shape with NumPy backend."""
    input_size, hidden_size, batch_size, seq_len = _get_rnn_params()
    layer = modules.CfC(input_size=input_size, hidden_size=hidden_size)
    x = tensor.random_normal((batch_size, seq_len, input_size))
    y, h_final = layer(x)
    assert isinstance(y, tensor.EmberTensor), "y not EmberTensor"
    assert isinstance(h_final, tensor.EmberTensor), "h_final not EmberTensor"
    assert tensor.shape(y) == (batch_size, seq_len, hidden_size), "y shape mismatch"
    assert tensor.shape(h_final) == (batch_size, hidden_size), "h_final shape mismatch"

def test_ltc_cell_forward_numpy(numpy_backend): # Use fixture
    """Tests LTCCell forward pass shape with NumPy backend."""
    input_size, hidden_size, batch_size, _ = _get_rnn_params()
    neuron_map = modules.wiring.FullyConnectedMap(units=hidden_size, output_size=hidden_size, input_size=input_size)
    cell = modules.LTCCell(neuron_map=neuron_map)
    x_t = tensor.random_normal((batch_size, input_size))
    h_prev = tensor.random_normal((batch_size, hidden_size))
    h_next = cell(x_t, h_prev)
    assert isinstance(h_next, tensor.EmberTensor), "Output not EmberTensor"
    assert tensor.shape(h_next) == (batch_size, hidden_size), "Shape mismatch"

def test_ltc_layer_forward_numpy(numpy_backend): # Use fixture
    """Tests LTC layer forward pass shape with NumPy backend."""
    input_size, hidden_size, batch_size, seq_len = _get_rnn_params()
    layer = modules.LTC(input_size=input_size, hidden_size=hidden_size, neuron_map='auto')
    x = tensor.random_normal((batch_size, seq_len, input_size))
    y, h_final = layer(x)
    assert isinstance(y, tensor.EmberTensor), "y not EmberTensor"
    assert isinstance(h_final, tensor.EmberTensor), "h_final not EmberTensor"
    assert tensor.shape(y) == (batch_size, seq_len, hidden_size), "y shape mismatch"
    assert tensor.shape(h_final) == (batch_size, hidden_size), "h_final shape mismatch"