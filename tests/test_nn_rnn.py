import pytest
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn import modules

# Assume conftest.py provides 'backend' fixture

@pytest.fixture
def rnn_params(backend):
    """Fixture for RNN parameters."""
    ops.set_backend(backend)
    input_size = 5
    hidden_size = 10
    batch_size = 4
    seq_len = 3
    return input_size, hidden_size, batch_size, seq_len

# --- RNN Tests ---

def test_rnn_cell_forward(rnn_params, backend):
    """Tests the forward pass shape for RNNCell."""
    ops.set_backend(backend)
    input_size, hidden_size, batch_size, _ = rnn_params
    
    cell = modules.RNNCell(input_size, hidden_size)
    x_t = tensor.random_normal((batch_size, input_size))
    h_prev = tensor.random_normal((batch_size, hidden_size))
    
    h_next = cell(x_t, h_prev)
    
    assert isinstance(h_next, tensor.EmberTensor), "RNNCell output is not EmberTensor"
    assert tensor.shape(h_next) == (batch_size, hidden_size), "RNNCell output shape mismatch"

def test_rnn_layer_forward(rnn_params, backend):
    """Tests the forward pass shape for RNN layer."""
    ops.set_backend(backend)
    input_size, hidden_size, batch_size, seq_len = rnn_params

    layer = modules.RNN(input_size, hidden_size)
    x = tensor.random_normal((batch_size, seq_len, input_size))
    
    y, h_final = layer(x)
    
    assert isinstance(y, tensor.EmberTensor), "RNN layer output y is not EmberTensor"
    assert isinstance(h_final, tensor.EmberTensor), "RNN layer output h_final is not EmberTensor"
    assert tensor.shape(y) == (batch_size, seq_len, hidden_size), "RNN layer output y shape mismatch"
    assert tensor.shape(h_final) == (batch_size, hidden_size), "RNN layer output h_final shape mismatch"

# --- LSTM Tests ---

def test_lstm_cell_forward(rnn_params, backend):
    """Tests the forward pass shape for LSTMCell."""
    ops.set_backend(backend)
    input_size, hidden_size, batch_size, _ = rnn_params

    cell = modules.LSTMCell(input_size, hidden_size)
    x_t = tensor.random_normal((batch_size, input_size))
    h_prev = tensor.random_normal((batch_size, hidden_size))
    c_prev = tensor.random_normal((batch_size, hidden_size))
    
    h_next, c_next = cell(x_t, (h_prev, c_prev))

    assert isinstance(h_next, tensor.EmberTensor), "LSTMCell output h_next is not EmberTensor"
    assert isinstance(c_next, tensor.EmberTensor), "LSTMCell output c_next is not EmberTensor"
    assert tensor.shape(h_next) == (batch_size, hidden_size), "LSTMCell output h_next shape mismatch"
    assert tensor.shape(c_next) == (batch_size, hidden_size), "LSTMCell output c_next shape mismatch"

def test_lstm_layer_forward(rnn_params, backend):
    """Tests the forward pass shape for LSTM layer."""
    ops.set_backend(backend)
    input_size, hidden_size, batch_size, seq_len = rnn_params

    layer = modules.LSTM(input_size, hidden_size)
    x = tensor.random_normal((batch_size, seq_len, input_size))

    y, (h_final, c_final) = layer(x)

    assert isinstance(y, tensor.EmberTensor), "LSTM layer output y is not EmberTensor"
    assert isinstance(h_final, tensor.EmberTensor), "LSTM layer output h_final is not EmberTensor"
    assert isinstance(c_final, tensor.EmberTensor), "LSTM layer output c_final is not EmberTensor"
    assert tensor.shape(y) == (batch_size, seq_len, hidden_size), "LSTM layer output y shape mismatch"
    assert tensor.shape(h_final) == (batch_size, hidden_size), "LSTM layer output h_final shape mismatch"
    assert tensor.shape(c_final) == (batch_size, hidden_size), "LSTM layer output c_final shape mismatch"

# --- GRU Tests ---

def test_gru_cell_forward(rnn_params, backend):
    """Tests the forward pass shape for GRUCell."""
    ops.set_backend(backend)
    input_size, hidden_size, batch_size, _ = rnn_params

    cell = modules.GRUCell(input_size, hidden_size)
    x_t = tensor.random_normal((batch_size, input_size))
    h_prev = tensor.random_normal((batch_size, hidden_size))

    h_next = cell(x_t, h_prev)

    assert isinstance(h_next, tensor.EmberTensor), "GRUCell output is not EmberTensor"
    assert tensor.shape(h_next) == (batch_size, hidden_size), "GRUCell output shape mismatch"

def test_gru_layer_forward(rnn_params, backend):
    """Tests the forward pass shape for GRU layer."""
    ops.set_backend(backend)
    input_size, hidden_size, batch_size, seq_len = rnn_params

    layer = modules.GRU(input_size, hidden_size)
    x = tensor.random_normal((batch_size, seq_len, input_size))

    y, h_final = layer(x)



# --- CfC Tests ---

def test_cfc_cell_forward(rnn_params, backend):
    """Tests the forward pass shape for CfCCell."""
    ops.set_backend(backend)
    input_size, hidden_size, batch_size, _ = rnn_params

    # CfCCell requires input_size to be passed during init
    cell = modules.CfCCell(input_size=input_size, hidden_size=hidden_size)
    x_t = tensor.random_normal((batch_size, input_size))
    h_prev = tensor.random_normal((batch_size, hidden_size))

    h_next = cell(x_t, h_prev)

    assert isinstance(h_next, tensor.EmberTensor), "CfCCell output is not EmberTensor"
    assert tensor.shape(h_next) == (batch_size, hidden_size), "CfCCell output shape mismatch"

def test_cfc_layer_forward(rnn_params, backend):
    """Tests the forward pass shape for CfC layer."""
    ops.set_backend(backend)
    input_size, hidden_size, batch_size, seq_len = rnn_params

    layer = modules.CfC(input_size=input_size, hidden_size=hidden_size)
    x = tensor.random_normal((batch_size, seq_len, input_size))

    y, h_final = layer(x)

    assert isinstance(y, tensor.EmberTensor), "CfC layer output y is not EmberTensor"
    assert isinstance(h_final, tensor.EmberTensor), "CfC layer output h_final is not EmberTensor"
    assert tensor.shape(y) == (batch_size, seq_len, hidden_size), "CfC layer output y shape mismatch"
    assert tensor.shape(h_final) == (batch_size, hidden_size), "CfC layer output h_final shape mismatch"

# --- LTC Tests ---

def test_ltc_cell_forward(rnn_params, backend):
    """Tests the forward pass shape for LTCCell."""
    ops.set_backend(backend)
    input_size, hidden_size, batch_size, _ = rnn_params

    # LTCCell requires a NeuronMap
    # Use AutoNCP logic to create a default map for testing shape
    # Or create a simple FullyConnectedMap
    neuron_map = modules.wiring.FullyConnectedMap(units=hidden_size, output_size=hidden_size, input_size=input_size)
    cell = modules.LTCCell(neuron_map=neuron_map)
    
    x_t = tensor.random_normal((batch_size, input_size))
    h_prev = tensor.random_normal((batch_size, hidden_size))

    h_next = cell(x_t, h_prev)

    assert isinstance(h_next, tensor.EmberTensor), "LTCCell output is not EmberTensor"
    assert tensor.shape(h_next) == (batch_size, hidden_size), "LTCCell output shape mismatch"

def test_ltc_layer_forward(rnn_params, backend):
    """Tests the forward pass shape for LTC layer."""
    ops.set_backend(backend)
    input_size, hidden_size, batch_size, seq_len = rnn_params

    # LTC layer can auto-create map if neuron_map='auto'
    layer = modules.LTC(input_size=input_size, hidden_size=hidden_size, neuron_map='auto')
    x = tensor.random_normal((batch_size, seq_len, input_size))

    y, h_final = layer(x)

    assert isinstance(y, tensor.EmberTensor), "LTC layer output y is not EmberTensor"
    assert isinstance(h_final, tensor.EmberTensor), "LTC layer output h_final is not EmberTensor"
    assert tensor.shape(y) == (batch_size, seq_len, hidden_size), "LTC layer output y shape mismatch"
    assert tensor.shape(h_final) == (batch_size, hidden_size), "LTC layer output h_final shape mismatch"


    assert isinstance(y, tensor.EmberTensor), "GRU layer output y is not EmberTensor"
    assert isinstance(h_final, tensor.EmberTensor), "GRU layer output h_final is not EmberTensor"
    assert tensor.shape(y) == (batch_size, seq_len, hidden_size), "GRU layer output y shape mismatch"
    assert tensor.shape(h_final) == (batch_size, hidden_size), "GRU layer output h_final shape mismatch"

# TODO: Add tests for CfC, CfCCell, WiredCfCCell
# TODO: Add tests for LTC, LTCCell
# TODO: Add tests for StrideAware base, StrideAwareCell, StrideAwareCfC, StrideAwareWiredCfCCell
# TODO: Add tests for initial state handling
# TODO: Add tests for returning full sequence vs. last output