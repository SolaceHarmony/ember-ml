# tests/mlx_tests/test_nn_rnn.py
import pytest
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn import modules

# Note: Assumes conftest.py provides the mlx_backend fixture

# Helper function to get RNN parameters
def _get_rnn_params():
    input_size = 5
    hidden_size = 10
    batch_size = 4
    seq_len = 3
    return input_size, hidden_size, batch_size, seq_len

def test_rnn_cell_forward_mlx(mlx_backend): # Use fixture
    """Tests RNNCell forward pass shape with MLX backend."""
    input_size, hidden_size, batch_size, _ = _get_rnn_params()
    cell = modules.RNNCell(input_size, hidden_size)
    x_t = tensor.random_normal((batch_size, input_size))
    h_prev = tensor.random_normal((batch_size, hidden_size))
    # cell.forward returns output, new_state where new_state=[h_next]
    output, new_state = cell(x_t, [h_prev]) # Pass state as list
    h_next = output # For simple RNN, output is the new hidden state
    assert tensor.shape(h_next) == (batch_size, hidden_size), "Shape mismatch"
    assert isinstance(new_state, list) and len(new_state) == 1, "New state should be a list with one element"
    assert tensor.shape(new_state[0]) == (batch_size, hidden_size), "State shape mismatch"

def test_rnn_layer_forward_mlx(mlx_backend): # Use fixture
    """Tests RNN layer forward pass shape with MLX backend."""
    input_size, hidden_size, batch_size, seq_len = _get_rnn_params()
    layer = modules.RNN(input_size, hidden_size) # Default return_state=False
    x = tensor.random_normal((batch_size, seq_len, input_size))
    # Layer returns outputs by default. Set return_state=True to get final state.
    outputs = layer(x) # Get only outputs
    y = outputs
    # Now test with state return
    layer_state = modules.RNN(input_size, hidden_size, return_state=True)
    outputs_state, final_state = layer_state(x)
    h_final = final_state[0] # Unpack final state list
    assert tensor.shape(y) == (batch_size, seq_len, hidden_size), "y shape mismatch"
    # Shape is (num_layers * num_directions, batch_size, hidden_size)
    assert tensor.shape(h_final) == (1, batch_size, hidden_size), "h_final shape mismatch"

def test_lstm_cell_forward_mlx(mlx_backend): # Use fixture
    """Tests LSTMCell forward pass shape with MLX backend."""
    input_size, hidden_size, batch_size, _ = _get_rnn_params()
    cell = modules.LSTMCell(input_size, hidden_size)
    x_t = tensor.random_normal((batch_size, input_size))
    h_prev = tensor.random_normal((batch_size, hidden_size))
    c_prev = tensor.random_normal((batch_size, hidden_size))
    # cell.forward returns output, new_state where new_state=[h_next, c_next]
    output, new_state = cell(x_t, [h_prev, c_prev]) # Pass state as list
    h_next = output # For LSTM, output is the new hidden state h_next
    c_next = new_state[1] # Get cell state c_next from new_state list
    assert tensor.shape(h_next) == (batch_size, hidden_size), "h_next shape mismatch"
    assert tensor.shape(c_next) == (batch_size, hidden_size), "c_next shape mismatch"
    assert isinstance(new_state, list) and len(new_state) == 2, "New state should be a list with two elements"
    assert tensor.shape(new_state[0]) == (batch_size, hidden_size), "h_next state shape mismatch"

def test_lstm_layer_forward_mlx(mlx_backend): # Use fixture
    """Tests LSTM layer forward pass shape with MLX backend."""
    input_size, hidden_size, batch_size, seq_len = _get_rnn_params()
    layer = modules.LSTM(input_size, hidden_size) # Default return_state=False
    x = tensor.random_normal((batch_size, seq_len, input_size))
    # Layer returns outputs by default. Set return_state=True to get final state.
    outputs = layer(x) # Get only outputs
    y = outputs
    # Now test with state return
    layer_state = modules.LSTM(input_size, hidden_size, return_state=True)
    outputs_state, final_state = layer_state(x)
    h_final, c_final = final_state # Unpack final state tuple
    assert tensor.shape(y) == (batch_size, seq_len, hidden_size), "y shape mismatch"
    # Shape is (num_layers * num_directions, batch_size, hidden_size)
    assert tensor.shape(h_final) == (1, batch_size, hidden_size), "h_final shape mismatch"
    assert tensor.shape(c_final) == (1, batch_size, hidden_size), "c_final shape mismatch"

def test_gru_cell_forward_mlx(mlx_backend): # Use fixture
    """Tests GRUCell forward pass shape with MLX backend."""
    input_size, hidden_size, batch_size, _ = _get_rnn_params()
    cell = modules.GRUCell(input_size, hidden_size)
    x_t = tensor.random_normal((batch_size, input_size))
    h_prev = tensor.random_normal((batch_size, hidden_size))
    # cell.forward returns output, new_state where new_state=[h_next]
    output, new_state = cell(x_t, [h_prev]) # Pass state as list
    h_next = output # For GRU, output is the new hidden state
    assert tensor.shape(h_next) == (batch_size, hidden_size), "Shape mismatch"
    assert isinstance(new_state, list) and len(new_state) == 1, "New state should be a list with one element"
    assert tensor.shape(new_state[0]) == (batch_size, hidden_size), "State shape mismatch"

def test_gru_layer_forward_mlx(mlx_backend): # Use fixture
    """Tests GRU layer forward pass shape with MLX backend."""
    input_size, hidden_size, batch_size, seq_len = _get_rnn_params()
    layer = modules.GRU(input_size, hidden_size) # Default return_state=False
    x = tensor.random_normal((batch_size, seq_len, input_size))
    # Layer returns outputs by default. Set return_state=True to get final state.
    outputs = layer(x) # Get only outputs
    y = outputs
    # Now test with state return
    layer_state = modules.GRU(input_size, hidden_size, return_state=True)
    outputs_state, final_state = layer_state(x)
    h_final = final_state[0] # Unpack final state list
    assert tensor.shape(y) == (batch_size, seq_len, hidden_size), "y shape mismatch"
    # Shape is (num_layers * num_directions, batch_size, hidden_size)
    assert tensor.shape(h_final) == (1, batch_size, hidden_size), "h_final shape mismatch"

def test_cfc_cell_forward_mlx(mlx_backend): # Use fixture
    """Tests CfCCell forward pass shape with MLX backend."""
    input_size, hidden_size, batch_size, _ = _get_rnn_params()
    cell = modules.CfCCell(input_size=input_size, hidden_size=hidden_size)
    x_t = tensor.random_normal((batch_size, input_size))
    h_prev = tensor.random_normal((batch_size, hidden_size))
    t_prev = tensor.random_normal((batch_size, hidden_size)) # CfC also has time state
    # cell.forward returns output, new_state where new_state=[h_next, t_next]
    output, new_state = cell(x_t, [h_prev, t_prev]) # Pass state as list
    h_next = output # For CfC, output is the new hidden state h_next
    assert tensor.shape(h_next) == (batch_size, hidden_size), "Shape mismatch"
    assert isinstance(new_state, list) and len(new_state) == 2, "New state should be a list with two elements"
    assert tensor.shape(new_state[0]) == (batch_size, hidden_size), "h_next state shape mismatch"
    assert tensor.shape(new_state[1]) == (batch_size, hidden_size), "t_next state shape mismatch"

def test_cfc_layer_forward_mlx(mlx_backend): # Use fixture
    """Tests CfC layer forward pass shape with MLX backend."""
    input_size, hidden_size, batch_size, seq_len = _get_rnn_params()
    cell = modules.CfCCell(input_size=input_size, hidden_size=hidden_size) # Must create cell first
    layer = modules.CfC(cell_or_map=cell, return_sequences=True) # Set return_sequences to True
    x = tensor.random_normal((batch_size, seq_len, input_size))
    # Layer returns outputs by default. Set return_state=True to get final state.
    outputs = layer(x) # Get only outputs
    y = outputs
    assert tensor.shape(y) == (batch_size, seq_len, hidden_size), "y shape mismatch"

def test_ltc_cell_forward_mlx(mlx_backend): # Use fixture
    """Tests LTCCell forward pass shape with MLX backend."""
    from ember_ml.nn import modules
    input_size, hidden_size, batch_size, _ = _get_rnn_params()
    neuron_map = modules.FullyConnectedMap(units=hidden_size, output_dim=hidden_size, input_dim=input_size)
    cell = modules.LTCCell(neuron_map=neuron_map)
    x_t = tensor.random_normal((batch_size, input_size))
    h_prev = tensor.random_normal((batch_size, hidden_size))
    # cell.forward returns output, new_state where new_state is h_next for LTC
    output, new_state = cell(x_t, h_prev) # Pass previous state directly
    h_next = output # For LTC, output is the new hidden state
    assert tensor.shape(h_next) == (batch_size, hidden_size), "Shape mismatch"
    assert tensor.shape(new_state) == (batch_size, hidden_size), "State shape mismatch"

def test_ltc_layer_forward_mlx(mlx_backend): # Use fixture
    """Tests LTC layer forward pass shape with MLX backend."""
    from ember_ml.nn.modules import wiring
    input_size, hidden_size, batch_size, seq_len = _get_rnn_params()
    # Create a NeuronMap first, e.g., FullyConnectedMap
    neuron_map = wiring.FullyConnectedMap(units=hidden_size, input_dim=input_size, output_dim=hidden_size)
    layer = modules.LTC(neuron_map=neuron_map) # Default return_state=False
    x = tensor.random_normal((batch_size, seq_len, input_size))
    # Layer returns outputs by default. Set return_state=True to get final state.
    outputs = layer(x) # Get only outputs
    y = outputs
    # Now test with state return
    layer_state = modules.LTC(neuron_map=neuron_map, return_state=True)
    outputs_state, final_state = layer_state(x)
    h_final = final_state # Final state is just h_final for LTC
    assert tensor.shape(y) == (batch_size, seq_len, hidden_size), "y shape mismatch"
    assert tensor.shape(h_final) == (batch_size, hidden_size), "h_final shape mismatch"