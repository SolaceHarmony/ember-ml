# tests/nn/modules/test_serialization.py
import pytest
from ember_ml.nn import modules # Import top-level modules access
from ember_ml.nn.modules import wiring # Import wiring subpackage
from ember_ml.nn.modules import rnn
from ember_ml.nn.modules import Dense # Import from modules package
# Need to also import container for Sequential test
from ember_ml.nn import container

# Test NeuronMap subclasses
def test_fully_connected_map_serialization():
    """Tests saving and loading FullyConnectedMap."""
    units = 16
    output_dim = 8
    input_dim = 10 # For building
    seed = 123
    sparsity = 0.1 # Test non-default sparsity

    # Create original instance
    map_orig = wiring.FullyConnectedMap(
        units=units,
        output_dim=output_dim,
        input_dim=input_dim, # Pass input_dim to constructor
        sparsity_level=sparsity,
        seed=seed
    )
    # Build explicitly to ensure input_dim is set internally if needed by get_config
    map_orig.build(input_dim)

    # Get configuration
    config = map_orig.get_config()
    # Base NeuronMap get_config saves: units, output_dim, input_dim, sparsity_level, seed
    # FullyConnectedMap relies on base get_config

    # Reconstruct from configuration using the base class from_config
    # (which should call the correct subclass constructor)
    map_reconstructed = wiring.FullyConnectedMap.from_config(config)

    # Assertions
    assert isinstance(map_reconstructed, wiring.FullyConnectedMap)
    assert map_reconstructed.units == units
    assert map_reconstructed.output_dim == output_dim
    assert map_reconstructed.input_dim == input_dim # Check if input_dim is restored
    assert map_reconstructed.seed == seed
    assert map_reconstructed.sparsity_level == sparsity

def test_ncp_map_serialization():
    """Tests saving and loading NCPMap."""
    inter_neurons=10
    motor_neurons=4
    sensory_neurons=5 # Optional arg
    input_dim = 5 # Needed for build
    seed = 456
    sparsity = 0.3

    map_orig = wiring.NCPMap(
        inter_neurons=inter_neurons,
        motor_neurons=motor_neurons,
        sensory_neurons=sensory_neurons,
        sparsity_level=sparsity,
        seed=seed
        # Let units, input_dim, output_dim be calculated/set by __init__ or build
    )
    map_orig.build(input_dim=input_dim) # Build is crucial here

    config = map_orig.get_config()
    # NCPMap adds its specific args to the base config

    map_reconstructed = wiring.NCPMap.from_config(config)

    assert isinstance(map_reconstructed, wiring.NCPMap)
    assert map_reconstructed.inter_neurons == inter_neurons
    assert map_reconstructed.motor_neurons == motor_neurons
    assert map_reconstructed.sensory_neurons == sensory_neurons
    assert map_reconstructed.seed == seed
    assert map_reconstructed.sparsity_level == sparsity
    # Check calculated dims
    assert map_reconstructed.units == inter_neurons + motor_neurons + sensory_neurons
    assert map_reconstructed.output_dim == motor_neurons
    assert map_reconstructed.input_dim == input_dim


def test_random_map_serialization():
    """Tests saving and loading RandomMap."""
    units = 20
    output_dim = 10
    input_dim = 15
    seed = 789
    sparsity = 0.7

    map_orig = wiring.RandomMap(
        units=units,
        output_dim=output_dim,
        input_dim=input_dim,
        sparsity_level=sparsity,
        seed=seed
    )
    map_orig.build(input_dim=input_dim)

    config = map_orig.get_config()
    # Relies on base NeuronMap get_config

    map_reconstructed = wiring.RandomMap.from_config(config)

    assert isinstance(map_reconstructed, wiring.RandomMap)
    assert map_reconstructed.units == units
    assert map_reconstructed.output_dim == output_dim
    assert map_reconstructed.input_dim == input_dim
    assert map_reconstructed.seed == seed
    assert map_reconstructed.sparsity_level == sparsity


# Test Activation Modules
def test_relu_serialization():
    """Tests saving and loading ReLU activation module."""
    act_orig = modules.ReLU()
    config = act_orig.get_config() # Should be {}
    act_reconstructed = modules.ReLU.from_config(config)
    assert isinstance(act_reconstructed, modules.ReLU)

def test_softmax_serialization():
    """Tests saving and loading Softmax activation module."""
    axis = -2
    act_orig = modules.Softmax(axis=axis)
    config = act_orig.get_config()
    act_reconstructed = modules.Softmax.from_config(config)
    assert isinstance(act_reconstructed, modules.Softmax)
    assert act_reconstructed.axis == axis

def test_dropout_serialization():
    """Tests saving and loading Dropout module."""
    rate = 0.25
    seed = 111
    mod_orig = modules.Dropout(rate=rate, seed=seed)
    config = mod_orig.get_config()
    mod_reconstructed = modules.Dropout.from_config(config)
    assert isinstance(mod_reconstructed, modules.Dropout)
    assert mod_reconstructed.rate == rate
    assert mod_reconstructed.seed == seed

# Test Cells
def test_lstm_cell_serialization():
    """Tests saving and loading LSTMCell."""
    input_s = 10
    hidden_s = 20
    use_b = False
    cell_orig = rnn.LSTMCell(input_size=input_s, hidden_size=hidden_s, use_bias=use_b)
    config = cell_orig.get_config()
    cell_reconstructed = rnn.LSTMCell.from_config(config)
    assert isinstance(cell_reconstructed, rnn.LSTMCell)
    assert cell_reconstructed.input_size == input_s
    assert cell_reconstructed.hidden_size == hidden_s
    assert cell_reconstructed.use_bias == use_b

def test_ltc_cell_serialization():
    """Tests saving and loading LTCCell (wired)."""
    input_s = 12
    hidden_s = 16
    output_s = 8

    # Create a map (needs to be built)
    ncp_map = wiring.NCPMap(inter_neurons=hidden_s-output_s-input_s, motor_neurons=output_s, sensory_neurons=input_s, seed=1)
    # LTCCell init passes input_size to ModuleWiredCell which calls build if needed
    # ncp_map.build(input_dim=input_s) # Build explicitly just in case

    cell_orig = rnn.LTCCell(
        neuron_map=ncp_map,
        in_features=input_s, # Pass in_features for build call inside __init__
        implicit_param_constraints=True,
        ode_unfolds=8
    )
    config = cell_orig.get_config()

    # Reconstruction
    cell_reconstructed = rnn.LTCCell.from_config(config)

    assert isinstance(cell_reconstructed, rnn.LTCCell)
    assert isinstance(cell_reconstructed.neuron_map, wiring.NCPMap)
    assert cell_reconstructed.input_size == input_s
    assert cell_reconstructed.hidden_size == hidden_s
    assert cell_reconstructed.neuron_map.units == hidden_s
    assert cell_reconstructed._implicit_param_constraints == True
    assert cell_reconstructed._ode_unfolds == 8

# Test Layers
def test_dense_serialization():
    """Tests saving and loading Dense layer."""
    units = 5
    activation = 'relu'
    # Dense is already imported from ember_ml.nn.modules
    layer_orig = Dense(units=units, activation=activation)
    # Note: Dense builds weights on first forward pass, config is independent of build state
    config = layer_orig.get_config()
    layer_reconstructed = Dense.from_config(config)
    assert isinstance(layer_reconstructed, Dense)
    assert layer_reconstructed.units == units
    assert layer_reconstructed.activation == activation


def test_ltc_layer_serialization_with_map():
    """Tests saving and loading LTC layer constructed with a NeuronMap."""
    input_s = 10
    hidden_s = 20
    output_s = 5
    # Make sure the map has the input dimension information
    ncp_map = wiring.NCPMap(inter_neurons=hidden_s-output_s-input_s, motor_neurons=output_s, sensory_neurons=input_s, seed=2)
    # Explicitly build the map with input dimension
    ncp_map.build(input_s)
    
    # Use new API - only pass neuron_map
    layer_orig = rnn.LTC(
        neuron_map=ncp_map,
        return_sequences=False
    )
    config = layer_orig.get_config()

    # Reconstruction
    layer_reconstructed = rnn.LTC.from_config(config)

    assert isinstance(layer_reconstructed, rnn.LTC)
    assert isinstance(layer_reconstructed.rnn_cell, rnn.LTCCell)
    assert isinstance(layer_reconstructed.rnn_cell.neuron_map, wiring.NCPMap)
    assert layer_reconstructed.input_size == input_s
    assert layer_reconstructed.rnn_cell.hidden_size == hidden_s
    assert layer_reconstructed.return_sequences == False

# Test Sequential
def test_sequential_serialization():
    """Tests saving and loading Sequential container."""
    # Create map with input dimension specified
    map1 = wiring.FullyConnectedMap(units=8, input_dim=10)
    # Use new API - only pass neuron_map
    layer1 = rnn.LTC(neuron_map=map1)
    layer2 = modules.ReLU()
    layer3 = Dense(units=5)

    # Need to use container.Sequential for instantiation
    seq_orig = container.Sequential(layer1, layer2, layer3)
    config = seq_orig.get_config()

    # Reconstruction
    seq_reconstructed = container.Sequential.from_config(config)

    assert isinstance(seq_reconstructed, container.Sequential)
    assert len(seq_reconstructed) == len(seq_orig)
    assert isinstance(seq_reconstructed[0], rnn.LTC)
    assert isinstance(seq_reconstructed[1], modules.ReLU)
    assert isinstance(seq_reconstructed[2], Dense)
    # Check attributes of reconstructed layers
    assert seq_reconstructed[0].input_size == 10
    assert seq_reconstructed[0].rnn_cell.hidden_size == 8
    assert seq_reconstructed[2].units == 5