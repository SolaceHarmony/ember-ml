import pytest
# Remove numpy import -> import numpy as np 
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules import wiring # Import the wiring submodule
from ember_ml.ops import stats # Import stats

# Assume conftest.py provides 'backend' fixture

# --- NCPMap Tests ---

def test_ncpmap_creation_properties(backend):
    """Tests NCPMap creation and basic properties."""
    ops.set_backend(backend)
    inter = 10
    command = 5
    motor = 3
    sensory = 8
    expected_units = inter + motor # Sensory map to input_size, not internal units

    neuron_map = wiring.NCPMap(
        inter_neurons=inter,
        motor_neurons=motor,
        sensory_neurons=sensory,
        seed=42
    )

    assert isinstance(neuron_map, wiring.NeuronMap), "NCPMap is not instance of NeuronMap"
    assert neuron_map.units == expected_units, f"NCPMap units mismatch: expected {expected_units}, got {neuron_map.units}"
    assert neuron_map.output_size == motor, f"NCPMap output_size mismatch: expected {motor}, got {neuron_map.output_size}"
    assert neuron_map.input_size == sensory, f"NCPMap input_size mismatch: expected {sensory}, got {neuron_map.input_size}"
    
    # Check matrix shapes (if they exist and are built)
    assert hasattr(neuron_map, 'adjacency_matrix'), "NCPMap missing adjacency_matrix"
    assert hasattr(neuron_map, 'sensory_adjacency_matrix'), "NCPMap missing sensory_adjacency_matrix"
    assert tensor.shape(neuron_map.adjacency_matrix) == (expected_units, expected_units), "NCPMap adjacency_matrix shape mismatch"
    assert tensor.shape(neuron_map.sensory_adjacency_matrix) == (sensory, expected_units), "NCPMap sensory_adjacency_matrix shape mismatch"

# --- FullyConnectedMap Tests ---

def test_fullyconnectedmap_creation_properties(backend):
    """Tests FullyConnectedMap creation and basic properties."""
    ops.set_backend(backend)
    units = 10
    output_size = 5
    input_size = 8

    neuron_map = wiring.FullyConnectedMap(
        units=units,
        output_size=output_size,
        input_size=input_size
    )

    assert isinstance(neuron_map, wiring.NeuronMap), "FullyConnectedMap is not instance of NeuronMap"
    assert neuron_map.units == units, f"FullyConnectedMap units mismatch: expected {units}, got {neuron_map.units}"
    assert neuron_map.output_size == output_size, f"FullyConnectedMap output_size mismatch: expected {output_size}, got {neuron_map.output_size}"
    assert neuron_map.input_size == input_size, f"FullyConnectedMap input_size mismatch: expected {input_size}, got {neuron_map.input_size}"

    # Check matrix shapes
    assert hasattr(neuron_map, 'adjacency_matrix'), "FullyConnectedMap missing adjacency_matrix"
    assert hasattr(neuron_map, 'sensory_adjacency_matrix'), "FullyConnectedMap missing sensory_adjacency_matrix"
    assert tensor.shape(neuron_map.adjacency_matrix) == (units, units), "FullyConnectedMap adjacency_matrix shape mismatch"
    assert tensor.shape(neuron_map.sensory_adjacency_matrix) == (input_size, units), "FullyConnectedMap sensory_adjacency_matrix shape mismatch"
    
    # Check if adjacency is all ones (fully connected)
    expected_adj = tensor.ones((units, units))
    # Need to ensure the backend uses float for comparison if ones creates float
    expected_adj = tensor.cast(expected_adj, tensor.dtype(neuron_map.adjacency_matrix))
    assert ops.all(ops.equal(neuron_map.adjacency_matrix, expected_adj)), "FullyConnectedMap adjacency matrix not all ones"


# --- RandomMap Tests ---

def test_randommap_creation_properties(backend):
    """Tests RandomMap creation and basic properties."""
    ops.set_backend(backend)
    units = 12
    output_size = 6
    input_size = 9
    sparsity = 0.3 # Expect roughly 70% connections (sparsity is proportion of ZEROS)

    neuron_map = wiring.RandomMap(
        units=units,
        output_size=output_size,
        input_size=input_size,
        sparsity_level=sparsity,
        seed=42
    )

    assert isinstance(neuron_map, wiring.NeuronMap), "RandomMap is not instance of NeuronMap"
    assert neuron_map.units == units, f"RandomMap units mismatch: expected {units}, got {neuron_map.units}"
    assert neuron_map.output_size == output_size, f"RandomMap output_size mismatch: expected {output_size}, got {neuron_map.output_size}"
    assert neuron_map.input_size == input_size, f"RandomMap input_size mismatch: expected {input_size}, got {neuron_map.input_size}"
    assert neuron_map.sparsity_level == sparsity, "RandomMap sparsity_level mismatch"

    # Check matrix shapes
    assert hasattr(neuron_map, 'adjacency_matrix'), "RandomMap missing adjacency_matrix"
    assert hasattr(neuron_map, 'sensory_adjacency_matrix'), "RandomMap missing sensory_adjacency_matrix"
    assert tensor.shape(neuron_map.adjacency_matrix) == (units, units), "RandomMap adjacency_matrix shape mismatch"
    assert tensor.shape(neuron_map.sensory_adjacency_matrix) == (input_size, units), "RandomMap sensory_adjacency_matrix shape mismatch"

    # Check sparsity roughly using ops/stats.sum
    is_zero = ops.equal(neuron_map.adjacency_matrix, 0.0)
    num_zeros = stats.sum(tensor.cast(is_zero, tensor.int32))
    total_possible = units * units
    actual_sparsity = tensor.item(num_zeros) / total_possible
    # Check if sparsity is within +/- 10% tolerance of expected
    assert abs(actual_sparsity - sparsity) < 0.10, f"RandomMap adjacency sparsity seems off: expected {sparsity}, got {actual_sparsity}"

def test_ncpmap_config_serialization(backend):
    """Tests get_config and from_config for NCPMap."""
    ops.set_backend(backend)
    inter = 8
    command = 4
    motor = 3
    sensory = 5
    seed = 123
    sparsity = 0.4 # Non-default sparsity

    # Create original map
    original_map = wiring.NCPMap(
        inter_neurons=inter,
        motor_neurons=motor,
        sensory_neurons=sensory,
        sparsity_level=sparsity,
        seed=seed
    )

    # Get config
    config = original_map.get_config()

    # Assert config contains expected keys
    assert 'name' in config
    assert config['name'] == 'NCPMap' # Assuming class name is stored
    assert 'units' in config
    assert 'output_size' in config
    assert 'input_size' in config
    assert 'sparsity_level' in config
    assert 'seed' in config
    # Check specific config values passed to constructor
    assert config['sparsity_level'] == sparsity
    assert config['seed'] == seed
    # Check derived values
    assert config['units'] == inter + motor
    assert config['output_size'] == motor
    assert config['input_size'] == sensory


    # Create new map from config
    new_map = wiring.NCPMap.from_config(config)

    # Check if new map matches original properties
    assert isinstance(new_map, wiring.NCPMap), "from_config did not return NCPMap"
    assert new_map.units == original_map.units, "Config units mismatch"
    assert new_map.output_size == original_map.output_size, "Config output_size mismatch"
    assert new_map.input_size == original_map.input_size, "Config input_size mismatch"
    assert new_map.sparsity_level == original_map.sparsity_level, "Config sparsity_level mismatch"
    assert new_map.seed == original_map.seed, "Config seed mismatch"

    # Check if generated matrices are the same (due to same seed)
    assert ops.allclose(new_map.adjacency_matrix, original_map.adjacency_matrix), "Config adjacency_matrix mismatch"
    assert ops.allclose(new_map.sensory_adjacency_matrix, original_map.sensory_adjacency_matrix), "Config sensory_adjacency_matrix mismatch"

# TODO: Add tests for NeuronMap base class functionality if any (e.g., config serialization)
# TODO: Add more detailed tests for matrix contents if specific structures are guaranteed (e.g., NCPMap sections)