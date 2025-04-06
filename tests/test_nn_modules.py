import pytest
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn import modules

# Assume conftest.py provides 'backend' fixture

# --- Test Base Module and Parameter ---

class SimpleModule(modules.Module):
    """A simple module for testing parameter registration."""
    def __init__(self, size):
        super().__init__()
        # Ensure parameters have unique underlying tensors if checking identity later
        self.param1 = modules.Parameter(tensor.zeros(size))
        self.param2 = modules.Parameter(tensor.ones((size, size)))
        # This should not be registered as a parameter
        self.non_param = tensor.arange(size)
        # Test nested modules
        # The input dimension for Dense is the same as the 'size' of the SimpleModule input
        # Use ops.multiply and tensor.item() for backend consistency and correct type
        units_val = tensor.item(ops.multiply(tensor.convert_to_tensor(size), tensor.convert_to_tensor(2)))
        self.nested = modules.Dense(input_dim=size, units=units_val) # Example nested module

def test_module_parameter_registration(backend):
    """Tests if parameters are correctly registered in a Module, including nested."""
    ops.set_backend(backend)
    module = SimpleModule(size=3)
    
    registered_params = list(module.parameters())
    
    # Expected parameters: param1, param2, nested.weight, nested.bias
    assert len(registered_params) == 4, f"Incorrect number of parameters registered: {len(registered_params)}"
    
    # More robust check: verify parameter shapes or content if possible/needed
    param_shapes = [tensor.shape(p.data) for p in registered_params]
    expected_shapes = [(3,), (3, 3), (3, 6), (6,)] # Based on SimpleModule and Dense(3, 6)
    
    # Check if all expected shapes are present (order might vary)
    # Use sets for comparison as order doesn't matter
    assert set(param_shapes) == set(expected_shapes), "Parameter shapes mismatch"

    # Check that the non-parameter tensor is not registered
    # Comparing underlying data might be needed if Parameter wraps tensors deeply
    is_non_param_registered = False
    for p in registered_params:
         # Use allclose for float comparison, direct equality for others if possible
         try:
              if ops.allclose(p.data, module.non_param):
                   is_non_param_registered = True
                   break
         except: # Handle potential type errors in comparison
              pass 
    assert not is_non_param_registered, "Non-parameter tensor was incorrectly registered"


def test_parameter_properties(backend):
    """Tests properties of the Parameter class."""
    ops.set_backend(backend)
    data = tensor.convert_to_tensor([1.0, 2.0])
    param = modules.Parameter(data, requires_grad=True)

    assert isinstance(param.data, tensor.EmberTensor), "Parameter data is not EmberTensor"
    assert param.requires_grad is True, "Parameter requires_grad is not True"
    assert tensor.shape(param.data) == tensor.shape(data), "Parameter shape mismatch"
    assert ops.allclose(param.data, data), "Parameter data content mismatch"

    # Test default requires_grad (should be True)
    param_default = modules.Parameter(tensor.ones(3))
    assert param_default.requires_grad is True, "Parameter default requires_grad is not True"

# --- Test Dense Layer ---

@pytest.fixture
def dense_layer(backend):
    """Fixture for Dense layer tests."""
    ops.set_backend(backend)
    in_features = 5
    out_features = 3
    layer = modules.Dense(input_dim=in_features, units=out_features)
    # Set predictable weights/bias for testing if needed, otherwise rely on initialization
    # Example:
    # layer.weight.data = tensor.arange(in_features * out_features).reshape((in_features, out_features))
    # layer.bias.data = tensor.arange(out_features)
    return layer, in_features, out_features

def test_dense_forward_shape(dense_layer, backend):
    """Tests the output shape of the Dense layer forward pass."""
    ops.set_backend(backend)
    layer, in_features, out_features = dense_layer
    batch_size = 4
    
    input_tensor = tensor.random_normal((batch_size, in_features))
    output = layer(input_tensor)

    assert isinstance(output, tensor.EmberTensor), "Dense output is not EmberTensor"
    expected_shape = (batch_size, out_features)
    assert tensor.shape(output) == expected_shape, f"Dense output shape mismatch: expected {expected_shape}, got {tensor.shape(output)}"

def test_dense_parameters(dense_layer, backend):
    """Tests if Dense layer registers weight and bias parameters."""
    ops.set_backend(backend)
    layer, in_features, out_features = dense_layer
    
    params = list(layer.parameters())
    assert len(params) == 2, "Dense layer should have 2 parameters (weight, bias)"

    # Check shapes
    weight_found = False
    bias_found = False
    for p in params:
        p_shape = tensor.shape(p.data)
        if p_shape == (in_features, out_features):
            weight_found = True
        elif p_shape == (out_features,):
            bias_found = True
            
    assert weight_found, "Dense layer weight parameter not found or wrong shape"
    assert bias_found, "Dense layer bias parameter not found or wrong shape"

def test_dense_no_bias(backend):
    """Tests Dense layer without bias."""
    ops.set_backend(backend)
    layer = modules.Dense(input_dim=4, units=2, use_bias=False)
    params = list(layer.parameters())
    assert len(params) == 1, "Dense layer without bias should have 1 parameter"
    assert tensor.shape(params[0].data) == (4, 2), "Dense weight shape incorrect when bias=False"

    # Test forward pass shape
    input_tensor = tensor.random_normal((3, 4))
    output = layer(input_tensor)
    assert tensor.shape(output) == (3, 2), "Dense forward shape incorrect when bias=False"

def test_dense_activation(backend):
    """Tests Dense layer with an activation function."""
    ops.set_backend(backend)
    # Import stats module needed below
    from ember_ml.ops import stats
    in_features = 4
    out_features = 3
    # Using string 'relu' relies on get_activation in ops
    layer = modules.Dense(input_dim=in_features, units=out_features, activation='relu')
    


# --- NCP / AutoNCP Tests ---

def test_ncp_instantiation_shape(backend):
    """Tests basic NCP instantiation and forward pass shape."""
    ops.set_backend(backend)
    # Requires a NeuronMap
    neuron_map = modules.wiring.NCPMap(
        inter_neurons=8, command_neurons=4, motor_neurons=3, sensory_neurons=5, seed=42
    )
    ncp_module = modules.NCP(neuron_map=neuron_map)

    batch_size = 2
    input_size = neuron_map.input_size
    input_tensor = tensor.random_normal((batch_size, input_size))
    output = ncp_module(input_tensor)

    assert isinstance(output, tensor.EmberTensor), "NCP output is not EmberTensor"
    expected_shape = (batch_size, neuron_map.output_size)
    assert tensor.shape(output) == expected_shape, f"NCP output shape mismatch: expected {expected_shape}, got {tensor.shape(output)}"
    # Check parameters exist (number depends on NCP implementation details)
    assert len(list(ncp_module.parameters())) > 0, "NCP module has no parameters"

def test_autoncp_instantiation_shape(backend):
    """Tests basic AutoNCP instantiation and forward pass shape."""
    ops.set_backend(backend)
    units = 15
    output_size = 4
    input_size = 6 # AutoNCP infers input size at build time, but we need it for input tensor

    # AutoNCP creates the map internally
    autoncp_module = modules.AutoNCP(units=units, output_size=output_size, sparsity_level=0.5, seed=43)

    batch_size = 2
    input_tensor = tensor.random_normal((batch_size, input_size))
    output = autoncp_module(input_tensor)

    assert isinstance(output, tensor.EmberTensor), "AutoNCP output is not EmberTensor"
    expected_shape = (batch_size, output_size)
    assert tensor.shape(output) == expected_shape, f"AutoNCP output shape mismatch: expected {expected_shape}, got {tensor.shape(output)}"
    # Check parameters exist
    assert len(list(autoncp_module.parameters())) > 0, "AutoNCP module has no parameters"
    # Check internal map properties (assuming it's stored as self.neuron_map)
    assert hasattr(autoncp_module, 'neuron_map'), "AutoNCP does not have neuron_map attribute"
    assert autoncp_module.neuron_map.units == units, "AutoNCP internal map units mismatch"
    assert autoncp_module.neuron_map.output_size == output_size, "AutoNCP internal map output_size mismatch"


    batch_size = 2
    # Input designed to produce positive and negative outputs before activation
    input_tensor = tensor.convert_to_tensor([[-1.0, -0.5, 0.5, 1.0], [0.1, -0.1, 2.0, -2.0]])

    output = layer(input_tensor)

    assert tensor.shape(output) == (batch_size, out_features), "Dense with activation shape mismatch"
    # Check that all output values are non-negative due to ReLU
    min_val = tensor.item(stats.min(output))
    # Use ops.subtract for backend consistency
    threshold = ops.subtract(tensor.convert_to_tensor(0.0), tensor.convert_to_tensor(1e-7))
    assert min_val >= tensor.item(threshold), f"Dense with ReLU produced negative output: {min_val}"

# TODO: Add tests for NCP module basic functionality
# TODO: Add tests for AutoNCP module basic functionality
# TODO: Add tests for activation modules (ReLU, Tanh, etc.) likely in a separate file or integrated here