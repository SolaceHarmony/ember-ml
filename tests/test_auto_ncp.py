import pytest
from ember_ml import ops
from ember_ml.backend import set_backend, get_backend
from ember_ml.nn.wirings import AutoNCP  # Import from wirings, not modules
import ember_ml.nn.tensor as tensor

@pytest.fixture(params=['numpy', 'torch', 'mlx'])
def backend_name(request):
    """Fixture to test with different backends."""
    return request.param

def test_auto_ncp(backend_name):
    """Test AutoNCP with different backends."""
    # Set the backend
    set_backend(backend_name)
    
    # Create an AutoNCP wiring
    auto_ncp_wiring = AutoNCP(
        units=15,
        output_size=5,
        sparsity_level=0.5
    )
    
    # Create a CfC cell with the AutoNCP wiring
    from ember_ml.nn.modules.rnn import WiredCfCCell
    cell = WiredCfCCell(
        input_size=15,
        wiring=auto_ncp_wiring,
        activation="tanh",
        use_bias=True
    )
    
    # Create input
    batch_size = 2
    inputs = tensor.random_normal((batch_size, 15))
    
    # Initial state
    state = tensor.zeros((batch_size, 15))
    
    # Forward pass
    output, new_state = cell(inputs, state)
    
    # Check output shape
    assert tensor.shape(output) == (batch_size, 5)