"""
Tests for the Neural Circuit Policy (NCP) module.

This module tests the NCP and AutoNCP classes to ensure they work correctly.
"""

import pytest
import numpy as np
from ember_ml.backend import set_backend
from ember_ml import ops
from ember_ml.nn.modules import NCP, AutoNCP
from ember_ml.nn.wirings.ncp_wiring import NCPWiring

# Set the backend to numpy for testing
@pytest.fixture(autouse=True)
def use_numpy_backend():
    """Use numpy backend for testing."""
    original_backend = ops.get_ops()
    set_backend('numpy')
    yield
    set_backend(original_backend)

# Skip all tests in this file for now
pytestmark = pytest.mark.skip("Skipping NCP tests until implementation is complete")

class TestNCP:
    """Test cases for the NCP class."""

    def test_ncp_initialization(self):
        """Test NCP initialization."""
        # Create a wiring configuration
        wiring = NCPWiring(
            inter_neurons=10,
            motor_neurons=5,
            sensory_neurons=0,
            sparsity_level=0.5,
            seed=42
        )
        
        # Create an NCP model
        model = NCP(
            wiring=wiring,
            activation="tanh",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            bias_initializer="zeros"
        )
        
        # Check that the model was initialized correctly
        assert model.wiring.units == 15
        assert model.wiring.output_dim == 5
        assert model.activation_name == "tanh"
        assert model.use_bias == True
        assert model.kernel_initializer == "glorot_uniform"
        assert model.recurrent_initializer == "orthogonal"
        assert model.bias_initializer == "zeros"
        
        # Check that the masks were created correctly
        assert model.input_mask is not None
        assert model.recurrent_mask is not None
        assert model.output_mask is not None

    def test_ncp_forward(self):
        """Test NCP forward pass."""
        # Create a wiring configuration
        wiring = NCPWiring(
            inter_neurons=10,
            motor_neurons=5,
            sensory_neurons=0,
            sparsity_level=0.5,
            seed=42
        )
        
        # Create an NCP model
        model = NCP(
            wiring=wiring,
            activation="tanh",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            bias_initializer="zeros"
        )
        
        # Create input tensor
        x = ops.random_normal(shape=(32, 15))
        
        # Forward pass
        output = model(x, return_state=False)
        
        # Check output shape
        assert output.shape == (32, 5)

    def test_ncp_reset_state(self):
        """Test NCP reset_state method."""
        # Create a wiring configuration
        wiring = NCPWiring(
            inter_neurons=10,
            motor_neurons=5,
            sensory_neurons=0,
            sparsity_level=0.5,
            seed=42
        )
        
        # Create an NCP model
        model = NCP(
            wiring=wiring,
            activation="tanh",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            bias_initializer="zeros"
        )
        
        # Create input tensor
        x = ops.random_normal(shape=(32, 15))
        
        # Forward pass
        output = model(x, return_state=False)
        
        # Reset state
        model.reset_state()
        
        # Check that the state is None
        assert model.state is None

    def test_ncp_config(self):
        """Test NCP get_config and from_config methods."""
        # Create a wiring configuration
        wiring = NCPWiring(
            inter_neurons=10,
            motor_neurons=5,
            sensory_neurons=0,
            sparsity_level=0.5,
            seed=42
        )
        
        # Create an NCP model
        model = NCP(
            wiring=wiring,
            activation="tanh",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            bias_initializer="zeros"
        )
        
        # Get config
        config = model.get_config()
        
        # Check that the config contains the expected keys
        assert "wiring" in config
        assert "activation" in config
        assert "use_bias" in config
        assert "kernel_initializer" in config
        assert "recurrent_initializer" in config
        assert "bias_initializer" in config
        
        # Create a new model from the config
        new_model = NCP.from_config(config)
        
        # Check that the new model has the same attributes
        assert new_model.wiring.units == model.wiring.units
        assert new_model.wiring.output_dim == model.wiring.output_dim
        assert new_model.activation_name == model.activation_name
        assert new_model.use_bias == model.use_bias
        assert new_model.kernel_initializer == model.kernel_initializer
        assert new_model.recurrent_initializer == model.recurrent_initializer
        assert new_model.bias_initializer == model.bias_initializer


class TestAutoNCP:
    """Test cases for the AutoNCP class."""

    def test_auto_ncp_initialization(self):
        """Test AutoNCP initialization."""
        # Create an AutoNCP model
        model = AutoNCP(
            units=20,
            output_size=5,
            sparsity_level=0.5,
            seed=42,
            activation="tanh",
            use_bias=True
        )
        
        # Check that the model was initialized correctly
        assert model.units == 20
        assert model.output_size == 5
        assert model.sparsity_level == 0.5
        assert model.seed == 42
        assert model.activation_name == "tanh"
        assert model.use_bias == True
        
        # Check that the wiring was created correctly
        assert model.wiring is not None
        # The wiring units will be different from the model units
        # because the AutoNCP class calculates the number of inter and command neurons
        # based on the units and output_size
        assert model.wiring.output_dim == 5

    def test_auto_ncp_forward(self):
        """Test AutoNCP forward pass."""
        # Create an AutoNCP model
        model = AutoNCP(
            units=20,
            output_size=5,
            sparsity_level=0.5,
            seed=42,
            activation="tanh",
            use_bias=True
        )
        
        # Create input tensor with the same shape as the wiring input_dim
        x = ops.random_normal(shape=(32, model.wiring.input_dim))
        
        # Forward pass
        output = model(x, return_state=False)
        
        # Check output shape
        assert output.shape == (32, 5)

    def test_auto_ncp_config(self):
        """Test AutoNCP get_config and from_config methods."""
        # Create an AutoNCP model
        model = AutoNCP(
            units=20,
            output_size=5,
            sparsity_level=0.5,
            seed=42,
            activation="tanh",
            use_bias=True
        )
        
        # Get config
        config = model.get_config()
        
        # Check that the config contains the expected keys
        assert "units" in config
        assert "output_size" in config
        assert "sparsity_level" in config
        assert "seed" in config
        assert "activation" in config
        assert "use_bias" in config
        
        # Create a new model from the config
        new_model = AutoNCP.from_config(config)
        
        # Check that the new model has the same attributes
        assert new_model.units == model.units
        assert new_model.output_size == model.output_size
        assert new_model.sparsity_level == model.sparsity_level
        assert new_model.seed == model.seed
        assert new_model.activation_name == model.activation_name
        assert new_model.use_bias == model.use_bias