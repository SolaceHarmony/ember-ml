"""
Pytest tests for the Neural Circuit Policy (NCP) module.

This module contains pytest tests for the NCP and AutoNCP classes.
"""

import pytest
import numpy as np

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.wirings import NCPWiring, FullyConnectedWiring, RandomWiring
from ember_ml.nn.modules import NCP, AutoNCP

def test_ncp_initialization():
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
    
    # Check that the model has the correct attributes
    assert model.wiring.units == 15
    assert model.wiring.output_dim == 5
    assert model.wiring.input_dim == 15
    assert model.wiring.sparsity_level == 0.5
    assert model.wiring.seed == 42
    
    # Check that the model has the correct parameters
    assert model.kernel is not None
    assert model.recurrent_kernel is not None
    assert model.bias is not None
    
    # Check that the model has the correct masks
    assert model.input_mask is not None
    assert model.recurrent_mask is not None
    assert model.output_mask is not None

def test_ncp_forward():
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
    
    # Create input data
    inputs = tensor.convert_to_tensor(np.random.randn(10, 15))
    
    # Forward pass
    outputs = model(inputs)
    
    # Check that the output has the correct shape
    assert ops.shape(outputs) == (10, 5)

def test_ncp_reset_state():
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
    
    # Create input data
    inputs = tensor.convert_to_tensor(np.random.randn(10, 15))
    
    # Forward pass
    outputs1 = model(inputs)
    
    # Reset state
    model.reset_state()
    
    # Forward pass again
    outputs2 = model(inputs)
    
    # Check that the outputs are the same
    assert np.allclose(ops.to_numpy(outputs1), ops.to_numpy(outputs2))

def test_ncp_config():
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
    
    # Create a new model from config
    model2 = NCP.from_config(config)
    
    # Check that the models have the same attributes
    assert model.wiring.units == model2.wiring.units
    assert model.wiring.output_dim == model2.wiring.output_dim
    assert model.wiring.input_dim == model2.wiring.input_dim
    assert model.wiring.sparsity_level == model2.wiring.sparsity_level
    assert model.wiring.seed == model2.wiring.seed

def test_auto_ncp_initialization():
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
    
    # Check that the model has the correct attributes
    assert model.units == 20
    assert model.output_size == 5
    assert model.sparsity_level == 0.5
    assert model.seed == 42
    
    # Check that the model has the correct parameters
    assert model.kernel is not None
    assert model.recurrent_kernel is not None
    assert model.bias is not None
    
    # Check that the model has the correct masks
    assert model.input_mask is not None
    assert model.recurrent_mask is not None
    assert model.output_mask is not None

def test_auto_ncp_forward():
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
    
    # Create input data
    inputs = tensor.convert_to_tensor(np.random.randn(10, 20))
    
    # Forward pass
    outputs = model(inputs)
    
    # Check that the output has the correct shape
    assert ops.shape(outputs) == (10, 5)

def test_auto_ncp_config():
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
    
    # Create a new model from config
    model2 = AutoNCP.from_config(config)
    
    # Check that the models have the same attributes
    assert model.units == model2.units
    assert model.output_size == model2.output_size
    assert model.sparsity_level == model2.sparsity_level
    assert model.seed == model2.seed