"""
Tests for LQNet and CTRQNet modules.

This module contains tests for the Liquid Quantum Neural Network (LQNet)
and Continuous-Time Recurrent Quantum Neural Network (CTRQNet) modules.
"""

import pytest
import numpy as np

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.wiring import NCPMap
from ember_ml.nn.modules.rnn import LQNet, CTRQNet


# Fixtures for test setup
@pytest.fixture
def test_data():
    """Create test data for LQNet and CTRQNet tests."""
    batch_size = 8
    seq_length = 10
    input_dim = 3
    hidden_dim = 16
    
    # Create input tensor
    inputs = tensor.random_normal(
        (batch_size, seq_length, input_dim)
    )
    
    # Create neuron map
    neuron_map = NCPMap(
        inter_neurons=hidden_dim // 2,
        command_neurons=hidden_dim // 4,
        motor_neurons=hidden_dim // 4,
        sensory_neurons=input_dim,
        seed=42
    )
    
    # Get the actual number of units from the neuron map
    # Let's print the neuron_map.units to see what's actually being used
    neuron_map.build(input_dim)  # Ensure the neuron map is built
    actual_units = neuron_map.units
    
    return {
        'batch_size': batch_size,
        'seq_length': seq_length,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'inputs': inputs,
        'neuron_map': neuron_map,
        'actual_units': actual_units
    }


# LQNet Tests
def test_lqnet_forward(test_data):
    """Test forward pass of LQNet."""
    # Create LQNet model
    lqnet = LQNet(
        neuron_map=test_data['neuron_map'],
        nu_0=1.0,
        beta=0.1,
        noise_scale=0.05,
        return_sequences=True,
        return_state=False,
        batch_first=True
    )
    
    # Forward pass
    outputs = lqnet(test_data['inputs'])
    
    # Check output shape
    assert tensor.shape(outputs) == (
        test_data['batch_size'], 
        test_data['seq_length'], 
        test_data['actual_units']
    )
    
    # Check output is not None or all zeros
    assert outputs is not None
    # Check that outputs are not all zeros
    assert not ops.allclose(outputs, tensor.zeros_like(outputs))


def test_lqnet_return_state(test_data):
    """Test LQNet with return_state=True."""
    # Create LQNet model with return_state=True
    lqnet = LQNet(
        neuron_map=test_data['neuron_map'],
        nu_0=1.0,
        beta=0.1,
        noise_scale=0.05,
        return_sequences=True,
        return_state=True,
        batch_first=True
    )
    
    # Forward pass
    outputs, state = lqnet(test_data['inputs'])
    
    # Check output shape
    assert tensor.shape(outputs) == (
        test_data['batch_size'], 
        test_data['seq_length'], 
        test_data['actual_units']
    )
    
    # Check state is a tuple of (h_c, h_s, t)
    assert isinstance(state, tuple)
    assert len(state) == 3
    
    # Check state shapes
    h_c, h_s, t = state
    assert tensor.shape(h_c) == (test_data['batch_size'], test_data['actual_units'])
    assert tensor.shape(h_s) == (test_data['batch_size'], test_data['actual_units'])
    assert tensor.shape(t) == (test_data['batch_size'], 1)


def test_lqnet_no_sequences(test_data):
    """Test LQNet with return_sequences=False."""
    # Create LQNet model with return_sequences=False
    lqnet = LQNet(
        neuron_map=test_data['neuron_map'],
        nu_0=1.0,
        beta=0.1,
        noise_scale=0.05,
        return_sequences=False,
        return_state=False,
        batch_first=True
    )
    
    # Forward pass
    outputs = lqnet(test_data['inputs'])
    
    # Check output shape (should be last output only)
    assert tensor.shape(outputs) == (test_data['batch_size'], test_data['actual_units'])


# CTRQNet Tests
def test_ctrqnet_forward(test_data):
    """Test forward pass of CTRQNet."""
    # Create CTRQNet model
    ctrqnet = CTRQNet(
        neuron_map=test_data['neuron_map'],
        nu_0=1.0,
        beta=0.1,
        noise_scale=0.05,
        time_scale_factor=1.0,
        use_harmonic_embedding=True,
        return_sequences=True,
        return_state=False,
        batch_first=True
    )
    
    # Forward pass
    outputs = ctrqnet(test_data['inputs'])
    
    # Check output shape
    assert tensor.shape(outputs) == (
        test_data['batch_size'], 
        test_data['seq_length'], 
        test_data['actual_units']
    )
    
    # Check output is not None or all zeros
    assert outputs is not None
    # Check that outputs are not all zeros
    assert not ops.allclose(outputs, tensor.zeros_like(outputs))


def test_ctrqnet_return_state(test_data):
    """Test CTRQNet with return_state=True."""
    # Create CTRQNet model with return_state=True
    ctrqnet = CTRQNet(
        neuron_map=test_data['neuron_map'],
        nu_0=1.0,
        beta=0.1,
        noise_scale=0.05,
        time_scale_factor=1.0,
        use_harmonic_embedding=True,
        return_sequences=True,
        return_state=True,
        batch_first=True
    )
    
    # Forward pass
    outputs, state = ctrqnet(test_data['inputs'])
    
    # Check output shape
    assert tensor.shape(outputs) == (
        test_data['batch_size'], 
        test_data['seq_length'], 
        test_data['actual_units']
    )
    
    # Check state is a tuple of (h_c, h_s, t)
    assert isinstance(state, tuple)
    assert len(state) == 3
    
    # Check state shapes
    h_c, h_s, t = state
    assert tensor.shape(h_c) == (test_data['batch_size'], test_data['actual_units'])
    assert tensor.shape(h_s) == (test_data['batch_size'], test_data['actual_units'])
    assert tensor.shape(t) == (test_data['batch_size'], 1)


def test_ctrqnet_no_harmonic(test_data):
    """Test CTRQNet without harmonic embedding."""
    # Create CTRQNet model without harmonic embedding
    ctrqnet = CTRQNet(
        neuron_map=test_data['neuron_map'],
        nu_0=1.0,
        beta=0.1,
        noise_scale=0.05,
        time_scale_factor=1.0,
        use_harmonic_embedding=False,
        return_sequences=True,
        return_state=False,
        batch_first=True
    )
    
    # Forward pass
    outputs = ctrqnet(test_data['inputs'])
    
    # Check output shape
    assert tensor.shape(outputs) == (
        test_data['batch_size'], 
        test_data['seq_length'], 
        test_data['actual_units']
    )
    
    # Check output is not None or all zeros
    assert outputs is not None
    # Check that outputs are not all zeros
    assert not ops.allclose(outputs, tensor.zeros_like(outputs))