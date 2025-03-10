"""
Test script for the RBM model.

This script tests the functionality of the RBM model
to ensure it works correctly with the ember_ml package.
"""
import numpy as np
from ember_ml import ops
from ember_ml.models.rbm import RestrictedBoltzmannMachine as RBM
from ember_ml.backend import get_device
from ember_ml.nn.backends.torch_backend import get_default_device

def test_rbm_initialization():
    """Test RBM initialization."""
    print("Testing RBM initialization...")
    
    # Get the default device
    device = get_device()
    
    # Create an RBM model
    visible_size = 10
    hidden_size = 5
    rbm = RBM(visible_size, hidden_size, device=device)
    
    # Check that the weights and biases exist and have the correct shape
    assert hasattr(rbm, 'weights'), "RBM should have weights"
    weights_shape = ops.shape(rbm.weights)
    assert weights_shape == (visible_size, hidden_size), f"Weights shape should be ({visible_size}, {hidden_size}), got {weights_shape}"
    
    assert hasattr(rbm, 'visible_bias'), "RBM should have visible bias"
    visible_bias_shape = ops.shape(rbm.visible_bias)
    assert visible_bias_shape == (visible_size,), f"Visible bias shape should be ({visible_size},), got {visible_bias_shape}"
    
    assert hasattr(rbm, 'hidden_bias'), "RBM should have hidden bias"
    hidden_bias_shape = ops.shape(rbm.hidden_bias)
    assert hidden_bias_shape == (hidden_size,), f"Hidden bias shape should be ({hidden_size},), got {hidden_bias_shape}"
    
    print("RBM initialization tests passed!")

def test_rbm_forward():
    """Test RBM forward pass."""
    print("Testing RBM forward pass...")
    
    # Get the default device
    device = get_device()
    
    # Create an RBM model
    visible_size = 10
    hidden_size = 5
    rbm = RBM(visible_size, hidden_size, device=device)
    
    # Create input data
    batch_size = 3
    x = ops.random_uniform((batch_size, visible_size), device=device)
    
    # Test forward pass
    h_prob, h_sample = rbm.forward(x)
    
    # Check shapes
    assert ops.shape(h_prob) == (batch_size, hidden_size), f"Hidden probabilities shape should be ({batch_size}, {hidden_size}), got {ops.shape(h_prob)}"
    assert ops.shape(h_sample) == (batch_size, hidden_size), f"Hidden samples shape should be ({batch_size}, {hidden_size}), got {ops.shape(h_sample)}"
    
    # Check values
    assert ops.all(h_prob >= 0) and ops.all(h_prob <= 1), "Hidden probabilities should be in [0, 1]"
    assert ops.all((h_sample == 0) | (h_sample == 1)), "Hidden samples should be binary (0 or 1)"
    
    print("RBM forward pass tests passed!")

def test_rbm_backward():
    """Test RBM backward pass."""
    print("Testing RBM backward pass...")
    
    # Get the default device
    device = get_device()
    
    # Create an RBM model
    visible_size = 10
    hidden_size = 5
    rbm = RBM(visible_size, hidden_size, device=device)
    
    # Create hidden data
    batch_size = 3
    h = ops.round(ops.random_uniform((batch_size, hidden_size), device=device))  # Binary hidden states
    
    # Test backward pass
    v_prob, v_sample = rbm.backward(h)
    
    # Check shapes
    assert ops.shape(v_prob) == (batch_size, visible_size), f"Visible probabilities shape should be ({batch_size}, {visible_size}), got {ops.shape(v_prob)}"
    assert ops.shape(v_sample) == (batch_size, visible_size), f"Visible samples shape should be ({batch_size}, {visible_size}), got {ops.shape(v_sample)}"
    
    # Check values
    assert ops.all(v_prob >= 0) and ops.all(v_prob <= 1), "Visible probabilities should be in [0, 1]"
    assert ops.all((v_sample == 0) | (v_sample == 1)), "Visible samples should be binary (0 or 1)"
    
    print("RBM backward pass tests passed!")

def test_rbm_reconstruction():
    """Test RBM reconstruction."""
    print("Testing RBM reconstruction...")
    
    # Get the default device
    device = get_device()
    
    # Create an RBM model
    visible_size = 10
    hidden_size = 5
    rbm = RBM(visible_size, hidden_size, device=device)
    
    # Create input data
    batch_size = 3
    x = ops.round(ops.random_uniform((batch_size, visible_size), device=device))  # Binary visible states
    
    # Test reconstruction
    h_prob, h_sample = rbm.forward(x)
    v_prob, v_sample = rbm.backward(h_sample)
    
    # Check shapes
    assert ops.shape(v_prob) == ops.shape(x), f"Reconstructed visible probabilities shape should match input shape"
    assert ops.shape(v_sample) == ops.shape(x), f"Reconstructed visible samples shape should match input shape"
    
    print("RBM reconstruction tests passed!")

def test_rbm_free_energy():
    """Test RBM free energy calculation."""
    print("Testing RBM free energy calculation...")
    
    # Get the default device
    device = get_device()
    
    # Create an RBM model
    visible_size = 10
    hidden_size = 5
    rbm = RBM(visible_size, hidden_size, device=device)
    
    # Create input data
    batch_size = 3
    x = ops.random_uniform((batch_size, visible_size), device=device)
    
    # Test free energy calculation
    free_energy = rbm.free_energy(x)
    
    # Check shape
    assert ops.shape(free_energy) == (batch_size,), f"Free energy shape should be ({batch_size},), got {ops.shape(free_energy)}"
    
    print("RBM free energy calculation tests passed!")

if __name__ == "__main__":
    print("Running tests for RBM model...")
    test_rbm_initialization()
    test_rbm_forward()
    test_rbm_backward()
    test_rbm_reconstruction()
    test_rbm_free_energy()
    print("All RBM tests passed!")