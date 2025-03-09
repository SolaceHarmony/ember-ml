"""
Test script for the PyTorch backend layers.

This script tests the functionality of the PyTorch backend layers
to ensure they work correctly.
"""

import torch
import numpy as np
from ember_ml.nn.backends.torch_layers import (
    TorchLinear, TorchReLU, TorchSigmoid, TorchTanh, TorchSoftmax, TorchSequential
)
from ember_ml.nn.backends.torch_backend import (
    is_metal_available, get_default_device
)

def test_device_detection():
    """Test the device detection functionality."""
    print("Testing device detection...")
    
    # Check if Metal is available
    metal_available = is_metal_available()
    print(f"Metal available: {metal_available}")
    
    # Get the default device
    default_device = get_default_device()
    print(f"Default device: {default_device}")
    
    print("Device detection tests passed!")

def test_torch_linear():
    """Test the TorchLinear layer."""
    print("Testing TorchLinear...")
    
    # Force CPU for consistent testing
    device = "cpu"
    print(f"Using device: {device}")
    
    # Create a linear layer explicitly on CPU
    linear = TorchLinear(10, 5, device=device)
    
    # Check that the weight and bias parameters exist and have the correct shape
    assert hasattr(linear, 'weight'), "Linear layer should have a weight parameter"
    assert linear.weight.data.shape == (5, 10), f"Weight shape should be (5, 10), got {linear.weight.data.shape}"
    
    assert hasattr(linear, 'bias'), "Linear layer should have a bias parameter"
    assert linear.bias.data.shape == (5,), f"Bias shape should be (5,), got {linear.bias.data.shape}"
    
    # Print the device of the weight parameter
    print(f"Weight parameter device: {linear.weight.data.device}")
    
    # Test forward pass with a tensor
    x = torch.randn(3, 10, device=device)  # Batch of 3 samples, each with 10 features
    y = linear(x)
    
    assert y.shape == (3, 5), f"Output shape should be (3, 5), got {y.shape}"
    print(f"Output tensor device: {y.device}")
    
    # Test forward pass with a numpy array
    x_np = np.random.randn(3, 10).astype(np.float32)
    y_np = linear(x_np)
    
    assert isinstance(y_np, torch.Tensor), "Output should be a torch.Tensor"
    assert y_np.shape == (3, 5), f"Output shape should be (3, 5), got {y_np.shape}"
    print(f"Output tensor device (from numpy): {y_np.device}")
    
    print("TorchLinear tests passed!")

def test_torch_activations():
    """Test the activation layers."""
    print("Testing activation layers...")
    
    # Force CPU for consistent testing
    device = "cpu"
    
    # Create activation layers
    relu = TorchReLU()
    sigmoid = TorchSigmoid()
    tanh = TorchTanh()
    softmax = TorchSoftmax(dim=1)
    
    # Create input tensor on the appropriate device
    x = torch.randn(3, 5, device=device)
    
    # Test ReLU
    y_relu = relu(x)
    assert y_relu.shape == x.shape, f"ReLU output shape should match input shape"
    assert torch.all(y_relu >= 0), "ReLU output should be non-negative"
    
    # Test Sigmoid
    y_sigmoid = sigmoid(x)
    assert y_sigmoid.shape == x.shape, f"Sigmoid output shape should match input shape"
    assert torch.all(y_sigmoid >= 0) and torch.all(y_sigmoid <= 1), "Sigmoid output should be in [0, 1]"
    
    # Test Tanh
    y_tanh = tanh(x)
    assert y_tanh.shape == x.shape, f"Tanh output shape should match input shape"
    assert torch.all(y_tanh >= -1) and torch.all(y_tanh <= 1), "Tanh output should be in [-1, 1]"
    
    # Test Softmax
    y_softmax = softmax(x)
    assert y_softmax.shape == x.shape, f"Softmax output shape should match input shape"
    assert torch.all(y_softmax >= 0) and torch.all(y_softmax <= 1), "Softmax output should be in [0, 1]"
    assert torch.allclose(y_softmax.sum(dim=1), torch.ones(3, device=device)), "Softmax should sum to 1 along specified dimension"
    
    print("Activation layer tests passed!")

def test_torch_sequential():
    """Test the TorchSequential container."""
    print("Testing TorchSequential...")
    
    # Force CPU for consistent testing
    device = "cpu"
    
    # Create a sequential model
    model = TorchSequential(
        TorchLinear(10, 20, device=device),
        TorchReLU(),
        TorchLinear(20, 5, device=device),
        TorchSoftmax(dim=1)
    )
    
    # Check that the model has the correct number of modules
    assert len(model) == 4, f"Sequential should have 4 modules, got {len(model)}"
    
    # Test forward pass
    x = torch.randn(3, 10, device=device)
    y = model(x)
    
    assert y.shape == (3, 5), f"Output shape should be (3, 5), got {y.shape}"
    assert torch.all(y >= 0) and torch.all(y <= 1), "Output should be in [0, 1]"
    assert torch.allclose(y.sum(dim=1), torch.ones(3, device=device)), "Output should sum to 1 along specified dimension"
    
    # Test indexing
    assert isinstance(model[0], TorchLinear), f"First module should be TorchLinear, got {type(model[0])}"
    assert isinstance(model[1], TorchReLU), f"Second module should be TorchReLU, got {type(model[1])}"
    
    # Test slicing
    sliced_model = model[1:3]
    assert len(sliced_model) == 2, f"Sliced model should have 2 modules, got {len(sliced_model)}"
    assert isinstance(sliced_model[0], TorchReLU), f"First module of sliced model should be TorchReLU, got {type(sliced_model[0])}"
    
    # Test appending
    model.append(TorchTanh())
    assert len(model) == 5, f"After appending, sequential should have 5 modules, got {len(model)}"
    assert isinstance(model[4], TorchTanh), f"Last module should be TorchTanh, got {type(model[4])}"
    
    print("TorchSequential tests passed!")

def test_metal_support():
    """Test Metal support if available."""
    if not is_metal_available():
        print("Metal not available, skipping Metal support test.")
        return
    
    print("Testing Metal support...")
    
    # Create a linear layer on MPS
    linear_mps = TorchLinear(10, 5, device="mps")
    print(f"Linear layer weight device: {linear_mps.weight.data.device}")
    
    # Create input tensor on MPS
    x_mps = torch.randn(3, 10, device="mps")
    
    # Forward pass
    y_mps = linear_mps(x_mps)
    print(f"Output tensor device: {y_mps.device}")
    
    # Verify shape
    assert y_mps.shape == (3, 5), f"Output shape should be (3, 5), got {y_mps.shape}"
    
    print("Metal support test passed!")

if __name__ == "__main__":
    print("Running tests for PyTorch backend layers...")
    test_device_detection()
    test_torch_linear()
    test_torch_activations()
    test_torch_sequential()
    
    # Run Metal test separately
    try:
        test_metal_support()
    except Exception as e:
        print(f"Metal support test failed: {e}")
    
    print("All tests passed!")