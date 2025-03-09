"""
Example of using neural_lib.nn with different backends.

This example demonstrates how to create and use neural network components
with different backends (NumPy, PyTorch, MLX).
"""

import numpy as np
import neural_lib as nl
import neural_lib.nn as nn

def create_model():
    """
    Create a simple neural network model.
    
    Returns:
        A sequential model with two linear layers and ReLU activation
    """
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

def train_step(model, x, y, learning_rate=0.01):
    """
    Perform a single training step.
    
    Args:
        model: Neural network model
        x: Input data
        y: Target data
        learning_rate: Learning rate for gradient descent
        
    Returns:
        Loss value
    """
    # Forward pass
    y_pred = model(x)
    
    # Compute loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(y_pred, y)
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass (not implemented yet)
    # loss.backward()
    
    # Update parameters (not implemented yet)
    # for param in model.parameters():
    #     param.data = param.data - learning_rate * param.grad
    
    return loss

def main():
    """Main function to demonstrate neural network components."""
    # Create random data
    x = nl.random_normal((32, 10))  # 32 samples, 10 features
    y = nl.random_normal((32, 1))   # 32 samples, 1 target
    
    # Try with different backends
    backends = ['numpy', 'torch', 'mlx']
    available_backends = []
    
    # Check which backends are available
    try:
        import numpy
        available_backends.append('numpy')
    except ImportError:
        pass
    
    try:
        import torch
        available_backends.append('torch')
    except ImportError:
        pass
    
    try:
        import mlx
        available_backends.append('mlx')
    except ImportError:
        pass
    
    print(f"Available backends: {', '.join(available_backends)}")
    
    for backend in available_backends:
        print(f"\n--- Using {backend} backend ---")
        
        # Set the backend
        nl.set_backend(backend)
        
        # Create model
        model = create_model()
        print(f"Model architecture:\n{model}")
        
        # Forward pass
        y_pred = model(x)
        print(f"Output shape: {nl.shape(y_pred)}")
        
        # Compute loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y)
        print(f"Loss: {loss}")
        
        # Training step (will be limited without backward pass)
        loss = train_step(model, x, y)
        print(f"Loss after training step: {loss}")

if __name__ == "__main__":
    main()