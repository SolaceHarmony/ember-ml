"""
Example of using Ember ML with the PyTorch backend.

This example demonstrates how to create and use neural network components
with the PyTorch backend.
"""

import ember_ml as eh
import ember_ml.nn as nn

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
    # Set the backend to PyTorch
    try:
        eh.set_backend('torch')
        print("Using PyTorch backend")
    except ImportError:
        print("PyTorch is not available")
        return
    
    # Create random data
    x = eh.random_normal((32, 10))  # 32 samples, 10 features
    y = eh.random_normal((32, 1))   # 32 samples, 1 target
    
    # Create model
    model = create_model()
    print(f"Model architecture:\n{model}")
    
    # Forward pass
    y_pred = model(x)
    print(f"Output shape: {eh.shape(y_pred)}")
    
    # Compute loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(y_pred, y)
    print(f"Loss: {loss}")
    
    # Training step (will be limited without backward pass)
    loss = train_step(model, x, y)
    print(f"Loss after training step: {loss}")

if __name__ == "__main__":
    main()