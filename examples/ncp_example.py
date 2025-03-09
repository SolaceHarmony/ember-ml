"""
Neural Circuit Policy (NCP) example.

This example demonstrates how to use the NCP and AutoNCP classes
to create and train a neural circuit policy.
"""

import numpy as np
import matplotlib.pyplot as plt

from ember_ml import ops
from ember_ml.nn.wirings import NCPWiring, FullyConnectedWiring, RandomWiring
from ember_ml.nn.modules import NCP, AutoNCP

def main():
    """Run the NCP example."""
    print("Neural Circuit Policy (NCP) Example")
    print("===================================")
    
    # Create a simple dataset
    print("\nCreating dataset...")
    X = ops.reshape(ops.linspace(0, 2 * np.pi, 100), (-1, 1))
    y = ops.sin(X)
    
    # Convert to numpy for splitting
    X_np = ops.to_numpy(X)
    y_np = ops.to_numpy(y)
    
    # Split into train and test sets
    X_train, X_test = X_np[:80], X_np[80:]
    y_train, y_test = y_np[:80], y_np[80:]
    
    # Create a wiring configuration
    print("\nCreating wiring configuration...")
    wiring = NCPWiring(
        inter_neurons=10,
        motor_neurons=1,
        sensory_neurons=0,
        sparsity_level=0.5,
        seed=42
    )
    
    # Create an NCP model
    print("\nCreating NCP model...")
    model = NCP(
        wiring=wiring,
        activation="tanh",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros"
    )
    
    # Train the model
    print("\nTraining NCP model...")
    learning_rate = 0.01
    epochs = 10  # Reduced from 100 to 10 for a smoke test
    batch_size = 16
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Shuffle the data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Train in batches
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass
            model.reset_state()
            y_pred = model(ops.convert_to_tensor(X_batch))
            
            # Compute loss
            loss = ops.mean(ops.square(y_pred - ops.convert_to_tensor(y_batch)))
            
            # Compute gradients
            params = list(model.parameters())
            grads = ops.gradients(loss, params)
            
            # Update parameters
            for param, grad in zip(params, grads):
                param.data = ops.subtract(param.data, ops.multiply(ops.convert_to_tensor(learning_rate), grad))
            
            epoch_loss += ops.to_numpy(loss)
        
        epoch_loss /= (len(X_train) // batch_size)
        losses.append(epoch_loss)
        
        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.6f}")
    
    # Evaluate the model
    print("\nEvaluating NCP model...")
    model.reset_state()
    y_pred = ops.to_numpy(model(ops.convert_to_tensor(X_test)))
    test_loss = np.mean(np.square(y_pred - y_test))
    print(f"Test Loss: {test_loss:.6f}")
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Plot the loss
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    # Plot the predictions
    plt.subplot(2, 1, 2)
    plt.plot(X_test, y_test, label="True")
    plt.plot(X_test, y_pred, label="Predicted")
    plt.title("Predictions")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("ncp_example.png")
    plt.show()
    
    # Create an AutoNCP model
    print("\nCreating AutoNCP model...")
    auto_model = AutoNCP(
        units=20,
        output_size=1,
        sparsity_level=0.5,
        seed=42,
        activation="tanh",
        use_bias=True
    )
    
    print("\nAutoNCP model created successfully!")
    print(f"Units: {auto_model.units}")
    print(f"Output size: {auto_model.output_size}")
    print(f"Sparsity level: {auto_model.sparsity_level}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()