"""
Gated Recurrent Unit (GRU) Neural Network Example.

This example demonstrates how to use the GRU and GRUCell classes
to create and train a recurrent neural network for sequence prediction.
"""

import matplotlib.pyplot as plt

from ember_ml import ops
from ember_ml.nn.modules.rnn import GRU, GRUCell
from ember_ml.nn import Module, Sequential, tensor
from ember_ml.training import Optimizer, Loss

def generate_sine_wave_data(num_samples=1000, seq_length=100, num_features=1):
    """Generate sine wave data for sequence prediction."""
    # Generate time points
    t = ops.linspace(0, 2 * ops.pi, seq_length)
    
    # Generate sine waves with random phase shifts
    X = ops.zeros((num_samples, seq_length, num_features))
    y = ops.zeros((num_samples, seq_length, num_features))
    
    for i in range(num_samples):
        # Random phase shift
        phase_shift = ops.random_uniform(0, 2 * ops.pi)
        
        # Generate sine wave with phase shift
        signal = ops.sin(t + phase_shift)
        
        # Add some noise
        noise = ops.random_normal(0, 0.1, seq_length)
        noisy_signal = ops.add(signal, noise)
        
        # Store input and target
        X = ops.tensor_scatter_nd_update(
            X,
            ops.stack([
                ops.ones((seq_length,), dtype=ops.int32) * i,
                ops.arange(seq_length),
                ops.zeros((seq_length,), dtype=ops.int32)
            ], axis=1),
            noisy_signal
        )
        
        y = ops.tensor_scatter_nd_update(
            y,
            ops.stack([
                ops.ones((seq_length,), dtype=ops.int32) * i,
                ops.arange(seq_length),
                ops.zeros((seq_length,), dtype=ops.int32)
            ], axis=1),
            signal
        )
    
    return X, y

def train_gru_model(model, X_train, y_train, epochs=50, batch_size=32, learning_rate=0.001):
    """Train a GRU model."""
    # Convert data to tensors
    X_train_tensor = tensor.convert_to_tensor(X_train, dtype=ops.float32)
    y_train_tensor = tensor.convert_to_tensor(y_train, dtype=ops.float32)
    
    # Define optimizer and loss function
    optimizer = Optimizer.adam(model.parameters(), learning_rate=learning_rate)
    loss_fn = Loss.mse()
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Shuffle the data
        indices = ops.random_permutation(ops.shape(X_train)[0])
        shuffled_X = ops.gather(X_train_tensor, indices)
        shuffled_y = ops.gather(y_train_tensor, indices)
        
        # Train in batches
        for i in range(0, ops.shape(X_train)[0], batch_size):
            batch_X = shuffled_X[i:i+batch_size]
            batch_y = shuffled_y[i:i+batch_size]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            
            # Compute loss
            loss = loss_fn(outputs, batch_y)
            
            # Backward pass and optimize
            grads = ops.gradients(loss, model.parameters())
            optimizer.step(grads)
            
            epoch_loss += ops.to_numpy(loss)
        
        # Print progress
        avg_loss = epoch_loss / (ops.shape(X_train)[0] // batch_size)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses

def evaluate_model(model, X_test, y_test):
    """Evaluate a GRU model."""
    # Convert data to tensors
    X_test_tensor = tensor.convert_to_tensor(X_test, dtype=ops.float32)
    y_test_tensor = tensor.convert_to_tensor(y_test, dtype=ops.float32)
    
    # Make predictions
    y_pred = model(X_test_tensor)
    
    # Compute loss
    loss_fn = Loss.mse()
    loss = loss_fn(y_pred, y_test_tensor)
    
    # Convert predictions to numpy
    y_pred_np = ops.to_numpy(y_pred)
    
    return ops.to_numpy(loss), y_pred_np

def main():
    """Run the GRU example."""
    print("Gated Recurrent Unit (GRU) Neural Network Example")
    print("================================================")
    
    # Generate data
    print("\nGenerating data...")
    X, y = generate_sine_wave_data(num_samples=1000, seq_length=100, num_features=1)
    
    # Split data into train and test sets
    train_size = int(0.8 * ops.shape(X)[0])
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Train data shape: {ops.shape(X_train)}")
    print(f"Test data shape: {ops.shape(X_test)}")
    
    # Create a single-layer GRU model
    print("\nCreating single-layer GRU model...")
    single_layer_model = Sequential([
        GRU(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            return_sequences=True
        )
    ])
    
    # Train the single-layer model
    print("\nTraining single-layer GRU model...")
    single_layer_losses = train_gru_model(
        single_layer_model,
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Evaluate the single-layer model
    print("\nEvaluating single-layer GRU model...")
    single_layer_loss, single_layer_preds = evaluate_model(single_layer_model, X_test, y_test)
    print(f"Single-layer GRU Test Loss: {single_layer_loss:.6f}")
    
    # Create a multi-layer bidirectional GRU model
    print("\nCreating multi-layer bidirectional GRU model...")
    multi_layer_model = Sequential([
        GRU(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            return_sequences=True
        )
    ])
    
    # Train the multi-layer model
    print("\nTraining multi-layer bidirectional GRU model...")
    multi_layer_losses = train_gru_model(
        multi_layer_model,
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Evaluate the multi-layer model
    print("\nEvaluating multi-layer bidirectional GRU model...")
    multi_layer_loss, multi_layer_preds = evaluate_model(multi_layer_model, X_test, y_test)
    print(f"Multi-layer Bidirectional GRU Test Loss: {multi_layer_loss:.6f}")
    
    # Convert to numpy for visualization
    X_test_np = ops.to_numpy(X_test)
    y_test_np = ops.to_numpy(y_test)
    single_layer_losses_np = ops.to_numpy(tensor.convert_to_tensor(single_layer_losses))
    multi_layer_losses_np = ops.to_numpy(tensor.convert_to_tensor(multi_layer_losses))
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    # Plot the training losses
    plt.subplot(2, 1, 1)
    plt.plot(single_layer_losses_np, label='Single-layer GRU')
    plt.plot(multi_layer_losses_np, label='Multi-layer Bidirectional GRU')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot the predictions for a sample
    plt.subplot(2, 1, 2)
    sample_idx = 0
    plt.plot(X_test_np[sample_idx, :, 0], 'b-', label='Input (Noisy)')
    plt.plot(y_test_np[sample_idx, :, 0], 'g-', label='Target')
    plt.plot(single_layer_preds[sample_idx, :, 0], 'r--', label='Single-layer GRU')
    plt.plot(multi_layer_preds[sample_idx, :, 0], 'm--', label='Multi-layer Bidirectional GRU')
    plt.title('Predictions')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('gru_example.png')
    plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()