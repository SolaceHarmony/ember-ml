"""
Binary Wave Neural Network for MNIST.

This script demonstrates how to use binary wave neural networks
to classify MNIST digits.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union, Dict, Any

from mlx_binary_wave import MLXBinaryWave

# Try to import MNIST dataset
try:
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    print("scikit-learn not available. Using random data for demonstration.")
    SKLEARN_AVAILABLE = False

class BinaryWaveMNISTClassifier:
    """
    Binary Wave Neural Network for MNIST classification.
    
    This network uses binary wave operations to classify MNIST digits.
    """
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, output_dim: int = 10):
        """
        Initialize the classifier.
        
        Args:
            input_dim: Input dimension (784 for MNIST)
            hidden_dim: Hidden dimension
            output_dim: Output dimension (10 for MNIST)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize weights
        self.w1 = tensor.convert_to_tensor(mx.random.uniform(shape=(input_dim, hidden_dim)) < 0.5, dtype=mx.int32)
        self.w2 = tensor.convert_to_tensor(mx.random.uniform(shape=(hidden_dim, output_dim)) < 0.5, dtype=mx.int32)
        
        # Initialize biases
        self.b1 = tensor.convert_to_tensor(mx.random.uniform(shape=(hidden_dim,)) < 0.5, dtype=mx.int32)
        self.b2 = tensor.convert_to_tensor(mx.random.uniform(shape=(output_dim,)) < 0.5, dtype=mx.int32)
    
    def forward(self, x: TensorLike) -> TensorLike:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Output tensor (batch_size, output_dim)
        """
        # Binarize input
        x_binary = TensorLike(x >= 0.5, dtype=mx.int32)
        
        # First layer
        h1 = self._binary_layer(x_binary, self.w1, self.b1)
        
        # Second layer
        output = self._binary_layer(h1, self.w2, self.b2)
        
        return output
    
    def _binary_layer(self, x: TensorLike, w: mx.array, b: mx.array) -> mx.array:
        """
        Binary layer computation.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            w: Weight tensor (input_dim, output_dim)
            b: Bias tensor (output_dim,)
            
        Returns:
            Output tensor (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        output_dim = w.shape[1]
        
        # Initialize output
        output = mx.zeros((batch_size, output_dim), dtype=mx.int32)
        
        # For each output neuron
        for j in range(output_dim):
            # For each input feature
            for i in range(x.shape[1]):
                # Get input and weight
                x_i = x[:, i]
                w_ij = w[i, j]
                
                # Compute contribution: x_i AND w_ij
                contribution = MLXBinaryWave.bitwise_and(x_i, mx.array([w_ij] * batch_size))
                
                # Add contribution to output
                output = mx.array([
                    *output[:, :j],
                    MLXBinaryWave.bitwise_xor(output[:, j], contribution),
                    *output[:, j+1:]
                ])
            
            # Add bias
            output = mx.array([
                *output[:, :j],
                MLXBinaryWave.bitwise_xor(output[:, j], mx.array([b[j]] * batch_size)),
                *output[:, j+1:]
            ])
        
        return output
    
    def predict(self, x: mx.array) -> mx.array:
        """
        Predict class labels.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Predicted class labels (batch_size,)
        """
        # Forward pass
        output = self.forward(x)
        
        # Count ones in each output
        counts = mx.zeros((output.shape[0], output.shape[1]), dtype=mx.int32)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                counts = mx.array([
                    *counts[:i],
                    [*counts[i, :j], MLXBinaryWave.count_ones(output[i, j]), *counts[i, j+1:]],
                    *counts[i+1:]
                ])
        
        # Return class with highest count
        return mx.argmax(counts, axis=1)
    
    def evaluate(self, x: mx.array, y: mx.array) -> float:
        """
        Evaluate the classifier.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            y: Target tensor (batch_size,)
            
        Returns:
            Accuracy
        """
        # Predict
        y_pred = self.predict(x)
        
        # Calculate accuracy
        correct = mx.sum(y_pred == y)
        total = len(y)
        
        return float(correct) / total

def load_mnist() -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """
    Load MNIST dataset.
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    if SKLEARN_AVAILABLE:
        # Load MNIST from scikit-learn
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X = mnist.data.astype('float32') / 255.0
        y = mnist.target.astype('int32')
        
        # Split into train and test
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to MLX arrays
        x_train = mx.array(x_train)
        y_train = mx.array(y_train)
        x_test = mx.array(x_test)
        y_test = mx.array(y_test)
    else:
        # Create random data for demonstration
        x_train = mx.array(mx.random.uniform(shape=(1000, 784)))
        y_train = mx.array(mx.random.randint(0, 10, shape=(1000,)))
        x_test = mx.array(mx.random.uniform(shape=(200, 784)))
        y_test = mx.array(mx.random.randint(0, 10, shape=(200,)))
    
    return x_train, y_train, x_test, y_test

def visualize_mnist(x: mx.array, y: mx.array, y_pred: Optional[mx.array] = None, n_samples: int = 10):
    """
    Visualize MNIST digits.
    
    Args:
        x: Input tensor (batch_size, 784)
        y: Target tensor (batch_size,)
        y_pred: Predicted tensor (batch_size,)
        n_samples: Number of samples to visualize
    """
    # Create figure
    fig, axs = plt.subplots(1, n_samples, figsize=(n_samples * 2, 2))
    
    # Plot samples
    for i in range(n_samples):
        # Reshape to 28x28
        img = x[i].reshape(28, 28)
        
        # Plot
        axs[i].imshow(img, cmap='gray')
        
        # Set title
        if y_pred is None:
            axs[i].set_title(f"Label: {y[i]}")
        else:
            axs[i].set_title(f"Label: {y[i]}\nPred: {y_pred[i]}")
        
        # Remove ticks
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('mnist_visualization.png')
    
    print("MNIST visualization saved to 'mnist_visualization.png'")

def main():
    """Run MNIST classification with binary wave neural network."""
    print("Binary Wave Neural Network for MNIST")
    print("===================================\n")
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"Loaded {len(x_train)} training samples and {len(x_test)} test samples")
    
    # Create classifier
    print("\nCreating binary wave neural network...")
    classifier = BinaryWaveMNISTClassifier(input_dim=784, hidden_dim=128, output_dim=10)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    # Use a subset of test set for faster evaluation
    test_subset = 100
    accuracy = classifier.evaluate(x_test[:test_subset], y_test[:test_subset])
    print(f"Accuracy: {accuracy:.4f}")
    
    # Visualize some examples
    print("\nVisualizing examples...")
    y_pred = classifier.predict(x_test[:10])
    visualize_mnist(x_test[:10], y_test[:10], y_pred)
    
    print("\nDone!")

if __name__ == "__main__":
    main()