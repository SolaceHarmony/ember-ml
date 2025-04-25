"""
Binary Wave Neural Network implementation using MLX.

This module demonstrates how to implement a binary wave neural network
using the MLXBinaryWave class for binary operations.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import List, Tuple, Optional, Union, Dict, Any

from mlx_binary_wave import MLXBinaryWave

class BinaryLayer:
    """
    Binary layer for binary wave neural networks.
    
    This layer performs matrix multiplication using binary operations,
    where both weights and activations are binary (0 or 1).
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize a binary layer.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize weights
        self.weights = mx.array(mx.random.uniform(shape=(input_dim, output_dim)) < 0.5, dtype=mx.uint16)
        
        # Initialize bias
        self.bias = mx.array(mx.random.uniform(shape=(output_dim,)) < 0.5, dtype=mx.uint16)
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through the binary layer.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Output tensor (batch_size, output_dim)
        """
        # Binarize input
        x_binary = mx.array(x >= 0.5, dtype=mx.uint16)
        
        # Initialize output
        batch_size = x.shape[0]
        output = mx.zeros((batch_size, self.output_dim), dtype=mx.uint16)
        
        # Matrix multiplication using binary operations
        for i in range(batch_size):
            for j in range(self.output_dim):
                # Initialize accumulator for this output element
                acc = mx.array(0, dtype=mx.uint16)
                
                # Compute dot product using binary operations
                for k in range(self.input_dim):
                    # AND operation
                    bit_product = MLXBinaryWave.bitwise_and(x_binary[i, k], self.weights[k, j])
                    # XOR with accumulator (simulating addition in binary)
                    acc = MLXBinaryWave.bitwise_xor(acc, bit_product)
                
                # XOR with bias
                result = MLXBinaryWave.bitwise_xor(acc, self.bias[j])
                
                # Update output using at
                output = output.at[i, j].add(result)
        
        return output

class BinaryWaveNeuron:
    """
    Binary wave neuron for binary wave neural networks.
    
    This neuron processes inputs using binary wave operations,
    maintaining state through phase information.
    """
    
    def __init__(self, input_dim: int, state_dim: int):
        """
        Initialize a binary wave neuron.
        
        Args:
            input_dim: Input dimension
            state_dim: State dimension
        """
        self.input_dim = input_dim
        self.state_dim = state_dim
        
        # Initialize weights
        self.weights = mx.array(mx.random.uniform(shape=(input_dim, state_dim)) < 0.5, dtype=mx.uint16)
        
        # Initialize state
        self.state = mx.zeros((state_dim,), dtype=mx.uint16)
    
    def __call__(self, x: mx.array, time_delta: float = 1.0) -> mx.array:
        """
        Forward pass through the binary wave neuron.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            time_delta: Time delta for state update
            
        Returns:
            Output tensor (batch_size, state_dim)
        """
        # Binarize input
        x_binary = mx.array(x >= 0.5, dtype=mx.uint16)
        
        # Initialize output
        batch_size = x.shape[0]
        output = mx.zeros((batch_size, self.state_dim), dtype=mx.uint16)
        
        # Matrix multiplication using binary operations
        for i in range(batch_size):
            for j in range(self.state_dim):
                # Initialize accumulator for this output element
                acc = mx.array(0, dtype=mx.uint16)
                
                # Compute dot product using binary operations
                for k in range(self.input_dim):
                    # AND operation
                    bit_product = MLXBinaryWave.bitwise_and(x_binary[i, k], self.weights[k, j])
                    # XOR with accumulator (simulating addition in binary)
                    acc = MLXBinaryWave.bitwise_xor(acc, bit_product)
                
                # Store result using at
                output = output.at[i, j].add(acc)
        
        # Update state based on time_delta
        # Convert time_delta to phase shift
        phase_shift = int(time_delta * 8) % 8
        
        # Shift the state
        shifted_state = MLXBinaryWave.binary_wave_propagate(self.state, phase_shift)
        
        # Update state with output
        self.state = MLXBinaryWave.bitwise_xor(shifted_state, output[0])
        
        return output

class BinaryWaveNetwork:
    """
    Binary wave neural network.
    
    This network combines binary layers and binary wave neurons
    to process inputs using binary wave operations.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize a binary wave neural network.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Create layers
        self.input_layer = BinaryLayer(input_dim, hidden_dim)
        self.wave_neuron = BinaryWaveNeuron(hidden_dim, output_dim)
    
    def __call__(self, x: mx.array, time_delta: float = 1.0) -> mx.array:
        """
        Forward pass through the binary wave neural network.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            time_delta: Time delta for wave neuron update
            
        Returns:
            Output tensor (batch_size, output_dim)
        """
        # Process input through input layer
        hidden = self.input_layer(x)
        
        # Process hidden through wave neuron
        output = self.wave_neuron(hidden, time_delta)
        
        return output


# Example usage
if __name__ == "__main__":
    # Create a binary wave neural network
    input_dim = 8
    hidden_dim = 16
    output_dim = 4
    network = BinaryWaveNetwork(input_dim, hidden_dim, output_dim)
    
    # Create a random binary input
    batch_size = 2
    x = mx.array(mx.random.uniform(shape=(batch_size, input_dim)) < 0.5, dtype=mx.uint16)
    
    # Forward pass
    output = network(x)
    
    print("Input shape:", x.shape)
    print("Input:", x)
    print("Output shape:", output.shape)
    print("Output:", output)
    
    # Test with different time deltas
    output1 = network(x, time_delta=0.5)
    output2 = network(x, time_delta=1.0)
    output3 = network(x, time_delta=1.5)
    
    print("\nOutput with time_delta=0.5:", output1)
    print("Output with time_delta=1.0:", output2)
    print("Output with time_delta=1.5:", output3)