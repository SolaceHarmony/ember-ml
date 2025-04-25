"""
Binary Wave Neural Network implementation using MLXMegaBinary.

This module demonstrates how to implement a binary wave neural network
using the MLXMegaBinary class for binary operations.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import List, Tuple, Optional, Union, Dict, Any

from mlx_mega_binary import MLXMegaBinary, InterferenceMode

class BinaryWaveNeuron:
    """
    Binary wave neuron using MLXMegaBinary.
    
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
        
        # Initialize weights as MLXMegaBinary objects
        self.weights = []
        for i in range(state_dim):
            row = []
            for j in range(input_dim):
                # Initialize with random binary pattern
                # For simplicity, we'll use a fixed pattern here
                if (i + j) % 2 == 0:
                    row.append(MLXMegaBinary("1010"))
                else:
                    row.append(MLXMegaBinary("0101"))
            self.weights.append(row)
        
        # Initialize state
        self.state = []
        for i in range(state_dim):
            self.state.append(MLXMegaBinary("0"))
    
    def forward(self, x: List[MLXMegaBinary], time_step: MLXMegaBinary) -> List[MLXMegaBinary]:
        """
        Forward pass through the binary wave neuron.
        
        Args:
            x: List of MLXMegaBinary objects representing the input
            time_step: MLXMegaBinary representing the time step
            
        Returns:
            List of MLXMegaBinary objects representing the output
        """
        # Process input
        input_waves = []
        for i in range(self.state_dim):
            # Combine input waves
            waves = []
            for j in range(self.input_dim):
                # Multiply input by weight
                wave = x[j].mul(self.weights[i][j])
                waves.append(wave)
            
            # Interfere waves
            input_wave = MLXMegaBinary.interfere(waves, InterferenceMode.XOR)
            input_waves.append(input_wave)
        
        # Update state
        new_state = []
        for i in range(self.state_dim):
            # Propagate state
            propagated_state = self.state[i].propagate(time_step)
            
            # Combine with input
            new_state_i = MLXMegaBinary.interfere(
                [propagated_state, input_waves[i]],
                InterferenceMode.XOR
            )
            
            new_state.append(new_state_i)
        
        self.state = new_state
        return self.state

class BinaryWaveLayer:
    """
    Binary wave layer using MLXMegaBinary.
    
    This layer consists of multiple binary wave neurons.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize a binary wave layer.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create neurons
        self.neurons = []
        for i in range(output_dim):
            self.neurons.append(BinaryWaveNeuron(input_dim, 1))
    
    def forward(self, x: List[MLXMegaBinary], time_step: MLXMegaBinary) -> List[MLXMegaBinary]:
        """
        Forward pass through the binary wave layer.
        
        Args:
            x: List of MLXMegaBinary objects representing the input
            time_step: MLXMegaBinary representing the time step
            
        Returns:
            List of MLXMegaBinary objects representing the output
        """
        output = []
        for neuron in self.neurons:
            neuron_output = neuron.forward(x, time_step)
            output.append(neuron_output[0])
        
        return output

class BinaryWaveNetwork:
    """
    Binary wave neural network using MLXMegaBinary.
    
    This network consists of multiple binary wave layers.
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
        self.hidden_layer = BinaryWaveLayer(input_dim, hidden_dim)
        self.output_layer = BinaryWaveLayer(hidden_dim, output_dim)
    
    def forward(self, x: List[MLXMegaBinary], time_step: MLXMegaBinary) -> List[MLXMegaBinary]:
        """
        Forward pass through the binary wave neural network.
        
        Args:
            x: List of MLXMegaBinary objects representing the input
            time_step: MLXMegaBinary representing the time step
            
        Returns:
            List of MLXMegaBinary objects representing the output
        """
        # Process hidden layer
        hidden = self.hidden_layer.forward(x, time_step)
        
        # Process output layer
        output = self.output_layer.forward(hidden, time_step)
        
        return output

def create_binary_wave_input(input_data: List[int]) -> List[MLXMegaBinary]:
    """
    Create binary wave input from integer data.
    
    Args:
        input_data: List of integers
        
    Returns:
        List of MLXMegaBinary objects
    """
    binary_input = []
    for value in input_data:
        # Convert to binary string
        bin_str = bin(value)[2:]
        
        # Create MLXMegaBinary
        binary_input.append(MLXMegaBinary(bin_str))
    
    return binary_input

def decode_binary_wave_output(output: List[MLXMegaBinary]) -> List[int]:
    """
    Decode binary wave output to integers.
    
    Args:
        output: List of MLXMegaBinary objects
        
    Returns:
        List of integers
    """
    decoded_output = []
    for wave in output:
        # Count the number of 1 bits
        bits = wave.to_bits()
        count = sum(bits)
        
        decoded_output.append(count)
    
    return decoded_output

def binary_wave_xor_example():
    """
    Example of using a binary wave neural network to learn XOR.
    """
    # Create network
    network = BinaryWaveNetwork(2, 4, 1)
    
    # Create time step
    time_step = MLXMegaBinary("1")
    
    # Create training data
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]
    
    # Train network
    print("Training binary wave network on XOR...")
    for epoch in range(10):
        total_error = 0
        
        for inputs, targets in training_data:
            # Convert inputs to binary waves
            binary_inputs = create_binary_wave_input(inputs)
            
            # Forward pass
            binary_outputs = network.forward(binary_inputs, time_step)
            
            # Decode outputs
            outputs = decode_binary_wave_output(binary_outputs)
            
            # Calculate error
            error = sum((outputs[i] - targets[i]) ** 2 for i in range(len(targets)))
            total_error += error
            
            # Print results
            print(f"Inputs: {inputs}, Outputs: {outputs}, Targets: {targets}")
        
        print(f"Epoch {epoch + 1}, Error: {total_error}")
    
    print("Training complete!")

def binary_wave_pattern_recognition_example():
    """
    Example of using a binary wave neural network for pattern recognition.
    """
    # Create network
    network = BinaryWaveNetwork(8, 16, 4)
    
    # Create time step
    time_step = MLXMegaBinary("1")
    
    # Create patterns
    patterns = [
        # Pattern 1: 10101010
        MLXMegaBinary("10101010"),
        # Pattern 2: 11001100
        MLXMegaBinary("11001100"),
        # Pattern 3: 11110000
        MLXMegaBinary("11110000"),
        # Pattern 4: 10001000
        MLXMegaBinary("10001000")
    ]
    
    # Create input
    input_patterns = []
    for pattern in patterns:
        # Create 8 copies of the pattern with different shifts
        pattern_inputs = []
        for i in range(8):
            shift = MLXMegaBinary(bin(i)[2:])
            shifted_pattern = pattern.propagate(shift)
            pattern_inputs.append(shifted_pattern)
        
        input_patterns.append(pattern_inputs)
    
    # Process patterns
    print("Processing patterns...")
    for i, pattern_inputs in enumerate(input_patterns):
        print(f"Pattern {i + 1}:")
        
        # Forward pass
        outputs = network.forward(pattern_inputs, time_step)
        
        # Decode outputs
        decoded_outputs = decode_binary_wave_output(outputs)
        
        print(f"  Outputs: {decoded_outputs}")
    
    print("Processing complete!")

def binary_wave_duty_cycle_example():
    """
    Example of using binary wave duty cycles for encoding information.
    """
    # Create duty cycles
    duty_cycles = [
        # 25% duty cycle
        MLXMegaBinary.create_duty_cycle(MLXMegaBinary("1000"), MLXMegaBinary("10")),
        # 50% duty cycle
        MLXMegaBinary.create_duty_cycle(MLXMegaBinary("1000"), MLXMegaBinary("100")),
        # 75% duty cycle
        MLXMegaBinary.create_duty_cycle(MLXMegaBinary("1000"), MLXMegaBinary("110"))
    ]
    
    # Print duty cycles
    print("Duty cycles:")
    for i, duty_cycle in enumerate(duty_cycles):
        print(f"  {i + 1}: {duty_cycle.to_string()}")
    
    # Create network
    network = BinaryWaveNetwork(3, 6, 3)
    
    # Create time step
    time_step = MLXMegaBinary("1")
    
    # Forward pass
    outputs = network.forward(duty_cycles, time_step)
    
    # Decode outputs
    decoded_outputs = decode_binary_wave_output(outputs)
    
    print(f"Outputs: {decoded_outputs}")

def binary_wave_interference_example():
    """
    Example of using binary wave interference for combining information.
    """
    # Create waves
    waves = [
        # Wave 1: 10101010
        MLXMegaBinary("10101010"),
        # Wave 2: 11001100
        MLXMegaBinary("11001100"),
        # Wave 3: 11110000
        MLXMegaBinary("11110000")
    ]
    
    # Print waves
    print("Waves:")
    for i, wave in enumerate(waves):
        print(f"  {i + 1}: {wave.to_string()}")
    
    # Create interference patterns
    xor_interference = MLXMegaBinary.interfere(waves, InterferenceMode.XOR)
    and_interference = MLXMegaBinary.interfere(waves, InterferenceMode.AND)
    or_interference = MLXMegaBinary.interfere(waves, InterferenceMode.OR)
    
    # Print interference patterns
    print("Interference patterns:")
    print(f"  XOR: {xor_interference.to_string()}")
    print(f"  AND: {and_interference.to_string()}")
    print(f"  OR: {or_interference.to_string()}")
    
    # Create network
    network = BinaryWaveNetwork(3, 6, 3)
    
    # Create time step
    time_step = MLXMegaBinary("1")
    
    # Forward pass
    outputs = network.forward([xor_interference, and_interference, or_interference], time_step)
    
    # Decode outputs
    decoded_outputs = decode_binary_wave_output(outputs)
    
    print(f"Outputs: {decoded_outputs}")

if __name__ == "__main__":
    print("Binary Wave Neural Network Examples")
    print("==================================")
    
    print("\nXOR Example:")
    binary_wave_xor_example()
    
    print("\nPattern Recognition Example:")
    binary_wave_pattern_recognition_example()
    
    print("\nDuty Cycle Example:")
    binary_wave_duty_cycle_example()
    
    print("\nInterference Example:")
    binary_wave_interference_example()