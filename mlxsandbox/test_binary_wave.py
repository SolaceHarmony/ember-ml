"""
Test script for binary wave neural networks.

This script demonstrates how to use the binary wave neural network
implementations in the MLX sandbox.
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from mlx_binary_wave import MLXBinaryWave
from binary_wave_nn import BinaryWaveNetwork
from mlx_mega_binary import MLXMegaBinary, InterferenceMode
from binary_wave_mega_nn import BinaryWaveNetwork as MegaBinaryWaveNetwork
from binary_wave_mega_nn import create_binary_wave_input, decode_binary_wave_output

def test_binary_wave_operations():
    """Test basic binary wave operations."""
    print("Testing binary wave operations...")
    
    # Create binary patterns
    a = mx.array([0b1010, 0b1100, 0b1111], dtype=mx.int32)
    b = mx.array([0b0101, 0b1010, 0b0000], dtype=mx.int32)
    
    print("a:", a)
    print("b:", b)
    
    # Test bitwise operations
    print("a & b:", MLXBinaryWave.bitwise_and(a, b))
    print("a | b:", MLXBinaryWave.bitwise_or(a, b))
    print("a ^ b:", MLXBinaryWave.bitwise_xor(a, b))
    print("~a:", MLXBinaryWave.bitwise_not(a))
    
    # Test shift operations
    print("a << 1:", MLXBinaryWave.left_shift(a, mx.array(1)))
    print("a >> 1:", MLXBinaryWave.right_shift(a, mx.array(1)))
    
    # Test bit operations
    print("Count ones in a:", MLXBinaryWave.count_ones(a))
    print("Get bit 1 from a:", MLXBinaryWave.get_bit(a, mx.array(1)))
    print("Set bit 2 in a to 1:", MLXBinaryWave.set_bit(a, mx.array(2), mx.array(1)))
    print("Toggle bit 0 in a:", MLXBinaryWave.toggle_bit(a, mx.array(0)))
    
    # Test wave operations
    print("Wave interference (XOR):", MLXBinaryWave.binary_wave_interference([a, b], mode='xor'))
    print("Wave propagation (shift 1):", MLXBinaryWave.binary_wave_propagate(a, mx.array(1)))
    print("Duty cycle (0.5, length 8):", MLXBinaryWave.create_duty_cycle(8, 0.5))
    print("Blocky sin (length 8, half_period 2):", MLXBinaryWave.generate_blocky_sin(8, 2))
    
    print("Binary wave operations test complete!\n")

def test_binary_wave_nn():
    """Test binary wave neural network."""
    print("Testing binary wave neural network...")
    
    # Create a binary wave neural network
    input_dim = 8
    hidden_dim = 16
    output_dim = 4
    network = BinaryWaveNetwork(input_dim, hidden_dim, output_dim)
    
    # Create a random binary input
    batch_size = 2
    x = mx.array(mx.random.uniform(shape=(batch_size, input_dim)) < 0.5, dtype=mx.int32)
    
    print("Input shape:", x.shape)
    print("Input:", x)
    
    # Forward pass
    output = network(x)
    
    print("Output shape:", output.shape)
    print("Output:", output)
    
    # Test with different time deltas
    output1 = network(x, time_delta=0.5)
    output2 = network(x, time_delta=1.0)
    output3 = network(x, time_delta=1.5)
    
    print("Output with time_delta=0.5:", output1)
    print("Output with time_delta=1.0:", output2)
    print("Output with time_delta=1.5:", output3)
    
    print("Binary wave neural network test complete!\n")

def test_mega_binary_operations():
    """Test MegaBinary operations."""
    print("Testing MegaBinary operations...")
    
    # Create MLXMegaBinary objects
    a = MLXMegaBinary("1010")
    b = MLXMegaBinary("0101")
    
    print("a:", a)
    print("b:", b)
    
    # Test bitwise operations
    print("a & b:", a.bitwise_and(b))
    print("a | b:", a.bitwise_or(b))
    print("a ^ b:", a.bitwise_xor(b))
    print("~a:", a.bitwise_not())
    
    # Test shift operations
    print("a << 2:", a.shift_left(MLXMegaBinary("10")))
    print("a >> 1:", a.shift_right(MLXMegaBinary("1")))
    
    # Test bit operations
    print("a.get_bit(1):", a.get_bit(MLXMegaBinary("1")))
    a.set_bit(MLXMegaBinary("10"), True)
    print("a after setting bit 2:", a)
    
    # Test wave operations
    print("Blocky sin (length=8, half_period=2):", MLXMegaBinary.generate_blocky_sin(
        MLXMegaBinary("1000"), MLXMegaBinary("10")))
    print("Duty cycle (length=8, duty_cycle=0.5):", MLXMegaBinary.create_duty_cycle(
        MLXMegaBinary("1000"), MLXMegaBinary("100")))
    
    # Test interference
    print("Interference (XOR):", MLXMegaBinary.interfere([a, b], InterferenceMode.XOR))
    print("Interference (AND):", MLXMegaBinary.interfere([a, b], InterferenceMode.AND))
    print("Interference (OR):", MLXMegaBinary.interfere([a, b], InterferenceMode.OR))
    
    print("MegaBinary operations test complete!\n")

def test_mega_binary_wave_nn():
    """Test binary wave neural network with MegaBinary."""
    print("Testing binary wave neural network with MegaBinary...")
    
    # Create a binary wave neural network
    input_dim = 2
    hidden_dim = 4
    output_dim = 1
    network = MegaBinaryWaveNetwork(input_dim, hidden_dim, output_dim)
    
    # Create input
    x = create_binary_wave_input([10, 20])
    
    # Create time step
    time_step = MLXMegaBinary("1")
    
    print("Input:", [x_i.to_string() for x_i in x])
    
    # Forward pass
    output = network.forward(x, time_step)
    
    print("Output:", [o.to_string() for o in output])
    
    # Decode output
    decoded_output = decode_binary_wave_output(output)
    
    print("Decoded output:", decoded_output)
    
    print("Binary wave neural network with MegaBinary test complete!\n")

def visualize_binary_waves():
    """Visualize binary waves."""
    print("Visualizing binary waves...")
    
    # Create binary waves
    length = 32
    half_period1 = 4
    half_period2 = 8
    
    # Create waves using MLXBinaryWave
    wave1 = MLXBinaryWave.generate_blocky_sin(length, half_period1)
    wave2 = MLXBinaryWave.generate_blocky_sin(length, half_period2)
    
    # Convert to numpy for visualization
    wave1_np = tensor.convert_to_tensor(wave1)
    wave2_np = tensor.convert_to_tensor(wave2)
    
    # Create interference patterns
    xor_interference = MLXBinaryWave.binary_wave_interference([wave1, wave2], mode='xor')
    and_interference = MLXBinaryWave.binary_wave_interference([wave1, wave2], mode='and')
    or_interference = MLXBinaryWave.binary_wave_interference([wave1, wave2], mode='or')
    
    # Convert to numpy for visualization
    xor_interference_np = tensor.convert_to_tensor(xor_interference)
    and_interference_np = tensor.convert_to_tensor(and_interference)
    or_interference_np = tensor.convert_to_tensor(or_interference)
    
    # Create duty cycles
    duty_cycle25 = MLXBinaryWave.create_duty_cycle(length, 0.25)
    duty_cycle50 = MLXBinaryWave.create_duty_cycle(length, 0.5)
    duty_cycle75 = MLXBinaryWave.create_duty_cycle(length, 0.75)
    
    # Convert to numpy for visualization
    duty_cycle25_np = tensor.convert_to_tensor(duty_cycle25)
    duty_cycle50_np = tensor.convert_to_tensor(duty_cycle50)
    duty_cycle75_np = tensor.convert_to_tensor(duty_cycle75)
    
    # Create figure
    fig, axs = plt.subplots(3, 3, figsize=(12, 8))
    
    # Plot waves
    axs[0, 0].step(range(length), wave1_np, where='post')
    axs[0, 0].set_title('Wave 1 (half_period=4)')
    axs[0, 0].set_ylim(-0.1, 1.1)
    
    axs[0, 1].step(range(length), wave2_np, where='post')
    axs[0, 1].set_title('Wave 2 (half_period=8)')
    axs[0, 1].set_ylim(-0.1, 1.1)
    
    # Plot interference patterns
    axs[0, 2].step(range(length), xor_interference_np, where='post')
    axs[0, 2].set_title('XOR Interference')
    axs[0, 2].set_ylim(-0.1, 1.1)
    
    axs[1, 0].step(range(length), and_interference_np, where='post')
    axs[1, 0].set_title('AND Interference')
    axs[1, 0].set_ylim(-0.1, 1.1)
    
    axs[1, 1].step(range(length), or_interference_np, where='post')
    axs[1, 1].set_title('OR Interference')
    axs[1, 1].set_ylim(-0.1, 1.1)
    
    # Plot duty cycles
    axs[1, 2].step(range(length), duty_cycle25_np, where='post')
    axs[1, 2].set_title('Duty Cycle 25%')
    axs[1, 2].set_ylim(-0.1, 1.1)
    
    axs[2, 0].step(range(length), duty_cycle50_np, where='post')
    axs[2, 0].set_title('Duty Cycle 50%')
    axs[2, 0].set_ylim(-0.1, 1.1)
    
    axs[2, 1].step(range(length), duty_cycle75_np, where='post')
    axs[2, 1].set_title('Duty Cycle 75%')
    axs[2, 1].set_ylim(-0.1, 1.1)
    
    # Create a propagated wave
    propagated_wave = MLXBinaryWave.binary_wave_propagate(wave1, mx.array(4))
    propagated_wave_np = tensor.convert_to_tensor(propagated_wave)
    
    axs[2, 2].step(range(length), propagated_wave_np, where='post')
    axs[2, 2].set_title('Propagated Wave (shift=4)')
    axs[2, 2].set_ylim(-0.1, 1.1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('binary_waves.png')
    
    print("Binary waves visualization saved to 'binary_waves.png'")
    print("Visualization complete!\n")

def main():
    """Run all tests."""
    print("Binary Wave Neural Networks Test Suite")
    print("=====================================\n")
    
    # Test binary wave operations
    test_binary_wave_operations()
    
    # Test binary wave neural network
    test_binary_wave_nn()
    
    # Test MegaBinary operations
    test_mega_binary_operations()
    
    # Test binary wave neural network with MegaBinary
    test_mega_binary_wave_nn()
    
    # Visualize binary waves
    visualize_binary_waves()
    
    print("All tests complete!")

if __name__ == "__main__":
    main()