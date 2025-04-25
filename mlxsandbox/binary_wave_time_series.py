"""
Binary Wave Time Series Prediction.

This script demonstrates how to use binary wave neural networks
for time series prediction.
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union, Dict, Any
from ember_ml.nn.tensor import TensorLike
from ember_ml.nn import tensor
from mlx_mega_binary import MLXMegaBinary, InterferenceMode

class BinaryWaveTimeSeriesPredictor:
    """
    Binary Wave Time Series Predictor.
    
    This class uses binary wave operations to predict time series data.
    """
    
    def __init__(self, input_length: int, forecast_horizon: int, wave_length: int = 32):
        """
        Initialize the predictor.
        
        Args:
            input_length: Length of input sequence
            forecast_horizon: Number of steps to forecast
            wave_length: Length of binary wave patterns
        """
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.wave_length = wave_length
        
        # Initialize wave patterns
        self.wave_patterns = []
        for i in range(input_length):
            # Create a unique wave pattern for each input position
            half_period = 2 + i % 8  # Vary half period from 2 to 9
            pattern = MLXMegaBinary.generate_blocky_sin(
                MLXMegaBinary(bin(wave_length)[2:]),
                MLXMegaBinary(bin(half_period)[2:])
            )
            self.wave_patterns.append(pattern)
        
        # Initialize weights
        self.weights = []
        for i in range(forecast_horizon):
            # Create weights for each forecast step
            step_weights = []
            for j in range(input_length):
                # Initialize with random binary pattern
                if (i + j) % 2 == 0:
                    weight = MLXMegaBinary("1010" * (wave_length // 4))
                else:
                    weight = MLXMegaBinary("0101" * (wave_length // 4))
                step_weights.append(weight)
            self.weights.append(step_weights)
    
    def _encode_value(self, value: float) -> MLXMegaBinary:
        """
        Encode a float value as a binary wave.
        
        Args:
            value: Float value to encode
            
        Returns:
            Binary wave representation
        """
        # Scale value to [0, 1]
        scaled_value = max(0.0, min(1.0, value))
        
        # Convert to duty cycle
        duty_cycle = MLXMegaBinary(bin(int(scaled_value * self.wave_length))[2:])
        
        # Create binary pattern with specified duty cycle
        pattern = MLXMegaBinary.create_duty_cycle(
            MLXMegaBinary(bin(self.wave_length)[2:]),
            duty_cycle
        )
        
        return pattern
    
    def _decode_value(self, pattern: MLXMegaBinary) -> float:
        """
        Decode a binary wave to a float value.
        
        Args:
            pattern: Binary wave representation
            
        Returns:
            Float value
        """
        # Count the number of 1 bits
        bits = pattern.to_bits()
        count = sum(bits)
        
        # Scale to [0, 1]
        scaled_value = count / self.wave_length
        
        return scaled_value
    
    def predict(self, input_sequence: List[float]) -> List[float]:
        """
        Predict future values based on input sequence.
        
        Args:
            input_sequence: Input time series data
            
        Returns:
            Predicted future values
        """
        if len(input_sequence) != self.input_length:
            raise ValueError(f"Input sequence must have length {self.input_length}")
        
        # Encode input sequence
        encoded_input = [self._encode_value(value) for value in input_sequence]
        
        # Initialize predictions
        predictions = []
        
        # For each forecast step
        for step in range(self.forecast_horizon):
            # Initialize wave accumulator
            wave_accumulator = MLXMegaBinary("0" * self.wave_length)
            
            # For each input position
            for i in range(self.input_length):
                # Get input wave and weight
                input_wave = encoded_input[i]
                weight = self.weights[step][i]
                
                # Modulate input wave with weight
                modulated_wave = MLXMegaBinary.interfere(
                    [input_wave, weight],
                    InterferenceMode.XOR
                )
                
                # Accumulate
                wave_accumulator = MLXMegaBinary.interfere(
                    [wave_accumulator, modulated_wave],
                    InterferenceMode.XOR
                )
            
            # Decode prediction
            prediction = self._decode_value(wave_accumulator)
            predictions.append(prediction)
        
        return predictions

def generate_sine_wave(length: int, frequency: float = 0.1, noise_level: float = 0.1) -> Any:
    """
    Generate a sine wave with noise.
    
    Args:
        length: Length of the wave
        frequency: Frequency of the wave
        noise_level: Level of noise to add
        
    Returns:
        Sine wave with noise
    """
    # Generate time points
    t = np.arange(length)
    
    # Generate sine wave
    wave = ops.sin(2 * ops.pi * frequency * t)
    
    # Add noise
    noise = tensor.random_normal(0, noise_level, length)
    wave += noise
    
    # Scale to [0, 1]
    wave = (wave + 1) / 2
    
    return wave
def plot_time_series(input_data: TensorLike, predictions: TensorLike, actual: Optional[TensorLike] = None):
    """
    Plot time series data and predictions.
    
    Args:
        input_data: Input time series data
        predictions: Predicted future values
        actual: Actual future values (optional)
    """
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot input data
    plt.plot(range(len(input_data)), input_data, 'b-', label='Input Data')
    
    # Plot predictions
    plt.plot(range(len(input_data), len(input_data) + len(predictions)), predictions, 'r-', label='Predictions')
    
    # Plot actual future values if provided
    if actual is not None:
        plt.plot(range(len(input_data), len(input_data) + len(actual)), actual, 'g-', label='Actual')
    
    # Add labels and legend
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Prediction with Binary Waves')
    plt.legend()
    
    # Save figure
    plt.savefig('time_series_prediction.png')
    
    print("Time series prediction plot saved to 'time_series_prediction.png'")

def main():
    """Run time series prediction with binary waves."""
    print("Binary Wave Time Series Prediction")
    print("=================================\n")
    
    # Parameters
    input_length = 20
    forecast_horizon = 10
    total_length = input_length + forecast_horizon
    
    # Generate synthetic time series data
    print("Generating synthetic time series data...")
    data = generate_sine_wave(total_length, frequency=0.05, noise_level=0.1)
    
    # Split into input and future
    input_data = data[:input_length]
    future_data = data[input_length:]
    
    print(f"Input data shape: {input_data.shape}")
    print(f"Future data shape: {future_data.shape}")
    
    # Create predictor
    print("\nCreating binary wave time series predictor...")
    predictor = BinaryWaveTimeSeriesPredictor(input_length, forecast_horizon)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = predictor.predict(input_data.tolist())
    
    print(f"Predictions: {predictions}")
    
    # Calculate error
    mse = np.mean((tensor.convert_to_tensor(predictions) - future_data) ** 2)
    print(f"\nMean Squared Error: {mse:.4f}")
    
    # Plot results
    print("\nPlotting results...")
    plot_time_series(input_data, tensor.convert_to_tensor(predictions), future_data)
    
    print("\nDone!")

if __name__ == "__main__":
    main()