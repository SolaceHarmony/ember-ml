"""
Hebbian learning implementation.

This module provides the HebbianLayer class that implements Hebbian learning
for adapting connection weights based on correlated activity between
input and output units.
"""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray


class HebbianLayer:
    """
    Implements a layer with Hebbian learning capabilities.
    
    This layer adapts its weights based on correlated activity between
    input and output units following Hebb's rule: "Neurons that fire together,
    wire together."
    
    Attributes:
        weights (NDArray[np.float64]): Connection weight matrix
        eta (float): Learning rate for weight updates
        input_size (int): Number of input units
        output_size (int): Number of output units
    """
    
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 eta: float = 0.01,
                 weight_scale: float = 0.01):
        """
        Initialize a Hebbian learning layer.
        
        Args:
            input_size: Number of input units
            output_size: Number of output units
            eta: Learning rate for weight updates
            weight_scale: Scale factor for initial weight values
        """
        self.input_size = input_size
        self.output_size = output_size
        self.eta = eta
        
        # Initialize small random weights
        self.weights = np.random.randn(output_size, input_size) * weight_scale
        
        # Keep track of weight statistics
        self._weight_history: list[Tuple[float, float]] = []  # (mean, std)
    
    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute forward pass through the layer.
        
        Args:
            inputs: Input activity vector (input_size,)
            
        Returns:
            NDArray[np.float64]: Output activity vector (output_size,)
            
        Raises:
            ValueError: If input shape doesn't match input_size
        """
        if inputs.shape != (self.input_size,):
            raise ValueError(
                f"Expected input shape ({self.input_size},), "
                f"got {inputs.shape}"
            )
        
        return self.weights @ inputs
    
    def hebbian_update(self,
                      inputs: NDArray[np.float64],
                      outputs: NDArray[np.float64]) -> None:
        """
        Update weights using Hebbian learning rule.
        
        The update follows the rule:
        Δw = η * (post-synaptic activity ⊗ pre-synaptic activity)
        where ⊗ is the outer product.
        
        Args:
            inputs: Pre-synaptic activity vector (input_size,)
            outputs: Post-synaptic activity vector (output_size,)
            
        Raises:
            ValueError: If input/output shapes don't match layer dimensions
        """
        if inputs.shape != (self.input_size,):
            raise ValueError(
                f"Expected input shape ({self.input_size},), "
                f"got {inputs.shape}"
            )
        if outputs.shape != (self.output_size,):
            raise ValueError(
                f"Expected output shape ({self.output_size},), "
                f"got {outputs.shape}"
            )
        
        # Compute weight updates using outer product
        delta_w = self.eta * np.outer(outputs, inputs)
        
        # Apply updates
        self.weights += delta_w
        
        # Record weight statistics
        self._weight_history.append(
            (float(np.mean(self.weights)), float(np.std(self.weights)))
        )
    
    def get_weight_stats(self) -> list[Tuple[float, float]]:
        """
        Get history of weight statistics.
        
        Returns:
            list[Tuple[float, float]]: List of (mean, std) pairs for weights
        """
        return self._weight_history.copy()
    
    def reset_weights(self, weight_scale: float = 0.01) -> None:
        """
        Reset weights to random values.
        
        Args:
            weight_scale: Scale factor for new random weights
        """
        self.weights = np.random.randn(self.output_size, self.input_size) * weight_scale
        self._weight_history.clear()
    
    def get_weights(self) -> NDArray[np.float64]:
        """
        Get the current weight matrix.
        
        Returns:
            NDArray[np.float64]: Copy of weight matrix
        """
        return self.weights.copy()