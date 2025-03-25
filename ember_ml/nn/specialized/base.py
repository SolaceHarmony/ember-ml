"""
Base neuron implementations for the neural network.

This module provides the foundational WeightedLTCNeuron class that implements
basic Leaky Time Constant (LTC) dynamics with weighted inputs.
"""

from typing import Union, List
import numpy as np
from numpy.typing import NDArray


class WeightedLTCNeuron:
    """
    Base neuron class implementing Leaky Time Constant (LTC) dynamics with weighted inputs.
    
    This neuron type maintains a state that decays over time and responds to
    weighted input signals. It serves as the foundation for more specialized
    neuron implementations.
    
    Attributes:
        state (float): Current activation state of the neuron
        base_tau (float): Base time constant for state decay
        num_inputs (int): Number of input connections
        weights (NDArray[np.float32]): Input weight vector
    """
    
    def __init__(self, tau: float = 1.0, num_inputs: int = 3):
        """
        Initialize a weighted LTC neuron.
        
        Args:
            tau: Base time constant for the neuron
            num_inputs: Number of input connections to create
        """
        self.state: float = 0.0
        self.base_tau: float = tau
        self.num_inputs: int = num_inputs
        # Initialize weights uniformly in [-0.5, 0.5]
        self.weights: NDArray[np.float32] = np.random.uniform(
            -0.5, 0.5, size=num_inputs
        )
    
    def update(self,
              inputs: Union[List[float], NDArray[np.float32]],
              dt: float = 0.1,
              tau_mod: float = 1.0,
              feedback: float = 0.0) -> float:
        """
        Update the neuron's state based on inputs and current conditions.
        
        Args:
            inputs: Input signals to the neuron
            dt: Time step size for integration
            tau_mod: Modifier for the time constant (e.g., from dopamine)
            feedback: Additional feedback input (e.g., from recurrent connections)
            
        Returns:
            float: Updated neuron state
            
        Note:
            The state update follows the LTC dynamics:
            dstate/dt = (-state/tau + weighted_input + feedback) * dt
        """
        # Ensure inputs are numpy array
        input_array = np.array(inputs, dtype=np.float32)
        
        # Calculate effective time constant
        effective_tau = self.base_tau * tau_mod
        
        # Compute weighted input including feedback
        synaptic_input = np.dot(self.weights, input_array) + feedback
        
        # Calculate state change
        dstate = (-self.state / effective_tau) + synaptic_input
        
        # Update state
        self.state += dstate * dt
        
        return self.state
    
    def reset(self) -> None:
        """Reset the neuron's state to zero."""
        self.state = 0.0
    
    def get_weights(self) -> NDArray[np.float32]:
        """
        Get the current weight vector.
        
        Returns:
            NDArray[np.float32]: Copy of the weight vector
        """
        return self.weights.copy()
    
    def set_weights(self, weights: Union[List[float], NDArray[np.float32]]) -> None:
        """
        Set new weights for the neuron.
        
        Args:
            weights: New weight values to use
            
        Raises:
            ValueError: If weights length doesn't match num_inputs
        """
        weights_array = np.array(weights, dtype=np.float32)
        if weights_array.shape != (self.num_inputs,):
            raise ValueError(
                f"Weights must have shape ({self.num_inputs},), "
                f"got {weights_array.shape}"
            )
        self.weights = weights_array