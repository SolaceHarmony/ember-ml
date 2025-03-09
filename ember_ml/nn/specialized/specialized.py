"""
Specialized neuron implementations with role-specific behaviors.

This module provides the SpecializedNeuron class that extends the base
WeightedLTCNeuron with different behavioral roles such as memory,
inhibition, and amplification.
"""

from typing import Union, List, Literal
import numpy as np
from numpy.typing import NDArray

from .base import WeightedLTCNeuron


# Define valid role types
RoleType = Literal["default", "memory", "inhibition", "amplification"]


class SpecializedNeuron(WeightedLTCNeuron):
    """
    A neuron with role-specific behavior modifications.
    
    This class extends WeightedLTCNeuron to implement different functional roles:
    - memory: Slower decay for maintaining information
    - inhibition: Dampens input signals
    - amplification: Strengthens input signals
    - default: Standard LTC behavior
    
    Attributes:
        role (str): The neuron's functional role
        state (float): Current activation state
        base_tau (float): Base time constant
        num_inputs (int): Number of input connections
        weights (NDArray[np.float64]): Input weight vector
    """
    
    # Role-specific parameters
    MEMORY_TAU_FACTOR = 1.5      # Slower decay for memory
    INHIBITION_FACTOR = 0.5      # Dampening factor for inhibition
    AMPLIFICATION_FACTOR = 1.5   # Strengthening factor for amplification
    
    def __init__(self,
                 tau: float = 1.0,
                 role: RoleType = "default",
                 num_inputs: int = 3):
        """
        Initialize a specialized neuron with a specific role.
        
        Args:
            tau: Base time constant for the neuron
            role: Functional role determining behavior
            num_inputs: Number of input connections
        """
        super().__init__(tau=tau, num_inputs=num_inputs)
        self.role = role
    
    def update(self,
              inputs: Union[List[float], NDArray[np.float64]],
              dt: float = 0.1,
              tau_mod: float = 1.0,
              feedback: float = 0.0) -> float:
        """
        Update the neuron's state based on its role and inputs.
        
        Args:
            inputs: Input signals to the neuron
            dt: Time step size for integration
            tau_mod: Modifier for the time constant
            feedback: Additional feedback input
            
        Returns:
            float: Updated neuron state
        """
        # Ensure inputs are numpy array
        input_array = np.array(inputs, dtype=np.float64)
        
        # Calculate base synaptic input including feedback
        synaptic_input = np.dot(self.weights, input_array) + feedback
        
        # Apply role-specific modifications
        if self.role == "memory":
            # Memory role: slower decay
            effective_tau = self.base_tau * tau_mod * self.MEMORY_TAU_FACTOR
            dstate = (-self.state / effective_tau) + synaptic_input
            
        elif self.role == "inhibition":
            # Inhibition role: dampen signals
            inhibited_input = -synaptic_input * self.INHIBITION_FACTOR
            effective_tau = self.base_tau * tau_mod
            dstate = (-self.state / effective_tau) + inhibited_input
            
        elif self.role == "amplification":
            # Amplification role: strengthen signals
            amplified_input = synaptic_input * self.AMPLIFICATION_FACTOR
            effective_tau = self.base_tau * tau_mod
            dstate = (-self.state / effective_tau) + amplified_input
            
        else:  # default role
            # Standard LTC behavior
            effective_tau = self.base_tau * tau_mod
            dstate = (-self.state / effective_tau) + synaptic_input
        
        # Update state
        self.state += dstate * dt
        
        return self.state
    
    @property
    def role_name(self) -> str:
        """
        Get a human-readable name for the neuron's role.
        
        Returns:
            str: Capitalized role name
        """
        return self.role.capitalize()
    
    def __repr__(self) -> str:
        """
        Get a string representation of the neuron.
        
        Returns:
            str: Neuron description including role and tau
        """
        return (f"SpecializedNeuron(role={self.role}, "
                f"tau={self.base_tau:.2f}, "
                f"num_inputs={self.num_inputs})")