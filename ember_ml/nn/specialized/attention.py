"""
Attention-enhanced neuron implementation.

This module provides the LTCNeuronWithAttention class that combines
LTC dynamics with an attention mechanism for adaptive behavior.
"""

from typing import Optional
import numpy as np

from ember_ml.attention.mechanisms.mechanism import CausalAttention


class LTCNeuronWithAttention:
    """
    A neuron that combines LTC dynamics with an attention mechanism.
    
    This neuron type uses attention to modulate its time constant and
    input processing based on prediction accuracy and novelty detection.
    
    Attributes:
        id (int): Unique identifier for the neuron
        tau (float): Base time constant
        state (float): Current activation state
        attention (CausalAttention): Attention mechanism
        last_prediction (float): Previous state prediction
    """
    
    def __init__(self,
                 neuron_id: int,
                 tau: float = 1.0,
                 attention_params: Optional[dict] = None):
        """
        Initialize an attention-enhanced LTC neuron.
        
        Args:
            neuron_id: Unique identifier for the neuron
            tau: Base time constant for LTC dynamics
            attention_params: Optional parameters for attention mechanism
                            (decay_rate, novelty_threshold, memory_length)
        """
        self.id = neuron_id
        self.tau = tau
        self.state = 0.0
        self.last_prediction = 0.0
        
        # Initialize attention mechanism with optional custom parameters
        if attention_params is None:
            attention_params = {}
        self.attention = CausalAttention(**attention_params)
    
    def update(self, input_signal: float, dt: float) -> float:
        """
        Update the neuron's state using attention-modulated dynamics.
        
        Args:
            input_signal: Current input value
            dt: Time step size for integration
            
        Returns:
            float: Updated neuron state
            
        Note:
            The update process:
            1. Calculate prediction error
            2. Update attention based on error and current state
            3. Modulate time constant based on attention
            4. Update state using attention-weighted input
        """
        # Calculate prediction error
        prediction_error = input_signal - self.last_prediction
        
        # Update attention
        attention_value = self.attention.update(
            neuron_id=self.id,
            prediction_error=prediction_error,
            current_state=self.state,
            target_state=input_signal
        )
        
        # Modulate time constant based on attention
        # Higher attention -> faster response (smaller effective tau)
        effective_tau = self.tau * (1.0 - 0.3 * attention_value)
        
        # Update LTC dynamics with attention-modulated input
        d_state = (1.0/effective_tau) * (
            # Attention increases input influence
            input_signal * (1.0 + attention_value) - self.state
        ) * dt
        
        # Update state
        self.state += d_state
        
        # Store prediction for next update
        self.last_prediction = self.state
        
        return self.state
    
    def reset(self) -> None:
        """Reset the neuron's state and prediction history."""
        self.state = 0.0
        self.last_prediction = 0.0
    
    def get_attention_value(self) -> float:
        """
        Get the current total attention value for this neuron.
        
        Returns:
            float: Current attention value
        """
        return self.attention.states.get(
            self.id,
            self.attention.states.get(self.id)
        ).compute_total()
    
    def __repr__(self) -> str:
        """
        Get a string representation of the neuron.
        
        Returns:
            str: Neuron description including ID and tau
        """
        return (f"LTCNeuronWithAttention(id={self.id}, "
                f"tau={self.tau:.2f}, "
                f"attention={self.get_attention_value():.3f})")