"""
Closed-form Continuous-time (CfC) Cell

This module provides an implementation of the CfC cell,
which is a type of recurrent neural network cell that operates in continuous time.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

from ember_ml import ops
from ember_ml.nn.modules import Module, Parameter

class CfCCell(Module):
    """
    Closed-form Continuous-time (CfC) cell.
    
    This cell implements a continuous-time recurrent neural network
    with closed-form solution for the hidden state dynamics.
    """
    
    def __init__(
        self,
        units: int,
        time_scale_factor: float = 1.0,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        mixed_memory: bool = False,
        **kwargs
    ):
        """
        Initialize the CfC cell.
        
        Args:
            units: Number of units in the cell
            time_scale_factor: Factor to scale the time constant
            activation: Activation function for the output
            recurrent_activation: Activation function for the recurrent step
            use_bias: Whether to use bias
            kernel_initializer: Initializer for the kernel weights
            recurrent_initializer: Initializer for the recurrent weights
            bias_initializer: Initializer for the bias
            mixed_memory: Whether to use mixed memory
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.units = units
        self.time_scale_factor = time_scale_factor
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.mixed_memory = mixed_memory
        
        # Initialize weights
        self._initialize_weights()
        
        # State size: [hidden_state, time_state]
        self.state_size = [self.units, self.units]
        self.output_size = self.units
    
    def _initialize_weights(self):
        """Initialize the weights for the cell."""
        # Input weights
        self.kernel = Parameter(ops.zeros((self.units, self.units * 4)))
        
        # Recurrent weights
        self.recurrent_kernel = Parameter(ops.zeros((self.units, self.units * 4)))
        
        # Bias
        if self.use_bias:
            self.bias = Parameter(ops.zeros((self.units * 4,)))
        
        # Time-scale parameter (learnable)
        self.time_scale = Parameter(ops.ones((self.units,)) * self.time_scale_factor)
        
        # Initialize weights
        if self.kernel_initializer == "glorot_uniform":
            self.kernel.data = ops.glorot_uniform((self.units, self.units * 4))
        
        if self.recurrent_initializer == "orthogonal":
            self.recurrent_kernel.data = ops.orthogonal((self.units, self.units * 4))
        
        if self.use_bias and self.bias_initializer == "zeros":
            self.bias.data = ops.zeros((self.units * 4,))
    
    def forward(self, inputs, states=None, timespans=None):
        """
        Forward pass through the cell.
        
        Args:
            inputs: Input tensor
            states: Previous states [hidden_state, time_state]
            timespans: Time spans for continuous-time dynamics (default: 1.0)
            
        Returns:
            Tuple of (output, [new_hidden_state, new_time_state])
        """
        # Initialize states if not provided
        if states is None:
            h_prev = ops.zeros((ops.shape(inputs)[0], self.units))
            t_prev = ops.zeros((ops.shape(inputs)[0], self.units))
        else:
            h_prev, t_prev = states
        
        # Default timespan is 1.0 if not provided
        ts = 1.0 if timespans is None else timespans
        
        # Compute gates
        z = ops.matmul(inputs, self.kernel)
        z = ops.add(z, ops.matmul(h_prev, self.recurrent_kernel))
        if self.use_bias:
            z = ops.add(z, self.bias)
        
        # Split into gates
        z_chunks = ops.split(z, 4, axis=-1)
        z_i, z_f, z_o, z_c = z_chunks
        
        # Apply activations
        if self.recurrent_activation == "sigmoid":
            i = ops.sigmoid(z_i)  # Input gate
            f = ops.sigmoid(z_f)  # Forget gate
            o = ops.sigmoid(z_o)  # Output gate
        else:
            i = getattr(ops, self.recurrent_activation)(z_i)
            f = getattr(ops, self.recurrent_activation)(z_f)
            o = getattr(ops, self.recurrent_activation)(z_o)
        
        if self.activation == "tanh":
            c = ops.tanh(z_c)     # Cell input
        else:
            c = getattr(ops, self.activation)(z_c)
        
        # Apply time scaling
        # Compute time decay factor
        decay = ops.exp(-ts / self.time_scale)
        
        # Update time state
        t = ops.add(ops.multiply(f, t_prev), ops.multiply(i, c))
        
        # Apply time decay to hidden state
        decay_term = ops.multiply(decay, h_prev)
        time_term = ops.multiply(ops.subtract(ops.ones_like(decay), decay), t)
        
        if self.activation == "tanh":
            h = ops.multiply(o, ops.tanh(ops.add(decay_term, time_term)))
        else:
            h = ops.multiply(o, getattr(ops, self.activation)(ops.add(decay_term, time_term)))
        
        return h, [h, t]
    
    def reset_state(self, batch_size=1):
        """
        Reset the cell state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Tuple of (hidden_state, time_state)
        """
        h = ops.zeros((batch_size, self.units))
        t = ops.zeros((batch_size, self.units))
        return [h, t]