"""
Stride-Aware Cell

This module provides an implementation of a stride-aware cell,
which can be used as a base for various recurrent neural network cells
that need to handle different timescales in a multi-scale architecture.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

import numpy as np
from ember_ml import ops
from ember_ml.nn.modules import Module, Parameter
from ember_ml.initializers import glorot_uniform

class StrideAwareCell(Module):
    """
    A generic stride-aware cell for multi-timescale processing.
    
    This cell provides a base implementation for cells that need to handle
    different timescales in a multi-scale architecture.
    """
    
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            stride_length: int = 1,
            time_scale_factor: float = 1.0,
            activation: str = "tanh",
            bias: bool = True,
            **kwargs
    ):
        """
        Initialize a stride-aware cell.
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden state dimension
            stride_length: Length of the stride this cell handles
            time_scale_factor: Scaling factor for temporal dynamics (multiplied by stride_length)
            activation: Activation function to use
            bias: Whether to use bias
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.stride_length = stride_length
        self.time_scale_factor = time_scale_factor
        self.activation = activation
        self.use_bias = bias
        
        # Input weights
        self.input_kernel = Parameter(glorot_uniform((self.input_size, self.hidden_size)))
        
        # Hidden weights
        self.hidden_kernel = Parameter(glorot_uniform((self.hidden_size, self.hidden_size)))
        
        # Output weights
        self.output_kernel = Parameter(glorot_uniform((self.hidden_size, self.hidden_size)))
        
        # Bias terms
        if self.use_bias:
            self.input_bias = Parameter(ops.zeros((self.hidden_size,)))
            self.hidden_bias = Parameter(ops.zeros((self.hidden_size,)))
            self.output_bias = Parameter(ops.zeros((self.hidden_size,)))
        
        # Time constant
        self.tau = Parameter(ops.ones((self.hidden_size,)))
    
    @property
    def state_size(self):
        """Return the state size."""
        return self.hidden_size
    
    @property
    def output_size(self):
        """Return the output size."""
        return self.hidden_size
    
    def forward(self, inputs, state=None, elapsed_time=1.0):
        """
        Forward pass through the cell with stride-specific temporal scaling.
        
        Args:
            inputs: Input tensor of shape [batch_size, input_size]
            state: Previous state of shape [batch_size, hidden_size]
            elapsed_time: Time elapsed since last update (default: 1.0)
            
        Returns:
            Tuple of (output, new_state)
        """
        # Apply stride-specific temporal scaling
        effective_time = elapsed_time * self.stride_length * self.time_scale_factor
        
        # Initialize state if None
        if state is None:
            batch_size = ops.shape(inputs)[0]
            state = ops.zeros((batch_size, self.hidden_size))
        
        # Compute input contribution
        input_signal = ops.matmul(inputs, self.input_kernel.data)
        if self.use_bias:
            input_signal = ops.add(input_signal, self.input_bias.data)
        
        # Compute hidden contribution
        hidden_signal = ops.matmul(state, self.hidden_kernel.data)
        if self.use_bias:
            hidden_signal = ops.add(hidden_signal, self.hidden_bias.data)
        
        # Update state using time-scaled dynamics
        scale_factor = 1.0 / (ops.multiply(self.tau.data, effective_time))
        new_state = ops.add(
            state,
            ops.multiply(
                scale_factor,
                ops.subtract(ops.add(input_signal, hidden_signal), state)
            )
        )
        
        # Compute output
        output = ops.matmul(new_state, self.output_kernel.data)
        if self.use_bias:
            output = ops.add(output, self.output_bias.data)
        
        # Apply activation function
        if self.activation == "tanh":
            output = ops.tanh(output)
        elif self.activation == "relu":
            output = ops.relu(output)
        elif self.activation == "sigmoid":
            output = ops.sigmoid(output)
        else:
            output = getattr(ops, self.activation)(output)
        
        return output, new_state
    
    def reset_state(self, batch_size=1):
        """
        Reset the cell state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state
        """
        return ops.zeros((batch_size, self.hidden_size))