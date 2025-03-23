"""
Gated Recurrent Unit (GRU) Cell

This module provides an implementation of the GRU cell,
which is a type of recurrent neural network cell that can learn long-term dependencies
with fewer parameters than LSTM.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

from ember_ml import ops
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn import tensor
class GRUCell(Module):
    """
    Gated Recurrent Unit (GRU) cell.
    
    This cell implements a standard GRU with reset and update gates.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        **kwargs
    ):
        """
        Initialize the GRU cell.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            bias: Whether to use bias
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias
        
        # Initialize weights
        self._initialize_weights()
        
        # State size: [hidden_state]
        self.state_size = [self.hidden_size]
        self.output_size = self.hidden_size
    
    def _initialize_weights(self):
        """Initialize the weights for the cell."""
        # Input weights
        self.input_kernel = Parameter(tensor.zeros((self.input_size, self.hidden_size * 3)))
        
        # Recurrent weights
        self.recurrent_kernel = Parameter(tensor.zeros((self.hidden_size, self.hidden_size * 3)))
        
        # Bias
        if self.use_bias:
            self.bias = Parameter(tensor.zeros((self.hidden_size * 3,)))
            self.recurrent_bias = Parameter(tensor.zeros((self.hidden_size * 3,)))
        
        # Initialize weights
        self.input_kernel.data = ops.glorot_uniform((self.input_size, self.hidden_size * 3))
        self.recurrent_kernel.data = ops.orthogonal((self.hidden_size, self.hidden_size * 3))
        
        if self.use_bias:
            self.bias.data = tensor.zeros((self.hidden_size * 3,))
            self.recurrent_bias.data = tensor.zeros((self.hidden_size * 3,))
    
    def forward(self, inputs, states=None):
        """
        Forward pass through the cell.
        
        Args:
            inputs: Input tensor
            states: Previous state
            
        Returns:
            Tuple of (output, [new_hidden_state])
        """
        # Initialize states if not provided
        if states is None:
            h_prev = tensor.zeros((ops.shape(inputs)[0], self.hidden_size))
        else:
            h_prev = states[0]
        
        # Compute input projection
        x_z = ops.matmul(inputs, self.input_kernel[:, :self.hidden_size])
        x_r = ops.matmul(inputs, self.input_kernel[:, self.hidden_size:self.hidden_size*2])
        x_h = ops.matmul(inputs, self.input_kernel[:, self.hidden_size*2:])
        
        # Compute recurrent projection
        h_z = ops.matmul(h_prev, self.recurrent_kernel[:, :self.hidden_size])
        h_r = ops.matmul(h_prev, self.recurrent_kernel[:, self.hidden_size:self.hidden_size*2])
        h_h = ops.matmul(h_prev, self.recurrent_kernel[:, self.hidden_size*2:])
        
        # Add bias if needed
        if self.use_bias:
            x_z = ops.add(x_z, self.bias[:self.hidden_size])
            x_r = ops.add(x_r, self.bias[self.hidden_size:self.hidden_size*2])
            x_h = ops.add(x_h, self.bias[self.hidden_size*2:])
            
            h_z = ops.add(h_z, self.recurrent_bias[:self.hidden_size])
            h_r = ops.add(h_r, self.recurrent_bias[self.hidden_size:self.hidden_size*2])
            h_h = ops.add(h_h, self.recurrent_bias[self.hidden_size*2:])
        
        # Compute gates
        z = ops.sigmoid(ops.add(x_z, h_z))  # Update gate
        r = ops.sigmoid(ops.add(x_r, h_r))  # Reset gate
        
        # Compute candidate hidden state
        h_tilde = ops.tanh(ops.add(x_h, ops.multiply(r, h_h)))
        
        # Compute new hidden state
        h = ops.add(
            ops.multiply(z, h_prev),
            ops.multiply(ops.subtract(ops.ones_like(z), z), h_tilde)
        )
        
        return h, [h]
    
    def reset_state(self, batch_size=1):
        """
        Reset the cell state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state
        """
        h = tensor.zeros((batch_size, self.hidden_size))
        return [h]