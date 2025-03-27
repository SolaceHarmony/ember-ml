"""
Long Short-Term Memory (LSTM) Cell

This module provides an implementation of the LSTM cell,
which is a type of recurrent neural network cell that can learn long-term dependencies.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

from ember_ml import ops
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn import tensor
class LSTMCell(Module):
    """
    Long Short-Term Memory (LSTM) cell.
    
    This cell implements a standard LSTM with input, forget, and output gates.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        **kwargs
    ):
        """
        Initialize the LSTM cell.
        
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
        
        # State size: [hidden_state, cell_state]
        self.state_size = [self.hidden_size, self.hidden_size]
        self.output_size = self.hidden_size
    
    def _initialize_weights(self):
        """Initialize the weights for the cell."""
        # Input weights
        self.input_kernel = Parameter(tensor.zeros((self.input_size, self.hidden_size * 4)))
        
        # Recurrent weights
        self.recurrent_kernel = Parameter(tensor.zeros((self.hidden_size, self.hidden_size * 4)))
        
        # Bias
        if self.use_bias:
            self.bias = Parameter(tensor.zeros((self.hidden_size * 4,)))
        
        # Initialize weights
        self.input_kernel.data = ops.glorot_uniform((self.input_size, self.hidden_size * 4))
        self.recurrent_kernel.data = ops.orthogonal((self.hidden_size, self.hidden_size * 4))
        
        if self.use_bias:
            # Initialize forget gate bias to 1.0 for better gradient flow
            bias_data = tensor.zeros((self.hidden_size * 4,))
            forget_gate_bias = bias_data[self.hidden_size:self.hidden_size*2]
            forget_gate_bias = tensor.ones_like(forget_gate_bias)
            bias_data = ops.tensor_scatter_nd_update(
                bias_data,
                ops.stack([ops.arange(self.hidden_size, self.hidden_size*2)], axis=1),
                forget_gate_bias
            )
            self.bias.data = bias_data
    
    def forward(self, inputs, states=None):
        """
        Forward pass through the cell.
        
        Args:
            inputs: Input tensor
            states: Previous states [hidden_state, cell_state]
            
        Returns:
            Tuple of (output, [new_hidden_state, new_cell_state])
        """
        # Initialize states if not provided
        if states is None:
            h_prev = tensor.zeros((tensor.shape(inputs)[0], self.hidden_size))
            c_prev = tensor.zeros((tensor.shape(inputs)[0], self.hidden_size))
        else:
            h_prev, c_prev = states
        
        # Compute gates
        z = ops.matmul(inputs, self.input_kernel)
        z = ops.add(z, ops.matmul(h_prev, self.recurrent_kernel))
        if self.use_bias:
            z = ops.add(z, self.bias)
        
        # Split into gates
        z_chunks = ops.split(z, 4, axis=-1)
        z_i, z_f, z_o, z_c = z_chunks
        
        # Apply activations
        i = ops.sigmoid(z_i)  # Input gate
        f = ops.sigmoid(z_f)  # Forget gate
        o = ops.sigmoid(z_o)  # Output gate
        c = ops.tanh(z_c)     # Cell input
        
        # Update cell state
        new_c = ops.add(ops.multiply(f, c_prev), ops.multiply(i, c))
        
        # Update hidden state
        new_h = ops.multiply(o, ops.tanh(new_c))
        
        return new_h, [new_h, new_c]
    
    def reset_state(self, batch_size=1):
        """
        Reset the cell state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Tuple of (hidden_state, cell_state)
        """
        h = tensor.zeros((batch_size, self.hidden_size))
        c = tensor.zeros((batch_size, self.hidden_size))
        return [h, c]