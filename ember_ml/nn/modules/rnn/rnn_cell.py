"""
Recurrent Neural Network (RNN) Cell

This module provides an implementation of the basic RNN cell,
which is a simple recurrent neural network cell.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

from ember_ml import ops
from ember_ml.nn.modules import Module, Parameter

class RNNCell(Module):
    """
    Basic Recurrent Neural Network (RNN) cell.
    
    This cell implements a simple RNN with a single activation function.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: str = "tanh",
        bias: bool = True,
        **kwargs
    ):
        """
        Initialize the RNN cell.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            activation: Activation function to use
            bias: Whether to use bias
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.use_bias = bias
        
        # Initialize weights
        self._initialize_weights()
        
        # State size: [hidden_state]
        self.state_size = [self.hidden_size]
        self.output_size = self.hidden_size
    
    def _initialize_weights(self):
        """Initialize the weights for the cell."""
        # Input weights
        self.input_kernel = Parameter(ops.zeros((self.input_size, self.hidden_size)))
        
        # Recurrent weights
        self.recurrent_kernel = Parameter(ops.zeros((self.hidden_size, self.hidden_size)))
        
        # Bias
        if self.use_bias:
            self.bias = Parameter(ops.zeros((self.hidden_size,)))
        
        # Initialize weights
        self.input_kernel.data = ops.glorot_uniform((self.input_size, self.hidden_size))
        self.recurrent_kernel.data = ops.orthogonal((self.hidden_size, self.hidden_size))
        
        if self.use_bias:
            self.bias.data = ops.zeros((self.hidden_size,))
    
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
            h_prev = ops.zeros((ops.shape(inputs)[0], self.hidden_size))
        else:
            h_prev = states[0]
        
        # Compute linear transformation
        x = ops.matmul(inputs, self.input_kernel)
        h = ops.matmul(h_prev, self.recurrent_kernel)
        
        # Add bias if needed
        if self.use_bias:
            x = ops.add(x, self.bias)
        
        # Compute new hidden state
        h_new = ops.add(x, h)
        
        # Apply activation function
        if self.activation == "tanh":
            h_new = ops.tanh(h_new)
        elif self.activation == "relu":
            h_new = ops.relu(h_new)
        elif self.activation == "sigmoid":
            h_new = ops.sigmoid(h_new)
        else:
            h_new = getattr(ops, self.activation)(h_new)
        
        return h_new, [h_new]
    
    def reset_state(self, batch_size=1):
        """
        Reset the cell state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state
        """
        h = ops.zeros((batch_size, self.hidden_size))
        return [h]