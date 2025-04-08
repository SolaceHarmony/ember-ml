"""
Recurrent Neural Network (RNN) Cell

This module provides an implementation of the basic RNN cell,
which is a simple recurrent neural network cell.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

from ember_ml import ops
from ember_ml.nn.initializers import glorot_uniform, orthogonal # Updated initializer path
from ember_ml.nn.modules import Parameter # Module removed
from ember_ml.nn.modules.module_cell import ModuleCell # Import ModuleCell
from ember_ml.nn import tensor
from ember_ml.nn.modules.activations import get_activation # Import the helper

class RNNCell(ModuleCell): # Inherit from ModuleCell
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
        # Call ModuleCell's init
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            activation=activation, # ModuleCell accepts activation name
            use_bias=bias, # ModuleCell accepts use_bias
            **kwargs
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.input_size, self.hidden_size, self.activation_name, self.use_bias
        # are set by parent init. self.activation (function) is also set by parent.
        
        # Initialize weights
        self._initialize_weights()
        
        # State size: [hidden_state]
        # state_size and output_size properties are inherited from ModuleCell
    
    def _initialize_weights(self):
        """Initialize the weights for the cell."""
        # Input weights
        self.input_kernel = Parameter(tensor.zeros((self.input_size, self.hidden_size)))
        
        # Recurrent weights
        self.recurrent_kernel = Parameter(tensor.zeros((self.hidden_size, self.hidden_size)))
        
        # Bias
        if self.use_bias:
            self.bias = Parameter(tensor.zeros((self.hidden_size,)))
        
        # Initialize weights
        self.input_kernel.data = glorot_uniform((self.input_size, self.hidden_size))
        self.recurrent_kernel.data = orthogonal((self.hidden_size, self.hidden_size))
        
        if self.use_bias:
            self.bias.data = tensor.zeros((self.hidden_size,))
    
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
            h_prev = tensor.zeros((tensor.shape(inputs)[0], self.hidden_size))
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
        
        # Apply activation function dynamically
        activation_fn = get_activation(self.activation) # Lookup happens here
        h_new = activation_fn(h_new)
        
        return h_new, [h_new]
    
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

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the RNN cell."""
        # Get config from ModuleCell (input_size, hidden_size, activation, use_bias)
        config = super().get_config()
        # RNNCell __init__ takes activation and use_bias.
        # ModuleCell saves activation_name as 'activation' and boolean as 'use_bias'.
        # All necessary args are already saved by parent get_config.
        return config

    # from_config can rely on ModuleCell's implementation