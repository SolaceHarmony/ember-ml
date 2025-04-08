"""
Gated Recurrent Unit (GRU) Cell

This module provides an implementation of the GRU cell,
which is a type of recurrent neural network cell that can learn long-term dependencies
with fewer parameters than LSTM.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

from ember_ml import ops
from ember_ml.nn.modules import Parameter # Module removed
from ember_ml.nn.modules.module_cell import ModuleCell # Import ModuleCell
from ember_ml.nn import initializers # Import initializers module
from ember_ml.nn.modules import activations # Import activations module
from ember_ml.nn import tensor
class GRUCell(ModuleCell): # Inherit from ModuleCell
    """
    Gated Recurrent Unit (GRU) cell.
    
    This cell implements a standard GRU with reset and update gates.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True, # Match ModuleCell arg name
        **kwargs
    ):
        """
        Initialize the GRU cell.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            use_bias: Whether to use bias
            **kwargs: Additional keyword arguments
        """
        # Call ModuleCell's init
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            # GRU doesn't typically have a separate activation param like ModuleCell
            # Pass other relevant args if needed
            **kwargs
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.input_size and self.hidden_size are set by parent init
        # self.use_bias is set by parent init
        
        # Initialize weights
        self._initialize_weights()
        
        # State size: [hidden_state]
        # state_size and output_size properties are inherited from ModuleCell
        # No need to set them explicitly here
    
    def _initialize_weights(self):
        """Initialize the weights for the cell."""
        # Input weights
        self.input_kernel = Parameter(tensor.zeros((self.input_size, self.hidden_size * 3)))
        
        # Recurrent weights
        self.recurrent_kernel = Parameter(tensor.zeros((self.hidden_size, self.hidden_size * 3)))
        
        # Bias
        # Use self.use_bias set by parent
        if self.use_bias:
            self.bias = Parameter(tensor.zeros((self.hidden_size * 3,)))
            self.recurrent_bias = Parameter(tensor.zeros((self.hidden_size * 3,)))
        else:
            self.bias = None
            self.recurrent_bias = None
        
        # Initialize weights
        self.input_kernel.data = initializers.glorot_uniform((self.input_size, self.hidden_size * 3))
        self.recurrent_kernel.data = initializers.orthogonal((self.hidden_size, self.hidden_size * 3))
        
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
            h_prev = tensor.zeros((tensor.shape(inputs)[0], self.hidden_size))
        else:
            h_prev = states[0]
        
        # Compute input projection
        x_z = ops.matmul(inputs, self.input_kernel.data[:, :self.hidden_size])
        x_r = ops.matmul(inputs, self.input_kernel.data[:, self.hidden_size:self.hidden_size*2])
        x_h = ops.matmul(inputs, self.input_kernel.data[:, self.hidden_size*2:])
        
        # Compute recurrent projection
        h_z = ops.matmul(h_prev, self.recurrent_kernel.data[:, :self.hidden_size])
        h_r = ops.matmul(h_prev, self.recurrent_kernel.data[:, self.hidden_size:self.hidden_size*2])
        h_h = ops.matmul(h_prev, self.recurrent_kernel.data[:, self.hidden_size*2:])
        
        # Add bias if needed
        if self.use_bias and self.bias is not None and self.recurrent_bias is not None:
            x_z = ops.add(x_z, self.bias.data[:self.hidden_size])
            x_r = ops.add(x_r, self.bias.data[self.hidden_size:self.hidden_size*2])
            x_h = ops.add(x_h, self.bias.data[self.hidden_size*2:])

            h_z = ops.add(h_z, self.recurrent_bias.data[:self.hidden_size])
            h_r = ops.add(h_r, self.recurrent_bias.data[self.hidden_size:self.hidden_size*2])
            h_h = ops.add(h_h, self.recurrent_bias.data[self.hidden_size*2:])
        
        # Compute gates
        z = activations.sigmoid(ops.add(x_z, h_z))  # Update gate
        r = activations.sigmoid(ops.add(x_r, h_r))  # Reset gate
        
        # Compute candidate hidden state
        h_tilde = activations.tanh(ops.add(x_h, ops.multiply(r, h_h)))
        
        # Compute new hidden state
        h = ops.add(
            ops.multiply(z, h_prev),
            ops.multiply(ops.subtract(tensor.ones_like(z), z), h_tilde)
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

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the GRU cell."""
        # Get config from ModuleCell (input_size, hidden_size, activation, use_bias)
        config = super().get_config()
        # GRUCell doesn't use 'activation' param from ModuleCell init, remove it
        config.pop('activation', None)
        # GRUCell __init__ takes use_bias, which ModuleCell provides. No change needed.
        return config

    # from_config can rely on ModuleCell's implementation