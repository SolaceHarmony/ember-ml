"""
Long Short-Term Memory (LSTM) Cell

This module provides an implementation of the LSTM cell,
which is a type of recurrent neural network cell that can learn long-term dependencies.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

from ember_ml import ops
from ember_ml.nn.modules import Parameter # Module removed
from ember_ml.nn.modules.module_cell import ModuleCell
from ember_ml.nn import initializers # Import initializers
from ember_ml.nn import tensor
class LSTMCell(ModuleCell): # Inherit from ModuleCell
    """
    Long Short-Term Memory (LSTM) cell.
    
    This cell implements a standard LSTM with input, forget, and output gates.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True, # Match ModuleCell arg name
        **kwargs
    ):
        """
        Initialize the LSTM cell.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            use_bias: Whether to use bias
            **kwargs: Additional keyword arguments
        """
        # Call ModuleCell's init with explicitly provided use_bias
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            use_bias=use_bias,
            **kwargs
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.input_size and self.hidden_size are set by parent init
        # self.use_bias is set by parent init
        
        # Initialize weights
        self._initialize_weights()
        
        # State size: [hidden_state, cell_state]
        # state_size and output_size properties are inherited from ModuleCell
        # No need to set them explicitly here unless overriding
    
    def _initialize_weights(self):
        """Initialize the weights for the cell."""
        # Input weights
        self.input_kernel = Parameter(tensor.zeros((self.input_size, self.hidden_size * 4)))
        
        # Recurrent weights
        self.recurrent_kernel = Parameter(tensor.zeros((self.hidden_size, self.hidden_size * 4)))
        
        # Bias
        # Use self.use_bias set by parent
        if self.use_bias:
            self.bias = Parameter(tensor.zeros((self.hidden_size * 4,)))
        else:
            self.bias = None # Ensure bias is None if not used
        
        # Initialize weights using functions from initializers module
        self.input_kernel.data = initializers.glorot_uniform((self.input_size, self.hidden_size * 4))
        self.recurrent_kernel.data = initializers.orthogonal((self.hidden_size, self.hidden_size * 4))
        
        if self.use_bias:
            # Initialize forget gate bias to 1.0 for better gradient flow
            bias_data = tensor.zeros((self.hidden_size * 4,))
            forget_gate_bias = bias_data[self.hidden_size:self.hidden_size*2]
            forget_gate_bias = tensor.ones_like(forget_gate_bias)
            bias_data = tensor.tensor_scatter_nd_update(
                bias_data,
                tensor.stack([tensor.arange(self.hidden_size, self.hidden_size*2)], axis=1),
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
        if self.use_bias and self.bias is not None:
            z = ops.add(z, self.bias)
        
        # Split into gates
        z_chunks = tensor.split(z, 4, axis=-1)
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

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the LSTM cell."""
        # Get config from ModuleCell (input_size, hidden_size, activation, use_bias)
        config = super().get_config()
        # LSTMCell doesn't add new args to config beyond what ModuleCell handles
        # We might want to remove activation if it's not used/configurable in LSTMCell
        config.pop('activation', None)
        
        # Ensure use_bias is correctly saved with the actual value used
        config['use_bias'] = self.use_bias
        
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LSTMCell':
        """Creates an LSTM cell from its configuration."""
        # ModuleCell saves 'use_bias', __init__ expects 'use_bias'
        # The base from_config should handle this mapping correctly via **config
        if 'bias' in config and 'use_bias' not in config:
             config['use_bias'] = config.pop('bias') # Map if old key was used

        # Remove activation if present, as LSTMCell doesn't use it in init
        config.pop('activation', None)
        
        # Make sure use_bias is properly passed and not overridden
        use_bias = config.get('use_bias', True)
        
        # Create a new instance with the extracted config
        return cls(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            use_bias=use_bias
        )