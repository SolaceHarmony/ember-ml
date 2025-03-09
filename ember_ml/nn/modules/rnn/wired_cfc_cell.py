"""
Wired Closed-form Continuous-time (CfC) Cell

This module provides an implementation of the Wired CfC cell,
which extends the CfC cell with support for custom wiring.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

from ember_ml import ops
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn.wirings import Wiring, NCPWiring, AutoNCP
from ember_ml.nn.modules.rnn.cfc_cell import CfCCell
from ember_ml.initializers import glorot_uniform, orthogonal

class WiredCfCCell(CfCCell):
    """
    CfC cell with custom wiring.
    
    This cell extends CfCCell with support for custom wiring,
    such as Neural Circuit Policies (NCPs).
    """
    
    def __init__(
        self,
        wiring,
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
        Initialize the Wired CfC cell.
        
        Args:
            wiring: Wiring configuration (e.g., AutoNCP)
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
        self.wiring = wiring
        units = wiring.units
        
        super().__init__(
            units=units,
            time_scale_factor=time_scale_factor,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            mixed_memory=mixed_memory,
            **kwargs
        )
    
    def _initialize_weights(self):
        """Initialize the weights for the cell with wiring constraints."""
        # Build wiring masks
        self.input_mask, self.recurrent_mask, self.output_mask = self.wiring.build()
        
        # Convert masks to tensors
        self.input_mask = ops.convert_to_tensor(self.input_mask, dtype=ops.float32)
        self.recurrent_mask = ops.convert_to_tensor(self.recurrent_mask, dtype=ops.float32)
        self.output_mask = ops.convert_to_tensor(self.output_mask, dtype=ops.float32)
        
        # Create kernel parameters first
        # Input weights
        if self.kernel_initializer == "glorot_uniform":
            kernel_data = glorot_uniform((self.units, self.units * 4))
        else:
            kernel_data = ops.zeros((self.units, self.units * 4))
        self.kernel = Parameter(kernel_data)
        
        # Recurrent weights
        if self.recurrent_initializer == "orthogonal":
            recurrent_data = orthogonal((self.units, self.units * 4))
        else:
            recurrent_data = ops.zeros((self.units, self.units * 4))
        self.recurrent_kernel = Parameter(recurrent_data)
        
        # Bias
        if self.use_bias:
            self.bias = Parameter(ops.zeros((self.units * 4,)))
        
        # Time-scale parameter (learnable)
        self.time_scale = Parameter(ops.ones((self.units,)) * self.time_scale_factor)
        
        # Apply masks to weights
        # Create expanded masks by concatenating the mask 4 times along the last axis
        self.kernel_mask = ops.concatenate([self.input_mask] * 4, axis=-1)
        self.recurrent_kernel_mask = ops.concatenate([self.recurrent_mask] * 4, axis=-1)
    
    def forward(self, inputs, states=None, timespans=None):
        """
        Forward pass through the cell with wiring constraints.
        
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
        
        # Apply wiring constraints
        masked_kernel = ops.multiply(self.kernel, self.kernel_mask)
        masked_recurrent_kernel = ops.multiply(self.recurrent_kernel, self.recurrent_kernel_mask)
        
        # Compute gates with wiring constraints
        z = ops.matmul(inputs, masked_kernel)
        z = ops.add(z, ops.matmul(h_prev, masked_recurrent_kernel))
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
        
        # Apply output mask
        output = ops.multiply(h, self.output_mask)
        
        return output, [h, t]