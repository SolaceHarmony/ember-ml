"""
Closed-form Continuous-time (CfC) Cell

This module provides an implementation of the CfC cell,
which is a type of recurrent neural network cell that operates in continuous time.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

from ember_ml import ops
from ember_ml.initializers import glorot_uniform, orthogonal
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn.modules.module_cell import ModuleCell

class CfCCell(ModuleCell):
    """
    Closed-form Continuous-time (CfC) cell.
    
    This cell implements a continuous-time recurrent neural network
    with closed-form solution for the hidden state dynamics.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mode: str = "default",
        backbone_activation: str = "tanh",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.0,
        sparsity_mask = None,
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
            input_size: Size of the input
            hidden_size: Size of the hidden state
            mode: Mode of operation ("default", "pure", or "no_gate")
            backbone_activation: Activation function for the backbone
            backbone_units: Number of units in the backbone
            backbone_layers: Number of layers in the backbone
            backbone_dropout: Dropout rate for the backbone
            sparsity_mask: Mask for sparse connections
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
        
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            activation=activation,
            use_bias=use_bias,
            **kwargs
        )
        
        # Store parameters
        self.mode = mode
        self.backbone_activation = backbone_activation
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.sparsity_mask = sparsity_mask
        self.time_scale_factor = time_scale_factor
        self.recurrent_activation = recurrent_activation
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.mixed_memory = mixed_memory
        
        # Initialize weights
        self._initialize_weights()
        
        # For backward compatibility
        self.units = self.hidden_size
        
        # Store the actual state size for CfC
        self._cfc_state_size = [self.hidden_size, self.hidden_size]
    
    @property
    def state_size(self) -> List[int]:
        """Return the size of the cell state."""
        return self._cfc_state_size
    
    @property
    def output_size(self) -> int:
        """Return the output size."""
        return self.units
    
    def _initialize_weights(self):
        """Initialize the weights for the cell."""
        # Input weights
        self.kernel = Parameter(tensor.zeros((self.units, self.units * 4)))
        
        # Recurrent weights
        self.recurrent_kernel = Parameter(tensor.zeros((self.units, self.units * 4)))
        
        # Bias
        if self.use_bias:
            self.bias = Parameter(tensor.zeros((self.units * 4,)))
        
        # Time-scale parameter (learnable)
        self.time_scale = Parameter(ops.ones((self.units,)) * self.time_scale_factor)
        
        # Initialize weights
        if self.kernel_initializer == "glorot_uniform":
            self.kernel.data = glorot_uniform((self.units, self.units * 4))
        
        if self.recurrent_initializer == "orthogonal":
            self.recurrent_kernel.data = orthogonal((self.units, self.units * 4))
        
        if self.use_bias and self.bias_initializer == "zeros":
            self.bias.data = tensor.zeros((self.units * 4,))
    
    def forward(self, inputs, state=None, **kwargs):
        """
        Forward pass through the cell.
        
        Args:
            inputs: Input tensor
            state: Previous state [hidden_state, time_state]
            **kwargs: Additional arguments, including:
                timespans: Time spans for continuous-time dynamics (default: 1.0)
            
        Returns:
            Tuple of (output, [new_hidden_state, new_time_state])
        """
        # Extract timespans from kwargs
        timespans = kwargs.get('timespans', None)
        
        # Initialize states if not provided
        if state is None:
            h_prev = tensor.zeros((ops.shape(inputs)[0], self.units))
            t_prev = tensor.zeros((ops.shape(inputs)[0], self.units))
        else:
            h_prev, t_prev = state
        
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
            activation_fn = getattr(ops, self.recurrent_activation)
            i = activation_fn(z_i)
            f = activation_fn(z_f)
            o = activation_fn(z_o)
        
        if self.activation == "tanh":
            c = ops.tanh(z_c)     # Cell input
        else:
            activation_fn = getattr(ops, self.activation_name)
            c = activation_fn(z_c)
        
        # Apply time scaling
        # Compute time decay factor
        decay = ops.exp(ops.divide(-ts,self.time_scale))
        
        # Update time state
        t = ops.add(ops.multiply(f, t_prev), ops.multiply(i, c))
        
        # Apply time decay to hidden state
        decay_term = ops.multiply(decay, h_prev)
        time_term = ops.multiply(ops.subtract(ops.ones_like(decay), decay), t)
        
        if self.activation_name == "tanh":
            h = ops.multiply(o, ops.tanh(ops.add(decay_term, time_term)))
        else:
            activation_fn = getattr(ops, self.activation_name)
            h = ops.multiply(o, activation_fn(ops.add(decay_term, time_term)))
        
        return h, [h, t]
    
    def reset_state(self, batch_size=1):
        """
        Reset the cell state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Tuple of (hidden_state, time_state)
        """
        h = tensor.zeros((batch_size, self.units))
        t = tensor.zeros((batch_size, self.units))
        return [h, t]