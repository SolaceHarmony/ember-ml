"""
Closed-form Continuous-time (CfC) Neural Network

This module provides an implementation of CfC cells and layers,
which are a type of recurrent neural network that operates in continuous time.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.wirings import Wiring, NCPWiring, AutoNCP
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
            self.kernel.data = ops.glorot_uniform((self.units, self.units * 4))
        
        if self.recurrent_initializer == "orthogonal":
            self.recurrent_kernel.data = ops.orthogonal((self.units, self.units * 4))
        
        if self.use_bias and self.bias_initializer == "zeros":
            self.bias.data = tensor.zeros((self.units * 4,))
    
    def forward(self, inputs, states=None):
        """
        Forward pass through the cell.
        
        Args:
            inputs: Input tensor
            states: Previous states [hidden_state, time_state]
            
        Returns:
            Tuple of (output, [new_hidden_state, new_time_state])
        """
        # Initialize states if not provided
        if states is None:
            h_prev = tensor.zeros((ops.shape(inputs)[0], self.units))
            t_prev = tensor.zeros((ops.shape(inputs)[0], self.units))
        else:
            h_prev, t_prev = states
        
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
        decay = ops.exp(-1.0 / self.time_scale)
        
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
        h = tensor.zeros((batch_size, self.units))
        t = tensor.zeros((batch_size, self.units))
        return [h, t]

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
        self.input_mask = tensor.convert_to_tensor(self.input_mask, dtype=ops.float32)
        self.recurrent_mask = tensor.convert_to_tensor(self.recurrent_mask, dtype=ops.float32)
        self.output_mask = tensor.convert_to_tensor(self.output_mask, dtype=ops.float32)
        
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
            self.kernel.data = ops.glorot_uniform((self.units, self.units * 4))
        
        if self.recurrent_initializer == "orthogonal":
            self.recurrent_kernel.data = ops.orthogonal((self.units, self.units * 4))
        
        if self.use_bias and self.bias_initializer == "zeros":
            self.bias.data = tensor.zeros((self.units * 4,))
        
        # Apply masks to weights
        self.kernel_mask = ops.repeat(self.input_mask, 4, axis=-1)
        self.recurrent_kernel_mask = ops.repeat(self.recurrent_mask, 4, axis=-1)
    
    def forward(self, inputs, states=None):
        """
        Forward pass through the cell with wiring constraints.
        
        Args:
            inputs: Input tensor
            states: Previous states [hidden_state, time_state]
            
        Returns:
            Tuple of (output, [new_hidden_state, new_time_state])
        """
        # Initialize states if not provided
        if states is None:
            h_prev = tensor.zeros((ops.shape(inputs)[0], self.units))
            t_prev = tensor.zeros((ops.shape(inputs)[0], self.units))
        else:
            h_prev, t_prev = states
        
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
        decay = ops.exp(-1.0 / self.time_scale)
        
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

class CfC(Module):
    """
    Closed-form Continuous-time (CfC) RNN layer.
    
    This layer wraps a CfCCell or WiredCfCCell to create a recurrent layer.
    """
    
    def __init__(
        self,
        cell_or_units,
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: bool = False,
        mixed_memory: bool = False,
        **kwargs
    ):
        """
        Initialize the CfC layer.
        
        Args:
            cell_or_units: CfCCell, WiredCfCCell, or number of units
            return_sequences: Whether to return the full sequence or just the last output
            return_state: Whether to return the final state
            go_backwards: Whether to process the sequence backwards
            mixed_memory: Whether to use mixed memory
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        
        # Handle different types of input
        if isinstance(cell_or_units, (CfCCell, WiredCfCCell)):
            self.cell = cell_or_units
        elif isinstance(cell_or_units, Wiring):
            self.cell = WiredCfCCell(wiring=cell_or_units, mixed_memory=mixed_memory)
        elif isinstance(cell_or_units, int):
            self.cell = CfCCell(units=cell_or_units, mixed_memory=mixed_memory)
        else:
            raise ValueError("cell_or_units must be a CfCCell, WiredCfCCell, Wiring, or int")
        
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.mixed_memory = mixed_memory
    
    def forward(self, inputs, initial_state=None):
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input tensor
            initial_state: Initial state
            
        Returns:
            Layer output
        """
        # Get input shape
        input_shape = ops.shape(inputs)
        batch_size, time_steps, input_dim = input_shape[0], input_shape[1], input_shape[2]
        
        # Create initial state if not provided
        if initial_state is None:
            initial_state = self.cell.reset_state(batch_size)
        
        # Process sequence
        outputs = []
        states = initial_state
        
        # Process sequence in reverse if go_backwards is True
        if self.go_backwards:
            time_range = range(time_steps - 1, -1, -1)
        else:
            time_range = range(time_steps)
        
        # Process each time step
        for t in time_range:
            output, states = self.cell(inputs[:, t], states)
            outputs.append(output)
        
        # Stack outputs
        if self.return_sequences:
            outputs = ops.stack(outputs, axis=1)
        else:
            outputs = outputs[-1]
        
        # Return outputs and states if requested
        if self.return_state:
            return outputs, states
        else:
            return outputs
    
    def reset_state(self, batch_size=1):
        """
        Reset the layer state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state
        """
        return self.cell.reset_state(batch_size)