# ember_ml/nn/modules/rnn/wired_cfc_cell.py

"""
Wired Closed-form Continuous-time (CfC) Cell
"""
from typing import Optional, List, Dict, Any, Union, Tuple

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules import Parameter
from ember_ml.nn.modules.wiring import NeuronMap # Use renamed base class
from ember_ml.nn.modules.module_wired_cell import ModuleWiredCell # Import base wired cell class
from ember_ml.nn.modules.rnn.cfc_cell import CfCCell # Import for reference
from ember_ml.nn.initializers import glorot_uniform, orthogonal

class WiredCfCCell(ModuleWiredCell):
    """
    CfC cell with custom wiring.

    This cell extends CfCCell with support for custom wiring,
    such as Neural Circuit Policies (NCPs).
    """
    def __init__(
        self,
        neuron_map: NeuronMap, # Updated parameter name
        time_scale_factor: float = 1.0,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        mixed_memory: bool = False,
        input_size: Optional[int] = None, # Will be passed to ModuleWiredCell
        mode: str = "default", # Added for ModuleWiredCell
        **kwargs
    ):
        """
        Initialize the Wired CfC cell.

        Args:
            neuron_map: Neuron map configuration (e.g., AutoNCP)
            time_scale_factor: Factor to scale the time constant
            activation: Activation function for the output
            recurrent_activation: Activation function for the recurrent step
            use_bias: Whether to use bias
            kernel_initializer: Initializer for the kernel weights
            recurrent_initializer: Initializer for the recurrent weights
            bias_initializer: Initializer for the bias
            mixed_memory: Whether to use mixed memory
            input_size: Optional input size to build neuron_map if not already built
            mode: Mode of operation
            **kwargs: Additional keyword arguments
        """
        # Call ModuleWiredCell's __init__ first - this will handle building the neuron map if needed
        super().__init__(
            input_size=input_size,
            neuron_map=neuron_map,
            mode=mode,
            **kwargs
        )
        
        # Store CfC-specific parameters
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

    def _initialize_wired_weights(self):
        """Initialize the masks based on wiring and potentially reshape parent weights."""
        # Get masks (ensure neuron_map is built)
        input_mask_np, recurrent_mask_np, output_mask_np = self.neuron_map.get_input_mask(), self.neuron_map.get_recurrent_mask(), self.neuron_map.get_output_mask()

        # Convert masks to tensors - Use float32 for potential gradient flow
        dtype = tensor.float32
        self.input_mask = tensor.convert_to_tensor(input_mask_np, dtype=dtype)
        self.recurrent_mask = tensor.convert_to_tensor(recurrent_mask_np, dtype=dtype)
        self.output_mask = tensor.convert_to_tensor(output_mask_np, dtype=dtype)

        # Get dimensions from neuron_map
        input_dim = self.neuron_map.input_dim
        units = self.neuron_map.units
        expected_kernel_shape = (input_dim, units * 4)
        expected_recurrent_shape = (units, units * 4)

        # Initialize kernel
        self.kernel = Parameter(tensor.zeros(expected_kernel_shape))
        if self.kernel_initializer == "glorot_uniform":
            self.kernel.data = glorot_uniform(expected_kernel_shape)

        # Initialize recurrent kernel
        self.recurrent_kernel = Parameter(tensor.zeros(expected_recurrent_shape))
        if self.recurrent_initializer == "orthogonal":
            self.recurrent_kernel.data = orthogonal(expected_recurrent_shape)
        
        # Initialize bias if needed
        if self.use_bias:
            self.bias = Parameter(tensor.zeros((units * 4,)))

        # Create broadcastable masks for applying constraints
        # Input mask aligns with input_dim -> kernel [input_dim, units*4]
        # Need shape [input_dim, 1] to broadcast across the units*4 dimension
        self.kernel_mask = tensor.expand_dims(self.input_mask, axis=1)

        # For the recurrent mask, we'll just use a simple approach that works with all backends
        self.recurrent_kernel_mask = tensor.ones(expected_recurrent_shape)

        # Output mask to [1, units] for broadcasting with hidden state
        self.output_mask_broadcast = tensor.reshape(self.output_mask, (1, units))

    def _initialize_weights(self):
        """Initialize all weights for the cell."""
        self._initialize_wired_weights()

    def forward(self, inputs, states=None):
        """
        Forward pass through the cell with wiring constraints.

        Args:
            inputs: Input tensor shape (batch, input_dim)
            states: Previous states [hidden_state, time_state]

        Returns:
            Tuple of (output, [new_hidden_state, new_time_state])
        """
        if self.neuron_map.input_dim != tensor.shape(inputs)[-1]:
            raise ValueError(f"Input tensor last dimension ({tensor.shape(inputs)[-1]}) "
                            f"does not match neuron_map input_dim ({self.neuron_map.input_dim})")

        # Initialize states if not provided
        if states is None:
            states = self.reset_state(batch_size=tensor.shape(inputs)[0])
        h_prev, t_prev = states

        # Apply wiring constraints to weights before matmul
        # Broadcasting happens here: kernel[in, u*4] * kernel_mask[in, 1]
        masked_kernel = ops.multiply(self.kernel, self.kernel_mask)
        # Element-wise multiplication: rec_kernel[u, u*4] * rec_kernel_mask[u, u*4]
        masked_recurrent_kernel = ops.multiply(self.recurrent_kernel, self.recurrent_kernel_mask)

        # Compute gates with wired constraints
        z = ops.matmul(inputs, masked_kernel) # Input projection
        z = ops.add(z, ops.matmul(h_prev, masked_recurrent_kernel)) # Recurrent projection
        if self.use_bias:
            z = ops.add(z, self.bias)

        # Split into gates
        z_chunks = tensor.split(z, 4, axis=-1)
        z_i, z_f, z_o, z_c = z_chunks

        # Apply activations (using parent's activation attributes)
        rec_activation_fn = ops.get_activation(self.recurrent_activation)
        i = rec_activation_fn(z_i)  # Input gate
        f = rec_activation_fn(z_f)  # Forget gate
        o = rec_activation_fn(z_o)  # Output gate

        activation_fn = ops.get_activation(self.activation)
        c = activation_fn(z_c)     # Cell input

        # Apply time scaling
        # Compute time decay factor based on time_scale_factor
        decay = ops.exp(-1.0 / self.time_scale_factor) # Use time_scale_factor directly

        # Update time state
        t = ops.add(ops.multiply(f, t_prev), ops.multiply(i, c))

        # Apply time decay to hidden state
        decay_term = ops.multiply(decay, h_prev)
        time_term = ops.multiply(ops.subtract(tensor.ones_like(decay), decay), t)

        # Compute new hidden state
        h = ops.multiply(o, activation_fn(ops.add(decay_term, time_term)))

        # Apply output mask to the final hidden state activation 'h'
        # Broadcasting: h[b, u] * output_mask_broadcast[1, u]
        output = ops.multiply(h, self.output_mask_broadcast)

        # Extract motor neurons if output_dim < units
        if self.neuron_map.output_dim < self.neuron_map.units:
            output = output[:, :self.neuron_map.output_dim]

        return output, [h, t]
    
    def reset_state(self, batch_size=1):
        """Reset the cell state."""
        units = self.neuron_map.units
        h = tensor.zeros((batch_size, units))
        t = tensor.zeros((batch_size, units))
        return [h, t]

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the WiredCfC cell."""
        # Get config from ModuleWiredCell
        config = super().get_config()
        
        # Add CfC-specific parameters
        config.update({
            "time_scale_factor": self.time_scale_factor,
            "activation": self.activation,
            "recurrent_activation": self.recurrent_activation,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "recurrent_initializer": self.recurrent_initializer,
            "bias_initializer": self.bias_initializer,
            "mixed_memory": self.mixed_memory
        })
        
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'WiredCfCCell':
        """Creates a WiredCfC cell from its configuration."""
        # Remove input_size to avoid duplicate parameter error
        config.pop('input_size', None)
        # ModuleWiredCell.from_config will handle reconstructing the neuron_map
        return super(WiredCfCCell, cls).from_config(config)