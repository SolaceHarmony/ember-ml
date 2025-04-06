"""
Neural Circuit Policy (NCP) module.

This module provides the NCP class, which implements a neural circuit policy
using a wiring configuration.
"""

from typing import Optional, Tuple, Dict, Any, Union

from ember_ml import ops
from ember_ml.ops import linearalg
from ember_ml.nn import tensor
from ember_ml.nn.modules.base_module import BaseModule as Module, Parameter
from ember_ml.nn.modules.wiring import NeuronMap # Use renamed base class

class NCP(Module):
    """
    Neural Circuit Policy (NCP) module.
    
    This module implements a neural circuit policy using a wiring configuration.
    It consists of a recurrent neural network with a specific connectivity pattern
    defined by the wiring configuration.
    """
    
    def __init__(
        self,
        neuron_map: NeuronMap, # Changed argument name
        activation: str = "tanh",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        dtype: Optional[Any] = None,
    ):
        """
        Initialize an NCP module.
        
        Args:
            neuron_map: NeuronMap configuration object
            activation: Activation function to use
            use_bias: Whether to use bias
            kernel_initializer: Initializer for the kernel weights
            recurrent_initializer: Initializer for the recurrent weights
            bias_initializer: Initializer for the bias weights
            dtype: Data type for the weights
        """
        super().__init__()
        
        self.neuron_map = neuron_map # Changed attribute name
        self.activation_name = activation
        self.activation = ops.get_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.dtype = dtype
        
        # Defer mask and weight initialization to build method
        self.input_mask = None
        self.recurrent_mask = None
        self.output_mask = None
        self._kernel = None
        self._recurrent_kernel = None
        self._bias = None
        self.built = False # Track build status of the layer
        
        # Initialize state
        self.state = tensor.zeros((1, self.neuron_map.units))
    
    @property
    def kernel(self):
        """Get the kernel parameter."""
        return self._kernel.data
    
    @property
    def recurrent_kernel(self):
        """Get the recurrent kernel parameter."""
        return self._recurrent_kernel.data
    
    @property
    def bias(self):
        """Get the bias parameter."""
        if self.use_bias and self._bias is not None:
            return self._bias.data
        return None
    

    def build(self, input_shape):
        """
        Build the layer's weights and masks based on the input shape.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        if self.built:
            return

        # Ensure input_shape is a tuple or list
        if not isinstance(input_shape, (tuple, list)):
             raise TypeError(f"Expected input_shape to be a tuple or list, got {type(input_shape)}")

        if len(input_shape) < 1:
             raise ValueError(f"Input shape must have at least one dimension, got {input_shape}")

        input_dim = input_shape[-1]

        # Build the neuron map if it hasn't been built or if input_dim changed
        if not self.neuron_map.is_built() or self.neuron_map.input_dim != input_dim:
            # build() method in NeuronMap subclasses should set self._built = True
            self.neuron_map.build(input_dim)

        # Check if map build was successful (it should set input_dim)
        if self.neuron_map.input_dim is None:
             raise RuntimeError("NeuronMap failed to set input_dim during build.")

        # Get masks after map is built
        self.input_mask = tensor.convert_to_tensor(self.neuron_map.get_input_mask())
        self.recurrent_mask = tensor.convert_to_tensor(self.neuron_map.get_recurrent_mask())
        self.output_mask = tensor.convert_to_tensor(self.neuron_map.get_output_mask())

        # Initialize weights now that input_dim is known
        self._kernel = Parameter(
            self._initialize_tensor(
                (self.neuron_map.input_dim, self.neuron_map.units),
                self.kernel_initializer
            ),
            name="kernel"
        )

        self._recurrent_kernel = Parameter(
            self._initialize_tensor(
                (self.neuron_map.units, self.neuron_map.units),
                self.recurrent_initializer
            ),
            name="recurrent_kernel"
        )

        if self.use_bias:
            self._bias = Parameter(
                self._initialize_tensor(
                    (self.neuron_map.units,),
                    self.bias_initializer
                ),
                name="bias"
            )
        else:
            self._bias = None

        self.built = True
        # It's good practice to call super().build, although BaseModule's build is empty
        super().build(input_shape)


    def _initialize_tensor(self, shape, initializer):
        """Initialize a tensor with the specified shape and initializer."""
        if initializer == "glorot_uniform":
            # Glorot uniform initialization
            fan_in = shape[0] if len(shape) >= 1 else 1
            fan_out = shape[1] if len(shape) >= 2 else 1
            limit = ops.sqrt(ops.divide(6, (fan_in + fan_out)))
            return tensor.random_uniform(shape, -limit, limit, dtype=self.dtype)
        elif initializer == "orthogonal":
            # Orthogonal initialization
            if len(shape) < 2:
                raise ValueError("Orthogonal initialization requires at least 2 dimensions")
            # Generate a random matrix
            a = tensor.random_normal(shape, dtype=self.dtype)
            # Compute the QR factorization
            q, r = linearalg.qr(tensor.to_numpy(a))
            # Make Q uniform according to https://arxiv.org/pdf/1312.6120.pdf
            d = linearalg.diag(r)
            ph = ops.sign(d)
            q = ops.multiply(ph,q)
            return tensor.convert_to_tensor(q)
        elif initializer == "zeros":
            # Zeros initialization
            return tensor.zeros(shape, dtype=self.dtype)
        else:
            raise ValueError(f"Unknown initializer: {initializer}")
    
    def forward(
        self,
        inputs: Any,
        state: Optional[Any] = None,
        return_state: bool = False
    ) -> Union[Any, Tuple[Any, Any]]:
        # Ensure the layer is built before proceeding
        if not self.built:
             # Need the shape of the input tensor to build
             self.build(tensor.shape(inputs))

        """
        Forward pass of the NCP module.
        
        Args:
            inputs: Input tensor
            state: Optional state tensor
            return_state: Whether to return the state
            
        Returns:
            Output tensor, or tuple of (output, state) if return_state is True
        """
        if state is None:
            state = self.state
            
        # Apply input mask
        masked_inputs = ops.multiply(inputs, self.input_mask)
        
        # Apply recurrent mask
        masked_state = ops.matmul(state, self.recurrent_mask)
        
        # Compute new state
        new_state = ops.matmul(masked_inputs, self.kernel)
        if self.use_bias:
            new_state = ops.add(new_state, self.bias)
        new_state = ops.add(new_state, ops.matmul(masked_state, self.recurrent_kernel))
        new_state = self.activation(new_state)
        
        # Compute output - only include motor neurons
        # The output mask is a binary mask that selects only the motor neurons
        masked_output = ops.multiply(new_state, self.output_mask)
        
        # Extract only the motor neurons (first output_dim neurons)
        output = masked_output[:, :self.neuron_map.output_dim]
        
        # Update state
        self.state = new_state
        
        if return_state:
            return output, new_state
        else:
            return output
    
    def reset_state(self) -> None:
        """
        Reset the state of the NCP module.
        """
        self.state = tensor.zeros((1, self.neuron_map.units))
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the NCP module.
        
        Returns:
            Dictionary containing the configuration
        """
        config = {
            # Save map config and class name
            "neuron_map": self.neuron_map.get_config(),
            "neuron_map_class": self.neuron_map.__class__.__name__,
            "activation": self.activation_name,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "recurrent_initializer": self.recurrent_initializer,
            "bias_initializer": self.bias_initializer,
            "dtype": self.dtype,
            "state": self.state
        }
        return config
    
    @classmethod
    # Need to update the constructor signature first before fixing from_config call
    # Let's update the __init__ signature in a separate step first.
    # For now, just fixing the import logic part.

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NCP':
        """
        Create an NCP module from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            NCP module
        """
        # Config should contain the map configuration under 'neuron_map' key
        # and the map class name under 'neuron_map_class'
        map_config = config.pop("neuron_map") # Changed key from "wiring"
        map_class_name = config.pop("neuron_map_class") # Changed key from "wiring_class"

        # Directly import known map classes from their location in nn.modules.wiring
        from ember_ml.nn.modules.wiring import NeuronMap, NCPMap, FullyConnectedMap, RandomMap
        # AutoNCP is a layer, not a map, so it's not loaded here.

        # Map class name string to class object
        neuron_map_class_map = {
            "NeuronMap": NeuronMap,
            "NCPMap": NCPMap,
            "FullyConnectedMap": FullyConnectedMap,
            "RandomMap": RandomMap,
        }

        neuron_map_class_obj = neuron_map_class_map.get(map_class_name)
        if neuron_map_class_obj is None:
             raise ImportError(f"Unknown NeuronMap class '{map_class_name}' specified in config.")

        # Create the map instance using the remaining config params
        neuron_map = neuron_map_class_obj.from_config(map_config)

        # Create the NCP module
        config.pop('state', None) # Remove state before passing to constructor
        # Pass the created map object using the new argument name
        return cls(neuron_map=neuron_map, **config)