"""
Closed-form Continuous-time (CfC) Neural Network

This module provides an implementation of CfC layers,
which are a type of recurrent neural network that operates in continuous time.
This implementation directly uses NeuronMap for both structure and dynamics.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.wiring import NeuronMap, NCPMap
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn.initializers import glorot_uniform, orthogonal
from ember_ml.nn.modules.activations import get_activation

class CfC(Module):
    """
    Closed-form Continuous-time (CfC) RNN layer.
    
    This layer directly uses NeuronMap for both structure and dynamics,
    without relying on separate cell objects.
    """
    
    def __init__(
        self,
        neuron_map: NCPMap,
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: bool = False,
        **kwargs
    ):
        """
        Initialize the CfC layer.
        
        Args:
            neuron_map: NCPMap instance defining both structure and dynamics
            return_sequences: Whether to return the full sequence or just the last output
            return_state: Whether to return the final state
            go_backwards: Whether to process the sequence backwards
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        
        # Validate neuron_map type
        if not isinstance(neuron_map, NeuronMap):
            raise TypeError("neuron_map must be a NeuronMap instance")
        
        # Store the neuron map
        self.neuron_map = neuron_map
        
        # Store layer-specific parameters
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        
        # Initialize parameters
        self.kernel = None
        self.recurrent_kernel = None
        self.bias = None
        self.built = False
    
    def build(self, input_shape):
        """Build the CfC layer."""
        # Get input dimension
        input_dim = input_shape[-1]
        
        # Build the neuron map if not already built
        if not self.neuron_map.is_built():
            self.neuron_map.build(input_dim)
        
        # Get dimensions from neuron map
        units = self.neuron_map.units
        
        # Initialize parameters
        self.kernel = Parameter(tensor.zeros((input_dim, units * 4)))
        self.recurrent_kernel = Parameter(tensor.zeros((units, units * 4)))
        
        # Initialize bias if needed
        if self.neuron_map.use_bias:
            self.bias = Parameter(tensor.zeros((units * 4,)))
        
        # Initialize weights
        if self.neuron_map.kernel_initializer == "glorot_uniform":
            self.kernel.data = glorot_uniform((input_dim, units * 4))
        
        if self.neuron_map.recurrent_initializer == "orthogonal":
            self.recurrent_kernel.data = orthogonal((units, units * 4))
        
        # Mark as built
        self.built = True
    
    def forward(self, inputs, initial_state=None, time_deltas=None):
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input tensor (batch, time, features)
            initial_state: Initial state(s) for the cell
            time_deltas: Time deltas between inputs (optional)
            
        Returns:
            Layer output(s)
        """
        # Build if not already built
        if not self.built:
            self.build(tensor.shape(inputs))
        
        # Get input shape
        input_shape = tensor.shape(inputs)
        if len(input_shape) != 3:
             raise ValueError("Input tensor must be 3D (batch, time, features)")
        batch_size, time_steps, _ = input_shape
        
        # Create initial state if not provided
        if initial_state is None:
            h0 = tensor.zeros((batch_size, self.neuron_map.units))
            t0 = tensor.zeros((batch_size, self.neuron_map.units))
            initial_state = [h0, t0]
        
        # Process sequence
        outputs = []
        states = initial_state
        
        # Get parameters from neuron_map
        time_scale_factor = self.neuron_map.time_scale_factor
        activation_fn = get_activation(self.neuron_map.activation)
        rec_activation_fn = get_activation(self.neuron_map.recurrent_activation)
        
        # Process sequence in reverse if go_backwards is True
        time_indices = range(time_steps - 1, -1, -1) if self.go_backwards else range(time_steps)
        
        # Process each time step
        for t in time_indices:
            # Get current input
            x_t = inputs[:, t]
            
            # Get time delta for this step if provided
            ts = 1.0
            if time_deltas is not None:
                ts = time_deltas[:, t]
            
            # Project input
            z = ops.matmul(x_t, self.kernel)
            z = ops.add(z, ops.matmul(states[0], self.recurrent_kernel))
            if self.neuron_map.use_bias:
                z = ops.add(z, self.bias)
            
            # Split into gates
            z_chunks = tensor.split(z, 4, axis=-1)
            z_i, z_f, z_o, z_c = z_chunks
            
            # Apply activations
            i = rec_activation_fn(z_i)  # Input gate
            f = rec_activation_fn(z_f)  # Forget gate
            o = rec_activation_fn(z_o)  # Output gate
            c = activation_fn(z_c)      # Cell input
            
            # Apply time scaling
            decay = ops.exp(ops.divide(-ts, time_scale_factor))
            
            # Update state
            t_next = ops.add(ops.multiply(f, states[1]), ops.multiply(i, c))
            h_next = ops.multiply(o, activation_fn(ops.add(
                ops.multiply(decay, states[0]),
                ops.multiply(ops.subtract(tensor.ones_like(decay), decay), t_next)
            )))
            
            # Store output and update state
            outputs.append(h_next)
            states = [h_next, t_next]
        
        # If processing backwards, reverse the outputs sequence
        if self.go_backwards:
            outputs.reverse()
        
        # Stack outputs
        if self.return_sequences:
            outputs_tensor = tensor.stack(outputs, axis=1)
        else:
            outputs_tensor = outputs[-1]
        
        # Return outputs and states if requested
        if self.return_state:
            return outputs_tensor, states
        else:
            return outputs_tensor
    
    def reset_state(self, batch_size=1):
        """
        Reset the layer state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state
        """
        h0 = tensor.zeros((batch_size, self.neuron_map.units))
        t0 = tensor.zeros((batch_size, self.neuron_map.units))
        return [h0, t0]
    
    def get_config(self):
        """Returns the configuration of the CfC layer."""
        config = super().get_config()
        config.update({
            "neuron_map": self.neuron_map.get_config(),
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "go_backwards": self.go_backwards
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Creates a CfC layer from its configuration."""
        # Extract neuron_map config
        neuron_map_config = config.pop("neuron_map", {})
        
        # Create neuron_map
        from ember_ml.nn.modules.wiring import NCPMap
        neuron_map = NCPMap.from_config(neuron_map_config)
        
        # Create layer
        return cls(neuron_map=neuron_map, **config)
