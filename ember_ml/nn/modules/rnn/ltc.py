"""
Liquid Time-Constant (LTC) Layer

This module provides an implementation of the LTC layer,
which wraps an LTCCell to create a recurrent layer.
"""

from typing import Dict, Any

import numpy as np
from ember_ml import ops
# Updated initializer import path
from ember_ml.nn.initializers import glorot_uniform, orthogonal
from ember_ml.nn.modules import Module
# Updated wiring import paths
from ember_ml.nn.modules.wiring import NeuronMap # Use renamed base class
from ember_ml.nn.modules.rnn.ltc_cell import LTCCell
from ember_ml.nn import tensor
from ember_ml.nn.modules.activations import tanh

class LTC(Module):
    """
    Liquid Time-Constant (LTC) RNN layer.
    
    This layer wraps an LTCCell to create a recurrent layer.
    """
    
    def __init__(
        self,
        neuron_map: NeuronMap,
        return_sequences: bool = True,
        return_state: bool = False, # Add return_state parameter
        batch_first: bool = True,
        mixed_memory: bool = False,
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        epsilon=1e-8,
        implicit_param_constraints=True
        # **kwargs removed
    ):
        """
        Initialize the LTC layer.
        
        Args:
            neuron_map: NeuronMap instance defining the connectivity structure
                        (input dimensions will be derived from the first input tensor)
            return_sequences: Whether to return the full sequence or just the last output
            batch_first: Whether the batch or time dimension is the first (0-th) dimension
            mixed_memory: Whether to augment the RNN by a memory-cell to help learn long-term dependencies
            input_mapping: Type of input mapping ('affine', 'linear', or None)
            output_mapping: Type of output mapping ('affine', 'linear', or None)
            ode_unfolds: Number of ODE solver unfoldings
            epsilon: Small constant to avoid division by zero
            implicit_param_constraints: Whether to use implicit parameter constraints
            # **kwargs removed from docstring
        """
        super().__init__() # Call base init without args
        
        # Store the map for reference/config
        self.neuron_map = neuron_map
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state  # Store return_state as an instance attribute
        self.mixed_memory = mixed_memory
        
        # Validate that neuron_map is actually a NeuronMap instance
        if not isinstance(neuron_map, NeuronMap):
            raise TypeError("neuron_map must be a NeuronMap instance")
            
        # Set input_size from neuron_map.input_dim if the map is already built
        # Otherwise, it will be set during the first forward pass
        self.input_size = getattr(neuron_map, 'input_dim', None)
        
        # Create LTC cell
        self.rnn_cell = LTCCell(
            neuron_map=neuron_map,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon,
            implicit_param_constraints=implicit_param_constraints
        )
        
        # Store the map internally for property access
        self._neuron_map = neuron_map
        
        # Create memory cell if using mixed memory
        # If input_size is not available yet, memory cell creation will be deferred
        self.memory_cell = None
        if self.mixed_memory and self.input_size is not None:
            self.memory_cell = self._create_memory_cell(self.input_size, self.state_size)
    
    def _create_memory_cell(self, input_size, state_size):
        """Create a memory cell for mixed memory mode."""
        # Simple memory cell implementation
        class MemoryCell(Module):
            def __init__(self, input_size, state_size):
                super().__init__()
                self.input_size = input_size
                self.state_size = state_size
                
                # Input gate
                self.input_kernel = glorot_uniform((input_size, state_size))
                self.input_recurrent_kernel = orthogonal((state_size, state_size))
                self.input_bias = tensor.zeros((state_size,))
                
                # Forget gate
                self.forget_kernel = glorot_uniform((input_size, state_size))
                self.forget_recurrent_kernel = orthogonal((state_size, state_size))
                self.forget_bias = tensor.ones((state_size,))  # Initialize with 1s for better gradient flow
                
                # Cell gate
                self.cell_kernel = glorot_uniform((input_size, state_size))
                self.cell_recurrent_kernel = orthogonal((state_size, state_size))
                self.cell_bias = tensor.zeros((state_size,))
                
                # Output gate
                self.output_kernel = glorot_uniform((input_size, state_size))
                self.output_recurrent_kernel = orthogonal((state_size, state_size))
                self.output_bias = tensor.zeros((state_size,))
            
            def forward(self, inputs, states):
                h_prev, c_prev = states
                
                # Input gate
                i = ops.sigmoid( # type: ignore
                    ops.matmul(inputs, self.input_kernel) + # type: ignore
                    ops.matmul(h_prev, self.input_recurrent_kernel) +
                    self.input_bias
                )
                
                # Forget gate
                f = ops.sigmoid(
                    ops.matmul(inputs, self.forget_kernel) +
                    ops.matmul(h_prev, self.forget_recurrent_kernel) +
                    self.forget_bias
                )
                
                # Cell gate
                g = ops.tanh(
                    ops.matmul(inputs, self.cell_kernel) +
                    ops.matmul(h_prev, self.cell_recurrent_kernel) +
                    self.cell_bias
                )
                
                # Output gate
                o = ops.sigmoid(
                    ops.matmul(inputs, self.output_kernel) +
                    ops.matmul(h_prev, self.output_recurrent_kernel) +
                    self.output_bias
                )
                
                # Update cell state
                c = f * c_prev + i * g
                
                # Update hidden state
                h = o * ops.tanh(c)
                
                return h, (h, c)
        
        return MemoryCell(input_size, state_size)
    
    @property
    def state_size(self):
        return self._neuron_map.units
    
    @property
    def sensory_size(self):
        return self._neuron_map.input_dim
    
    @property
    def motor_size(self):
        return self._neuron_map.output_dim
    
    @property
    def output_size(self):
        return self.motor_size
    
    @property
    def synapse_count(self):
        # Use ops/tensor for calculations, avoid numpy
        # Ensure adjacency_matrix is a tensor first
        adj_matrix_tensor = tensor.convert_to_tensor(self._neuron_map.adjacency_matrix)
        return tensor.sum(tensor.abs(adj_matrix_tensor))
    
    @property
    def sensory_synapse_count(self):
        # Use ops/tensor for calculations, avoid numpy
        sensory_matrix_tensor = tensor.convert_to_tensor(self._neuron_map.sensory_adjacency_matrix)
        # sum result might be a 0-dim tensor, convert to float if necessary
        sum_val = tensor.sum(tensor.abs(sensory_matrix_tensor))
        # Use item() to get Python scalar
        return float(tensor.item(sum_val))
    
    def forward(self, inputs, initial_state=None, timespans=None):
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_length, features) if batch_first=True,
                    or (seq_length, batch_size, features) if batch_first=False
            initial_state: Initial state of the RNN
            timespans: Time spans for continuous-time dynamics (default: 1.0)
            
        Returns:
            Layer output and final state if return_state is True, otherwise just the layer output
        """
        # Get device and batch information
        is_batched = len(tensor.shape(inputs)) == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        
        # Build or update neuron_map if needed based on input dimensions
        feature_dim = 2  # Assuming (batch, seq, features) or (seq, batch, features)
        input_features = tensor.shape(inputs)[feature_dim]
        
        # If input_size is not set or the map isn't built, build it now
        if self.input_size is None or not self.neuron_map.is_built():
            self.neuron_map.build(input_features)
            self.input_size = self.neuron_map.input_dim
            
            # Create memory cell now if using mixed memory and it wasn't created during init
            if self.mixed_memory and self.memory_cell is None:
                self.memory_cell = self._create_memory_cell(self.input_size, self.state_size)
        
        # Handle non-batched inputs
        if not is_batched:
            inputs = tensor.expand_dims(inputs, batch_dim)
            if timespans is not None:
                timespans = tensor.expand_dims(timespans, batch_dim)
        
        # Get batch size and sequence length
        input_shape = tensor.shape(inputs)
        batch_size = input_shape[batch_dim]
        seq_length = input_shape[seq_dim]
        
        # Initialize states if not provided
        if initial_state is None:
            h_state = tensor.zeros((batch_size, self.state_size))
            c_state = tensor.zeros((batch_size, self.state_size)) if self.mixed_memory else None
        else:
            if self.mixed_memory and not isinstance(initial_state, (list, tuple)):
                raise ValueError(
                    "When using mixed_memory=True, initial_state must be a tuple (h0, c0)"
                )
            h_state, c_state = initial_state if self.mixed_memory else (initial_state, None)
            
            # Handle non-batched states
            if is_batched and len(tensor.shape(h_state)) != 2:
                raise ValueError(
                    f"For batched inputs, initial_state should be 2D but got {len(tensor.shape(h_state))}D"
                )
            elif not is_batched and len(tensor.shape(h_state)) != 1:
                # Add batch dimension for non-batched states
                h_state = tensor.expand_dims(h_state, 0)
                c_state = tensor.expand_dims(c_state, 0) if c_state is not None else None
        
        # Process sequence
        output_sequence = []
        for t in range(seq_length):
            # Get input for current time step
            if self.batch_first:
                current_input = inputs[:, t]
                ts = 1.0 if timespans is None else timespans[:, t]
            else:
                current_input = inputs[t]
                ts = 1.0 if timespans is None else timespans[t]
            
            # Apply memory cell if using mixed memory
            if self.mixed_memory:
                h_state, (h_state, c_state) = self.memory_cell(current_input, (h_state, c_state))
            
            # Apply LTC cell
            output, h_state = self.rnn_cell(current_input, h_state, ts)
            
            # Store output if returning sequences
            if self.return_sequences:
                output_sequence.append(output)
        
        # Prepare output
        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            outputs = tensor.stack(output_sequence, axis=stack_dim)
        else:
            # If not returning sequences, use the last output
            outputs = output_sequence[-1] if output_sequence else None
        
        # Prepare final state
        final_state = (h_state, c_state) if self.mixed_memory else h_state
        
        # Handle non-batched outputs
        if not is_batched:
            outputs = tensor.squeeze(outputs, batch_dim)
            if self.mixed_memory:
                final_state = (tensor.squeeze(h_state, 0), tensor.squeeze(c_state, 0))
            else:
                final_state = tensor.squeeze(h_state, 0)
        
        if self.return_state:
            return outputs, final_state
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
        h_state = tensor.zeros((batch_size, self.state_size))
        if self.mixed_memory:
            c_state = tensor.zeros((batch_size, self.state_size))
            return (h_state, c_state)
        else:
            return h_state
    
    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the LTC layer."""
        config = super().get_config()
        
        # Save the cell's config
        cell_config = self.rnn_cell.get_config()
        
        # Add return_state to the config
        config.update({
            "return_state": self.return_state,
        })
        
        # Save layer's direct __init__ args
        config.update({
            # Don't save input_size as it's derived from the neuron_map
            # Save the neuron_map config
            "neuron_map": self.neuron_map.get_config(),
            "neuron_map_class": self.neuron_map.__class__.__name__,
            # Save other layer args
            "return_sequences": self.return_sequences,
            "batch_first": self.batch_first,
            "mixed_memory": self.mixed_memory,
            # Save cell args directly in the layer config
            "input_mapping": self.rnn_cell._input_mapping,
            "output_mapping": self.rnn_cell._output_mapping,
            "ode_unfolds": self.rnn_cell._ode_unfolds,
            "epsilon": self.rnn_cell._epsilon,
            "implicit_param_constraints": self.rnn_cell._implicit_param_constraints,
            # Also save cell config and class name
            "cell_config": cell_config,
            "cell_class_name": self.rnn_cell.__class__.__name__
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LTC':
        """Creates an LTC layer from its configuration."""
        # First handle the cell config
        cell_config = config.pop("cell_config", None)
        cell_class_name = config.pop("cell_class_name", "LTCCell")
        
        # Handle backward compatibility with old configs that used input_size or neuron_map_or_units
        # This ensures older models can still be loaded
        from ember_ml.nn.modules.wiring import FullyConnectedMap
        
        # Handle return_state parameter
        return_state = config.pop("return_state", False)
        
        # Handle old input_size parameter (needed for backward compatibility)
        old_input_size = config.pop("input_size", None)
        
        # Handle old neuron_map_or_units parameter
        if "neuron_map_or_units" in config and "neuron_map" not in config:
            map_or_units = config.pop("neuron_map_or_units")
            if isinstance(map_or_units, dict):
                # It was a map config
                config["neuron_map"] = map_or_units
                if "class_name" not in config["neuron_map"]:
                    config["neuron_map_class"] = "FullyConnectedMap"
            elif isinstance(map_or_units, int):
                # It was an integer (units)
                # Create a FullyConnectedMap config
                config["neuron_map"] = {"units": map_or_units}
                config["neuron_map_class"] = "FullyConnectedMap"
                
        # If we have old_input_size, make sure it's included in the map config
        # This ensures the map will be built correctly
        if old_input_size is not None and "neuron_map" in config and isinstance(config["neuron_map"], dict):
            config["neuron_map"]["input_dim"] = old_input_size
        
        # Reconstruct the NeuronMap
        if "neuron_map" in config and isinstance(config["neuron_map"], dict):
            map_config = config.pop("neuron_map")
            map_class_name = config.pop("neuron_map_class", "NeuronMap")
            
            from ember_ml.nn.modules.wiring import NeuronMap, NCPMap, FullyConnectedMap, RandomMap
            neuron_map_class_map = {
                "NeuronMap": NeuronMap,
                "NCPMap": NCPMap,
                "FullyConnectedMap": FullyConnectedMap,
                "RandomMap": RandomMap,
            }
            map_class_obj = neuron_map_class_map.get(map_class_name)
            if map_class_obj is None:
                raise ImportError(f"Unknown NeuronMap class '{map_class_name}' specified in config.")
            
            # Reconstruct map and put object back into config
            config['neuron_map'] = map_class_obj.from_config(map_config)
        
        # Let the BaseModule.from_config handle calling cls(**config)
        return super(LTC, cls).from_config(config)