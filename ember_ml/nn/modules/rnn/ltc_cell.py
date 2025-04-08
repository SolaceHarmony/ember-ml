"""
Liquid Time-Constant (LTC) Cell

This module provides an implementation of the LTC cell,
which is a type of recurrent neural network cell that operates in continuous time
with biologically-inspired dynamics.
"""

from typing import Dict, Any # Added imports
from ember_ml import ops
from ember_ml.nn.modules import Parameter # Module removed, inheriting from ModuleWiredCell
from ember_ml.nn.modules.module_wired_cell import ModuleWiredCell # Import parent class
from ember_ml.nn import tensor
from ember_ml.nn.modules.wiring import NeuronMap # Import the renamed base class
from ember_ml.nn.modules import activations # Import activations module
class LTCCell(ModuleWiredCell): # Inherit from ModuleWiredCell
    """
    Liquid Time-Constant (LTC) cell.
    
    This cell implements a continuous-time recurrent neural network
    with biologically-inspired dynamics.
    """
    
    def __init__(
        self,
        neuron_map: NeuronMap, # Use new name and type hint
        # Removed in_features, input_size comes from map via ModuleWiredCell
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        epsilon=1e-8,
        implicit_param_constraints=False,
        **kwargs
    ):
        """
        Initialize the LTC cell.
        
        Args:
            neuron_map: NeuronMap configuration object
            # in_features removed
            input_mapping: Type of input mapping ('affine', 'linear', or None)
            output_mapping: Type of output mapping ('affine', 'linear', or None)
            ode_unfolds: Number of ODE solver unfoldings
            epsilon: Small constant to avoid division by zero
            implicit_param_constraints: Whether to use implicit parameter constraints
            **kwargs: Additional keyword arguments
        """
        # Call ModuleWiredCell's __init__, passing only the map and kwargs.
        # input_size will be determined during the build phase.
        super().__init__(
             neuron_map=neuron_map,
             **kwargs
        )
        # self.wiring is set by parent init
        # self.input_size, self.hidden_size (units) are set by parent init

        # Store LTC specific parameters
        self.make_positive_fn = activations.softplus if implicit_param_constraints else lambda x: x
        self._implicit_param_constraints = implicit_param_constraints
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        # self._wiring attribute is no longer needed directly, use self.neuron_map from parent
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self._clip = activations.relu # Define clipping function here

        # Initialize LTC specific parameters
        # Parameter allocation moved to build method
    
    # Redundant properties removed as they are inherited from ModuleWiredCell
    
    def _get_init_value(self, shape, param_name):
        """Get initial values for parameters based on predefined ranges."""
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return tensor.ones(shape) * minval
        else:
            return tensor.random_uniform(shape, minval=minval, maxval=maxval) # Correct argument order
    
    def _allocate_parameters(self):
        """Allocate all parameters for the LTC cell."""
        # Neuron parameters
        self.gleak = Parameter(self._get_init_value((self.state_size,), "gleak"))
        self.vleak = Parameter(self._get_init_value((self.state_size,), "vleak"))
        self.cm = Parameter(self._get_init_value((self.state_size,), "cm"))
        
        # Synapse parameters
        self.sigma = Parameter(self._get_init_value((self.state_size, self.state_size), "sigma"))
        self.mu = Parameter(self._get_init_value((self.state_size, self.state_size), "mu"))
        self.w = Parameter(self._get_init_value((self.state_size, self.state_size), "w"))
        # Access map via self.neuron_map
        self.erev = Parameter(tensor.convert_to_tensor(self.neuron_map.erev_initializer(), dtype=tensor.float32))
        
        # Sensory synapse parameters
        self.sensory_sigma = Parameter(self._get_init_value((self.input_size, self.state_size), "sensory_sigma"))
        self.sensory_mu = Parameter(self._get_init_value((self.input_size, self.state_size), "sensory_mu"))
        self.sensory_w = Parameter(self._get_init_value((self.input_size, self.state_size), "sensory_w"))
        # Handle case where sensory initializer might return None
        sensory_erev_init = self.neuron_map.sensory_erev_initializer()
        if sensory_erev_init is not None:
            self.sensory_erev = Parameter(tensor.convert_to_tensor(sensory_erev_init, dtype=tensor.float32))
        else:
            # Initialize with zeros if no sensory matrix exists
            self.sensory_erev = Parameter(tensor.zeros((self.input_size, self.state_size), dtype=tensor.float32))
        
        # Sparsity masks
        self.sparsity_mask = Parameter(
            tensor.convert_to_tensor(ops.abs(self.neuron_map.adjacency_matrix), dtype=tensor.float32), # Use self.neuron_map
            requires_grad=False
        )
        # Handle case where sensory matrix might be None
        sensory_adj_matrix = getattr(self.neuron_map, 'sensory_adjacency_matrix', None)
        if sensory_adj_matrix is not None:
            self.sensory_sparsity_mask = Parameter(
                tensor.convert_to_tensor(ops.abs(sensory_adj_matrix), dtype=tensor.float32),
                requires_grad=False
            )
        else:
            # Initialize mask with zeros if no sensory matrix
            self.sensory_sparsity_mask = Parameter(
                tensor.zeros((self.input_size, self.state_size), dtype=tensor.float32),
                requires_grad=False
            )
        # Removed stray closing parenthesis
        
        # Input and output mapping parameters
        if self._input_mapping in ["affine", "linear"]:
            self.input_w = Parameter(tensor.ones((self.input_size,)))
        if self._input_mapping == "affine":
            self.input_b = Parameter(tensor.zeros((self.input_size,)))
        
        if self._output_mapping in ["affine", "linear"]:
            self.output_w = Parameter(tensor.ones((self.output_size,)))
        if self._output_mapping == "affine":
            self.output_b = Parameter(tensor.zeros((self.output_size,)))

    def build(self, input_shape):
        """Builds the LTC Cell's parameters."""
        # Call the parent build method first. This is crucial!
        # ModuleWiredCell.build will determine input_dim from input_shape,
        # build the neuron_map, and set self.input_size, self.hidden_size, self.output_size.
        super().build(input_shape)

        # Now that dimensions (self.input_size, self.state_size/hidden_size, self.output_size)
        # are set by the parent build, allocate LTC-specific parameters.
        self._allocate_parameters()

        # self.built flag is managed by BaseModule.__call__
    
    def _sigmoid(self, v_pre, mu, sigma):
        """Compute sigmoid activation for synapses."""
        v_pre = tensor.expand_dims(v_pre, -1)  # For broadcasting
        # Pass Parameter objects directly to ops functions
        mues = ops.subtract(v_pre, mu)
        x = ops.multiply(sigma, mues)
        return activations.sigmoid(x)
    
    def _ode_solver(self, inputs, state, elapsed_time):
        """Solve the ODE for the LTC dynamics."""
        v_pre = state
        
        # Pre-compute the effects of the sensory neurons
        # Apply make_positive_fn to Parameter, then use ops.multiply
        sensory_w_positive = self.make_positive_fn(self.sensory_w)
        sensory_sigmoid_out = self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        sensory_w_activation = ops.multiply(sensory_w_positive, sensory_sigmoid_out)
        sensory_w_activation = ops.multiply(sensory_w_activation, self.sensory_sparsity_mask) # Pass Parameter
    
        sensory_rev_activation = ops.multiply(sensory_w_activation, self.sensory_erev) # Pass Parameter
        
        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = ops.stats.sum(sensory_rev_activation, axis=1)
        w_denominator_sensory = ops.stats.sum(sensory_w_activation, axis=1)
        
        # cm/t is loop invariant
        # Apply make_positive_fn to Parameter, use ops.divide
        cm_positive = self.make_positive_fn(self.cm)
        time_term = ops.divide(elapsed_time, self._ode_unfolds)
        cm_t = ops.divide(cm_positive, time_term)
        
        # Unfold the ODE multiple times into one RNN step
        # Apply make_positive_fn to Parameter
        w_param = self.make_positive_fn(self.w)
        for t in range(self._ode_unfolds):
            # Use ops.multiply
            sigmoid_out = self._sigmoid(v_pre, self.mu, self.sigma) # _sigmoid takes Parameters
            w_activation = ops.multiply(w_param, sigmoid_out)
            w_activation = ops.multiply(w_activation, self.sparsity_mask) # Pass Parameter
            
            rev_activation = ops.multiply(w_activation, self.erev) # Pass Parameter
            
            # Reduce over dimension 1 (=source neurons)
            w_numerator = ops.add(ops.stats.sum(rev_activation, axis=1), w_numerator_sensory) # Use ops.add
            # Use ops.add for consistency, although '+' might work between native tensors
            w_denominator = ops.add(ops.stats.sum(w_activation, axis=1), w_denominator_sensory)

            # Apply make_positive_fn to Parameter
            gleak = self.make_positive_fn(self.gleak)
            # Use ops functions for numerator calculation
            term1 = ops.multiply(cm_t, v_pre)
            term2 = ops.multiply(gleak, self.vleak) # Pass Parameter
            numerator = ops.add(ops.add(term1, term2), w_numerator)
            # Use ops.add for denominator calculation
            denominator = ops.add(ops.add(cm_t, gleak), w_denominator)
            
            # Avoid dividing by 0
            # Use ops functions for denominator and division
            # Use ops.add for denominator calculation
            # Ensure ops.add is used for denominator calculation
            denominator = ops.add(ops.add(cm_t, gleak), w_denominator)
            v_pre = ops.divide(numerator, ops.add(denominator, self._epsilon))
        
        return v_pre
    
    def _map_inputs(self, inputs):
        """Apply input mapping to the inputs."""
        if self._input_mapping in ["affine", "linear"]:
            inputs = ops.multiply(inputs, self.input_w) # Pass Parameter
        if self._input_mapping == "affine":
            inputs = ops.add(inputs, self.input_b) # Pass Parameter
        return inputs
    
    def _map_outputs(self, state):
        """Apply output mapping to the state."""
        output = state
        if self.output_size < self.state_size:
            output = output[:, 0:self.output_size]  # slice
        
        if self._output_mapping in ["affine", "linear"]:
            output = ops.multiply(output, self.output_w) # Pass Parameter
        if self._output_mapping == "affine":
            output = ops.add(output, self.output_b) # Pass Parameter
        return output
    
    def apply_weight_constraints(self):
        """Apply constraints to the weights if not using implicit constraints."""
        if not self._implicit_param_constraints:
            # In implicit mode, the parameter constraints are implemented via
            # a softplus function at runtime
            self.w.data = self._clip(self.w.data)
            self.sensory_w.data = self._clip(self.sensory_w.data)
            self.cm.data = self._clip(self.cm.data)
            self.gleak.data = self._clip(self.gleak.data)
    
    def forward(self, inputs, states, elapsed_time=1.0):
        """
        Forward pass through the cell.
        
        Args:
            inputs: Input tensor
            states: Previous state
            elapsed_time: Time elapsed since last update (default: 1.0)
            
        Returns:
            Tuple of (output, new_state)
        """
        # Map inputs
        inputs = self._map_inputs(inputs)
        
        # Solve ODE
        next_state = self._ode_solver(inputs, states, elapsed_time)
        
        # Map outputs
        outputs = self._map_outputs(next_state)
        
        return outputs, next_state
    
    def reset_state(self, batch_size=1):
        """
        Reset the cell state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state
        """
        return tensor.zeros((batch_size, self.state_size))

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the LTC cell."""
        # Get config from ModuleWiredCell (map config, map class, mode)
        config = super().get_config()
        # Add LTC specific args from __init__
        config.update({
            # 'neuron_map' and 'neuron_map_class' are already saved by super()
            # 'input_size' ('in_features') is also handled by super() via ModuleWiredCell logic
            "input_mapping": self._input_mapping,
            "output_mapping": self._output_mapping,
            "ode_unfolds": self._ode_unfolds,
            "epsilon": self._epsilon,
            "implicit_param_constraints": self._implicit_param_constraints,
        })
        # Remove args handled by ModuleWiredCell/ModuleCell if not needed directly by __init__
        config.pop('mode', None)
        config.pop('activation', None) # LTCCell doesn't use activation param
        config.pop('use_bias', None) # LTCCell doesn't use bias param directly

        # Ensure we don't have duplicate parameters
        # Use input_size as in_features but don't include both
        config['in_features'] = self.input_size
        config.pop('input_size', None)

        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LTCCell':
        """Creates an LTC cell from its configuration."""
        # Extract and handle parameters to avoid conflicts
        in_features = config.pop('in_features', None)
        input_size = config.pop('input_size', None)
        
        # Make sure we only have one value for input size
        if in_features is not None:
            # Use in_features as the source of truth
            config['in_features'] = in_features
        elif input_size is not None:
            # If in_features is missing but input_size exists, use that
            config['in_features'] = input_size
            
        # Now create the instance with the cleaned config
        return cls(**config)