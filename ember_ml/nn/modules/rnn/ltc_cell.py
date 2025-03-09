"""
Liquid Time-Constant (LTC) Cell

This module provides an implementation of the LTC cell,
which is a type of recurrent neural network cell that operates in continuous time
with biologically-inspired dynamics.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

import numpy as np
from ember_ml import ops
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn.wirings import Wiring

class LTCCell(Module):
    """
    Liquid Time-Constant (LTC) cell.
    
    This cell implements a continuous-time recurrent neural network
    with biologically-inspired dynamics.
    """
    
    def __init__(
        self,
        wiring,
        in_features=None,
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
            wiring: Wiring configuration (e.g., AutoNCP)
            in_features: Number of input features
            input_mapping: Type of input mapping ('affine', 'linear', or None)
            output_mapping: Type of output mapping ('affine', 'linear', or None)
            ode_unfolds: Number of ODE solver unfoldings
            epsilon: Small constant to avoid division by zero
            implicit_param_constraints: Whether to use implicit parameter constraints
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        
        if in_features is not None:
            wiring.build(in_features)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'in_features' or call the 'wiring.build()'."
            )
        
        self.make_positive_fn = ops.softplus if implicit_param_constraints else lambda x: x
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
        self._wiring = wiring
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self._clip = ops.relu
        
        # Initialize parameters
        self._allocate_parameters()
    
    @property
    def state_size(self):
        return self._wiring.units
    
    @property
    def sensory_size(self):
        return self._wiring.input_dim
    
    @property
    def motor_size(self):
        return self._wiring.output_dim
    
    @property
    def output_size(self):
        return self.motor_size
    
    @property
    def synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))
    
    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self._wiring.sensory_adjacency_matrix))
    
    def _get_init_value(self, shape, param_name):
        """Get initial values for parameters based on predefined ranges."""
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return ops.ones(shape) * minval
        else:
            return ops.random_uniform(minval, maxval, shape)
    
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
        self.erev = Parameter(ops.convert_to_tensor(self._wiring.erev_initializer(), dtype=ops.float32))
        
        # Sensory synapse parameters
        self.sensory_sigma = Parameter(self._get_init_value((self.sensory_size, self.state_size), "sensory_sigma"))
        self.sensory_mu = Parameter(self._get_init_value((self.sensory_size, self.state_size), "sensory_mu"))
        self.sensory_w = Parameter(self._get_init_value((self.sensory_size, self.state_size), "sensory_w"))
        self.sensory_erev = Parameter(ops.convert_to_tensor(self._wiring.sensory_erev_initializer(), dtype=ops.float32))
        
        # Sparsity masks
        self.sparsity_mask = Parameter(
            ops.convert_to_tensor(np.abs(self._wiring.adjacency_matrix), dtype=ops.float32),
            requires_grad=False
        )
        self.sensory_sparsity_mask = Parameter(
            ops.convert_to_tensor(np.abs(self._wiring.sensory_adjacency_matrix), dtype=ops.float32),
            requires_grad=False
        )
        
        # Input and output mapping parameters
        if self._input_mapping in ["affine", "linear"]:
            self.input_w = Parameter(ops.ones((self.sensory_size,)))
        if self._input_mapping == "affine":
            self.input_b = Parameter(ops.zeros((self.sensory_size,)))
        
        if self._output_mapping in ["affine", "linear"]:
            self.output_w = Parameter(ops.ones((self.motor_size,)))
        if self._output_mapping == "affine":
            self.output_b = Parameter(ops.zeros((self.motor_size,)))
    
    def _sigmoid(self, v_pre, mu, sigma):
        """Compute sigmoid activation for synapses."""
        v_pre = ops.expand_dims(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return ops.sigmoid(x)
    
    def _ode_solver(self, inputs, state, elapsed_time):
        """Solve the ODE for the LTC dynamics."""
        v_pre = state
        
        # Pre-compute the effects of the sensory neurons
        sensory_w_activation = self.make_positive_fn(self.sensory_w) * self._sigmoid(
            inputs, self.sensory_mu, self.sensory_sigma
        )
        sensory_w_activation = sensory_w_activation * self.sensory_sparsity_mask
        
        sensory_rev_activation = sensory_w_activation * self.sensory_erev
        
        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = ops.sum(sensory_rev_activation, axis=1)
        w_denominator_sensory = ops.sum(sensory_w_activation, axis=1)
        
        # cm/t is loop invariant
        cm_t = self.make_positive_fn(self.cm) / (elapsed_time / self._ode_unfolds)
        
        # Unfold the ODE multiple times into one RNN step
        w_param = self.make_positive_fn(self.w)
        for t in range(self._ode_unfolds):
            w_activation = w_param * self._sigmoid(v_pre, self.mu, self.sigma)
            w_activation = w_activation * self.sparsity_mask
            
            rev_activation = w_activation * self.erev
            
            # Reduce over dimension 1 (=source neurons)
            w_numerator = ops.sum(rev_activation, axis=1) + w_numerator_sensory
            w_denominator = ops.sum(w_activation, axis=1) + w_denominator_sensory
            
            gleak = self.make_positive_fn(self.gleak)
            numerator = cm_t * v_pre + gleak * self.vleak + w_numerator
            denominator = cm_t + gleak + w_denominator
            
            # Avoid dividing by 0
            v_pre = numerator / (denominator + self._epsilon)
        
        return v_pre
    
    def _map_inputs(self, inputs):
        """Apply input mapping to the inputs."""
        if self._input_mapping in ["affine", "linear"]:
            inputs = inputs * self.input_w
        if self._input_mapping == "affine":
            inputs = inputs + self.input_b
        return inputs
    
    def _map_outputs(self, state):
        """Apply output mapping to the state."""
        output = state
        if self.motor_size < self.state_size:
            output = output[:, 0:self.motor_size]  # slice
        
        if self._output_mapping in ["affine", "linear"]:
            output = output * self.output_w
        if self._output_mapping == "affine":
            output = output + self.output_b
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
        return ops.zeros((batch_size, self.state_size))