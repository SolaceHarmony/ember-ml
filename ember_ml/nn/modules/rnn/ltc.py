"""
Liquid Time-Constant (LTC) Layer

This module provides an implementation of the LTC layer,
which wraps an LTCCell to create a recurrent layer.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

import numpy as np
from ember_ml import ops
from ember_ml.initializers.glorot import glorot_uniform, orthogonal
from ember_ml.nn.modules import Module
from ember_ml.nn.wirings import Wiring, FullyConnectedWiring as FullyConnected
from ember_ml.nn.modules.rnn.ltc_cell import LTCCell

class LTC(Module):
    """
    Liquid Time-Constant (LTC) RNN layer.
    
    This layer wraps an LTCCell to create a recurrent layer.
    """
    
    def __init__(
        self,
        input_size: int,
        units,
        return_sequences: bool = True,
        batch_first: bool = True,
        mixed_memory: bool = False,
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        epsilon=1e-8,
        implicit_param_constraints=True,
        **kwargs
    ):
        """
        Initialize the LTC layer.
        
        Args:
            input_size: Number of input features
            units: Wiring (Wiring instance) or integer representing the number of (fully-connected) hidden units
            return_sequences: Whether to return the full sequence or just the last output
            batch_first: Whether the batch or time dimension is the first (0-th) dimension
            mixed_memory: Whether to augment the RNN by a memory-cell to help learn long-term dependencies
            input_mapping: Type of input mapping ('affine', 'linear', or None)
            output_mapping: Type of output mapping ('affine', 'linear', or None)
            ode_unfolds: Number of ODE solver unfoldings
            epsilon: Small constant to avoid division by zero
            implicit_param_constraints: Whether to use implicit parameter constraints
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        
        self.input_size = input_size
        self.wiring_or_units = units
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.mixed_memory = mixed_memory
        
        # Create wiring if units is an integer
        if isinstance(units, Wiring):
            wiring = units
        else:
            wiring = FullyConnected(units)
        
        # Create LTC cell
        self.rnn_cell = LTCCell(
            wiring=wiring,
            in_features=input_size,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon,
            implicit_param_constraints=implicit_param_constraints
        )
        
        self._wiring = wiring
        
        # Create memory cell if using mixed memory
        if self.mixed_memory:
            self.memory_cell = self._create_memory_cell(input_size, self.state_size)
    
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
                self.input_bias = ops.zeros((state_size,))
                
                # Forget gate
                self.forget_kernel = glorot_uniform((input_size, state_size))
                self.forget_recurrent_kernel = orthogonal((state_size, state_size))
                self.forget_bias = ops.ones((state_size,))  # Initialize with 1s for better gradient flow
                
                # Cell gate
                self.cell_kernel = glorot_uniform((input_size, state_size))
                self.cell_recurrent_kernel = orthogonal((state_size, state_size))
                self.cell_bias = ops.zeros((state_size,))
                
                # Output gate
                self.output_kernel = glorot_uniform((input_size, state_size))
                self.output_recurrent_kernel = orthogonal((state_size, state_size))
                self.output_bias = ops.zeros((state_size,))
            
            def forward(self, inputs, states):
                h_prev, c_prev = states
                
                # Input gate
                i = ops.sigmoid(
                    ops.matmul(inputs, self.input_kernel) +
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
        matrix = np.asarray(self._wiring.sensory_adjacency_matrix)
        return float(np.sum(np.abs(matrix)))
    
    def forward(self, inputs, initial_state=None, timespans=None):
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_length, features) if batch_first=True,
                   or (seq_length, batch_size, features) if batch_first=False
            initial_state: Initial state of the RNN
            timespans: Time spans for continuous-time dynamics (default: 1.0)
            
        Returns:
            Layer output and final state
        """
        # Get device and batch information
        is_batched = len(ops.shape(inputs)) == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        
        # Handle non-batched inputs
        if not is_batched:
            inputs = ops.expand_dims(inputs, batch_dim)
            if timespans is not None:
                timespans = ops.expand_dims(timespans, batch_dim)
        
        # Get batch size and sequence length
        input_shape = ops.shape(inputs)
        batch_size = input_shape[batch_dim]
        seq_length = input_shape[seq_dim]
        
        # Initialize states if not provided
        if initial_state is None:
            h_state = ops.zeros((batch_size, self.state_size))
            c_state = ops.zeros((batch_size, self.state_size)) if self.mixed_memory else None
        else:
            if self.mixed_memory and not isinstance(initial_state, (list, tuple)):
                raise ValueError(
                    "When using mixed_memory=True, initial_state must be a tuple (h0, c0)"
                )
            h_state, c_state = initial_state if self.mixed_memory else (initial_state, None)
            
            # Handle non-batched states
            if is_batched and len(ops.shape(h_state)) != 2:
                raise ValueError(
                    f"For batched inputs, initial_state should be 2D but got {len(ops.shape(h_state))}D"
                )
            elif not is_batched and len(ops.shape(h_state)) != 1:
                raise ValueError(
                    f"For non-batched inputs, initial_state should be 1D but got {len(ops.shape(h_state))}D"
                )
                
                # Add batch dimension for non-batched states
                h_state = ops.expand_dims(h_state, 0)
                c_state = ops.expand_dims(c_state, 0) if c_state is not None else None
        
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
            outputs = ops.stack(output_sequence, axis=stack_dim)
        else:
            # If not returning sequences, use the last output
            outputs = output_sequence[-1] if output_sequence else None
        
        # Prepare final state
        final_state = (h_state, c_state) if self.mixed_memory else h_state
        
        # Handle non-batched outputs
        if not is_batched:
            outputs = ops.squeeze(outputs, batch_dim)
            if self.mixed_memory:
                final_state = (ops.squeeze(h_state, 0), ops.squeeze(c_state, 0))
            else:
                final_state = ops.squeeze(h_state, 0)
        
        return outputs, final_state
    
    def reset_state(self, batch_size=1):
        """
        Reset the layer state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state
        """
        h_state = ops.zeros((batch_size, self.state_size))
        if self.mixed_memory:
            c_state = ops.zeros((batch_size, self.state_size))
            return (h_state, c_state)
        else:
            return h_state