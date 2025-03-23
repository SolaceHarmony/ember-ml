"""
Gated Recurrent Unit (GRU) Layer

This module provides an implementation of the GRU layer,
which wraps a GRUCell to create a recurrent layer.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

from ember_ml import ops
from ember_ml.nn.modules import Module
from ember_ml.nn.modules.rnn.gru_cell import GRUCell
from ember_ml.nn import tensor

class GRU(Module):
    """
    Gated Recurrent Unit (GRU) RNN layer.
    
    This layer wraps a GRUCell to create a recurrent layer.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        return_sequences: bool = True,
        return_state: bool = False,
        **kwargs
    ):
        """
        Initialize the GRU layer.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of GRU layers
            bias: Whether to use bias
            batch_first: Whether the batch or time dimension is the first (0-th) dimension
            dropout: Dropout probability (applied between layers)
            bidirectional: Whether to use a bidirectional GRU
            return_sequences: Whether to return the full sequence or just the last output
            return_state: Whether to return the final state
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences
        self.return_state = return_state
        
        # Create GRU cells for each layer and direction
        self.forward_cells = []
        self.backward_cells = []
        
        # Input size for the first layer is the input size
        # For subsequent layers, it's the hidden size (or 2x hidden size if bidirectional)
        layer_input_size = input_size
        
        for layer in range(num_layers):
            # Forward direction cell
            self.forward_cells.append(
                GRUCell(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    bias=bias
                )
            )
            
            # Backward direction cell (if bidirectional)
            if bidirectional:
                self.backward_cells.append(
                    GRUCell(
                        input_size=layer_input_size,
                        hidden_size=hidden_size,
                        bias=bias
                    )
                )
            
            # Update input size for the next layer
            layer_input_size = hidden_size * (2 if bidirectional else 1)
    
    def forward(self, inputs, initial_state=None):
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_length, features) if batch_first=True,
                   or (seq_length, batch_size, features) if batch_first=False
            initial_state: Initial state of the RNN
            
        Returns:
            Layer output and final state
        """
        # Get batch information
        is_batched = len(ops.shape(inputs)) == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        
        # Handle non-batched inputs
        if not is_batched:
            inputs = ops.expand_dims(inputs, batch_dim)
        
        # Get batch size and sequence length
        input_shape = ops.shape(inputs)
        batch_size = input_shape[batch_dim]
        seq_length = input_shape[seq_dim]
        
        # Initialize states if not provided
        if initial_state is None:
            # Create initial states for each layer and direction
            h_states = []
            
            for layer in range(self.num_layers):
                h_states.append(tensor.zeros((batch_size, self.hidden_size)))
                
                if self.bidirectional:
                    h_states.append(tensor.zeros((batch_size, self.hidden_size)))
        else:
            # Unpack provided initial states
            h_states = initial_state
        
        # Process sequence through each layer
        layer_outputs = inputs
        final_h_states = []
        
        for layer in range(self.num_layers):
            # Get cells for this layer
            forward_cell = self.forward_cells[layer]
            backward_cell = self.backward_cells[layer] if self.bidirectional else None
            
            # Get initial states for this layer
            layer_idx = layer * (2 if self.bidirectional else 1)
            forward_h = h_states[layer_idx]
            
            if self.bidirectional:
                backward_h = h_states[layer_idx + 1]
            
            # Process sequence for this layer
            forward_outputs = []
            backward_outputs = []
            
            # Forward direction
            for t in range(seq_length):
                # Get input for current time step
                if self.batch_first:
                    current_input = layer_outputs[:, t]
                else:
                    current_input = layer_outputs[t]
                
                # Apply forward cell
                forward_h, [forward_h] = forward_cell(current_input, [forward_h])
                forward_outputs.append(forward_h)
            
            # Backward direction (if bidirectional)
            if self.bidirectional:
                for t in range(seq_length - 1, -1, -1):
                    # Get input for current time step
                    if self.batch_first:
                        current_input = layer_outputs[:, t]
                    else:
                        current_input = layer_outputs[t]
                    
                    # Apply backward cell
                    backward_h, [backward_h] = backward_cell(current_input, [backward_h])
                    backward_outputs.insert(0, backward_h)
            
            # Combine outputs
            if self.bidirectional:
                combined_outputs = []
                for t in range(seq_length):
                    combined = ops.concat([forward_outputs[t], backward_outputs[t]], axis=-1)
                    combined_outputs.append(combined)
            else:
                combined_outputs = forward_outputs
            
            # Stack outputs for this layer
            if self.batch_first:
                layer_outputs = ops.stack(combined_outputs, axis=1)
            else:
                layer_outputs = ops.stack(combined_outputs, axis=0)
            
            # Apply dropout (except for the last layer)
            if layer < self.num_layers - 1 and self.dropout > 0:
                layer_outputs = ops.dropout(layer_outputs, self.dropout)
            
            # Store final states for this layer
            final_h_states.append(forward_h)
            
            if self.bidirectional:
                final_h_states.append(backward_h)
        
        # Prepare output
        if not self.return_sequences:
            if self.batch_first:
                outputs = layer_outputs[:, -1]
            else:
                outputs = layer_outputs[-1]
        else:
            outputs = layer_outputs
        
        # Handle non-batched outputs
        if not is_batched:
            outputs = ops.squeeze(outputs, batch_dim)
        
        # Prepare final state
        final_state = ops.stack(final_h_states)
        
        # Return outputs and states if requested
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
        h_states = []
        
        for layer in range(self.num_layers):
            h_states.append(tensor.zeros((batch_size, self.hidden_size)))
            
            if self.bidirectional:
                h_states.append(tensor.zeros((batch_size, self.hidden_size)))
        
        return ops.stack(h_states)