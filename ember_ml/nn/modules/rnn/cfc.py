"""
Closed-form Continuous-time (CfC) Layer

This module provides an implementation of the CfC layer,
which wraps a CfCCell or WiredCfCCell to create a recurrent layer.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

from ember_ml import ops
from ember_ml.nn.modules import Module
from ember_ml.nn.wirings import Wiring
from ember_ml.nn.modules.rnn.cfc_cell import CfCCell
from ember_ml.nn.modules.rnn.wired_cfc_cell import WiredCfCCell

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
            cell_or_units: CfCCell, WiredCfCCell, Wiring, or number of units
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
    
    def forward(self, inputs, initial_state=None, timespans=None):
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input tensor
            initial_state: Initial state
            timespans: Time spans for continuous-time dynamics (default: 1.0)
            
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
            # Get timespan for current step
            ts = 1.0 if timespans is None else timespans[:, t]
            
            # Forward pass through cell
            output, states = self.cell(inputs[:, t], states, ts)
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