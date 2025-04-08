"""
Closed-form Continuous-time (CfC) Neural Network

This module provides an implementation of CfC cells and layers,
which are a type of recurrent neural network that operates in continuous time.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

from ember_ml import ops
from ember_ml.nn import tensor
# Updated imports for Wiring classes from nn.modules
from ember_ml.nn.modules.wiring import NeuronMap # Use renamed base class
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn.initializers import glorot_uniform, orthogonal
# CfC layer definition

# Import the correct CfCCell definition
from .cfc_cell import CfCCell

class CfC(Module):
    """
    Closed-form Continuous-time (CfC) RNN layer.
    
    This layer wraps a CfCCell or WiredCfCCell to create a recurrent layer.
    """
    
    def __init__(
        self,
        cell_or_map,
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: bool = False,
        mixed_memory: bool = False,
        **kwargs
    ):
        """
        Initialize the CfC layer.
        
        Args:
            cell_or_map: CfCCell, WiredCfCCell, or NeuronMap instance
            return_sequences: Whether to return the full sequence or just the last output
            return_state: Whether to return the final state
            go_backwards: Whether to process the sequence backwards
            mixed_memory: Whether to use mixed memory
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        
        # Import WiredCfCCell
        from ember_ml.nn.modules.rnn.wired_cfc_cell import WiredCfCCell
        if isinstance(cell_or_map, (CfCCell, WiredCfCCell)):
            self.cell = cell_or_map
        elif isinstance(cell_or_map, NeuronMap): # Check against renamed base class
            # Pass only the map and other relevant args. Build handles input_size.
            self.cell = WiredCfCCell(neuron_map=cell_or_map, mixed_memory=mixed_memory, **kwargs)
        else:
            # If an integer units is passed, create a default CfCCell
            # This part might need adjustment based on desired API for non-wired CfC layer
            if isinstance(cell_or_map, int):
                 # Need input_size for CfCCell - how is this passed to the CfC layer?
                 # Requires CfC __init__ signature change if we support this.
                 # For now, assume cell_or_map must be Cell or NeuronMap instance.
                 raise ValueError("Initializing CfC layer with integer units is not fully supported yet. Provide a CfCCell or NeuronMap.")
            else:
                raise ValueError("cell_or_map must be a CfCCell, WiredCfCCell, or NeuronMap instance")
        
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.mixed_memory = mixed_memory # Although applied to cell, store flag for config?
    
    def forward(self, inputs, initial_state=None):
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input tensor (batch, time, features)
            initial_state: Initial state(s) for the cell
            
        Returns:
            Layer output(s)
        """
        # Get input shape
        input_shape = tensor.shape(inputs)
        if len(input_shape) != 3:
             raise ValueError("Input tensor must be 3D (batch, time, features)")
        batch_size, time_steps, _ = input_shape # input_dim inferred by cell potentially
        
        # Create initial state if not provided
        if initial_state is None:
            initial_state = self.cell.reset_state(batch_size)
        
        # Process sequence
        outputs = []
        states = initial_state
        
        # Process sequence in reverse if go_backwards is True
        time_indices = range(time_steps - 1, -1, -1) if self.go_backwards else range(time_steps)
        
        # Process each time step
        for t in time_indices:
            # Slicing inputs: inputs[:, t] gives shape (batch, features)
            output, states = self.cell(inputs[:, t], states)
            outputs.append(output)

        # If processing backwards, reverse the outputs sequence
        if self.go_backwards:
            outputs.reverse()

        # Stack outputs
        if self.return_sequences:
            outputs_tensor = tensor.stack(outputs, axis=1) # Stack along time dimension
        else:
            outputs_tensor = outputs[-1] # Return only the last output
        
        # Return outputs and states if requested
        if self.return_state:
            return outputs_tensor, states
        else:
            return outputs_tensor
    
    def reset_state(self, batch_size=1):
        """
        Reset the layer state by resetting the underlying cell's state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state from the cell
        """
        return self.cell.reset_state(batch_size)

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the CfC layer."""
        config = super().get_config()
        config.update({
            # Save config needed to reconstruct the cell
            "cell_config": self.cell.get_config(),
            "cell_class_name": self.cell.__class__.__name__,
            # Save layer args
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "go_backwards": self.go_backwards,
            "mixed_memory": self.mixed_memory,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CfC':
        """Creates a CfC layer from its configuration."""
        cell_config = config.pop("cell_config")
        cell_class_name = config.pop("cell_class_name")

        # Import cell classes dynamically
        if cell_class_name == "CfCCell":
            from .cfc_cell import CfCCell as cell_cls
        elif cell_class_name == "WiredCfCCell":
            from .wired_cfc_cell import WiredCfCCell as cell_cls
        else:
            raise TypeError(f"Unsupported cell type for CfC: {cell_class_name}")

        # Reconstruct the cell instance
        cell = cell_cls.from_config(cell_config)

        # Prepare config for CfC layer __init__
        layer_config = config # Start with remaining config
        layer_config['cell_or_map'] = cell # Pass the reconstructed cell

        return super(CfC, cls).from_config(layer_config) # Calls cls(**layer_config)
