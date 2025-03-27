"""
Fully Connected Wiring for neural circuit policies.

This module provides a fully connected wiring configuration for neural
circuit policies, where all neurons are connected to all other neurons.
"""

import numpy as np
from typing import Optional, Tuple

from ember_ml import ops
from ember_ml.nn.wirings.wiring import Wiring
from ember_ml.nn import tensor

class FullyConnectedWiring(Wiring):
    """
    Fully connected wiring configuration.
    
    In a fully connected wiring, all neurons are connected to all other neurons,
    and all inputs are connected to all neurons.
    """
    
    def __init__(
        self,
        units: int,
        output_dim: Optional[int] = None,
        input_dim: Optional[int] = None,
        sparsity_level: float = 0.0,
        seed: Optional[int] = None
    ):
        """
        Initialize a fully connected wiring configuration.
        
        Args:
            units: Number of units in the circuit
            output_dim: Number of output dimensions (default: units)
            input_dim: Number of input dimensions (default: units)
            sparsity_level: Sparsity level for the connections (default: 0.0)
            seed: Random seed for reproducibility
        """
        # Don't pass input_dim to the parent class to avoid conflicts
        super().__init__(units, output_dim, None, sparsity_level, seed)
        # Store input_dim for later use
        self._stored_input_dim = input_dim
    
    def build(self, input_dim=None) -> Tuple:
        """
        Build the fully connected wiring configuration.
        
        Args:
            input_dim: Input dimension (optional)
        
        Returns:
            Tuple of (input_mask, recurrent_mask, output_mask)
        """
        # Use stored input_dim if available and no input_dim is provided
        if input_dim is None and self._stored_input_dim is not None:
            input_dim = self._stored_input_dim
        
        # Handle input_dim directly
        if input_dim is not None:
            self.set_input_dim(input_dim)
        
        # Create sensory synapses
        if self.sensory_adjacency_matrix is not None:
            for src in range(self.input_dim):
                for dest in range(self.units):
                    # Use a bias towards excitatory connections (2/3 probability)
                    polarity = 1 if tensor.random_uniform(()) > 0.33 else -1
                    self.add_sensory_synapse(src, dest, polarity)
        
        # Create internal synapses
        for src in range(self.units):
            for dest in range(self.units):
                # Use a bias towards excitatory connections (2/3 probability)
                polarity = 1 if tensor.random_uniform(()) > 0.33 else -1
                self.add_synapse(src, dest, polarity)
        
        # Create masks
        input_mask = tensor.ones((self.input_dim,), dtype='int32')
        recurrent_mask = tensor.ones((self.units, self.units), dtype='int32')
        output_mask = tensor.ones((self.units,), dtype='int32')
        
        return input_mask, recurrent_mask, output_mask
        # Set random seed for reproducibility
        if self.seed is not None:
            tensor.set_seed(self.seed)
        
        # Create masks
        if self.sparsity_level > 0.0:
            # Create sparse masks
            input_mask = tensor.cast(
                tensor.random_uniform((self.input_dim,)) >= self.sparsity_level,
                ops.int32
            )
            recurrent_mask = tensor.cast(
                tensor.random_uniform((self.units, self.units)) >= self.sparsity_level,
                ops.int32
            )
            output_mask = tensor.cast(
                tensor.random_uniform((self.units,)) >= self.sparsity_level,
                ops.int32
            )
        else:
            # Create dense masks
            input_mask = tensor.ones((self.input_dim,), dtype=ops.int32)
            recurrent_mask = tensor.ones((self.units, self.units), dtype=ops.int32)
            output_mask = tensor.ones((self.units,), dtype=ops.int32)
        
        # Convert to numpy arrays for consistency with the wiring interface
        input_mask = tensor.to_numpy(input_mask)
        recurrent_mask = tensor.to_numpy(recurrent_mask)
        output_mask = tensor.to_numpy(output_mask)
        
        return input_mask, recurrent_mask, output_mask