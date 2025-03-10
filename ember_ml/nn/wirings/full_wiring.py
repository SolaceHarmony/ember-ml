"""
Fully Connected Wiring for neural circuit policies.

This module provides a fully connected wiring configuration for neural
circuit policies, where all neurons are connected to all other neurons.
"""

import numpy as np
from typing import Optional, Tuple

from ember_ml import ops
from ember_ml.nn.wirings.wiring import Wiring

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
        super().__init__(units, output_dim, input_dim, sparsity_level, seed)
    
    def build(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the fully connected wiring configuration.
        
        Returns:
            Tuple of (input_mask, recurrent_mask, output_mask)
        """
        # Set random seed for reproducibility
        if self.seed is not None:
            ops.set_seed(self.seed)
        
        # Create masks
        if self.sparsity_level > 0.0:
            # Create sparse masks
            input_mask = ops.cast(
                ops.random_uniform((self.input_dim,)) >= self.sparsity_level,
                ops.int32
            )
            recurrent_mask = ops.cast(
                ops.random_uniform((self.units, self.units)) >= self.sparsity_level,
                ops.int32
            )
            output_mask = ops.cast(
                ops.random_uniform((self.units,)) >= self.sparsity_level,
                ops.int32
            )
        else:
            # Create dense masks
            input_mask = ops.ones((self.input_dim,), dtype=ops.int32)
            recurrent_mask = ops.ones((self.units, self.units), dtype=ops.int32)
            output_mask = ops.ones((self.units,), dtype=ops.int32)
        
        # Convert to numpy arrays for consistency with the wiring interface
        input_mask = ops.to_numpy(input_mask)
        recurrent_mask = ops.to_numpy(recurrent_mask)
        output_mask = ops.to_numpy(output_mask)
        
        return input_mask, recurrent_mask, output_mask