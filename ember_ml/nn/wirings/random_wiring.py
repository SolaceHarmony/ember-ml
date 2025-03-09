"""
Random Wiring for neural circuit policies.

This module provides a random wiring configuration for neural
circuit policies, where connections are randomly generated.
"""

import numpy as np
from typing import Optional, Tuple

from ember_ml import ops
from ember_ml.nn.wirings.wiring import Wiring

class RandomWiring(Wiring):
    """
    Random wiring configuration.
    
    In a random wiring, connections between neurons are randomly generated
    based on the specified sparsity level.
    """
    
    def __init__(
        self, 
        units: int, 
        output_dim: Optional[int] = None, 
        input_dim: Optional[int] = None,
        sparsity_level: float = 0.5, 
        seed: Optional[int] = None
    ):
        """
        Initialize a random wiring configuration.
        
        Args:
            units: Number of units in the circuit
            output_dim: Number of output dimensions (default: units)
            input_dim: Number of input dimensions (default: units)
            sparsity_level: Sparsity level for the connections (default: 0.5)
            seed: Random seed for reproducibility
        """
        super().__init__(units, output_dim, input_dim, sparsity_level, seed)
    
    def build(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the random wiring configuration.
        
        Returns:
            Tuple of (input_mask, recurrent_mask, output_mask)
        """
        # Set random seed for reproducibility
        if self.seed is not None:
            ops.set_random_seed(self.seed)
        
        # Create random masks
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
        
        # Convert to numpy arrays for consistency with the wiring interface
        input_mask = ops.to_numpy(input_mask)
        recurrent_mask = ops.to_numpy(recurrent_mask)
        output_mask = ops.to_numpy(output_mask)
        
        return input_mask, recurrent_mask, output_mask