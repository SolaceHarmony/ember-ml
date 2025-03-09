"""
Neural Circuit Policy (NCP) wiring.

This module provides the NCP class, which implements a neural circuit policy
using a wiring configuration.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Union, List

from ember_ml import ops
from ember_ml.nn.wirings.wiring import Wiring

class NCP(Wiring):
    """
    Neural Circuit Policy (NCP) wiring.
    
    This class implements a neural circuit policy using a wiring configuration.
    It defines the connectivity pattern for a recurrent neural network with
    specific input, recurrent, and output connections.
    """
    
    def __init__(
        self,
        units: int,
        input_dim: int,
        output_dim: int,
        sparsity_level: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        Initialize an NCP wiring.
        
        Args:
            units: Number of units in the circuit
            input_dim: Input dimension
            output_dim: Output dimension
            sparsity_level: Sparsity level for the connections (default: 0.5)
            seed: Random seed for reproducibility
        """
        super().__init__(units=units, input_dim=input_dim, output_dim=output_dim)
        
        self.sparsity_level = sparsity_level
        self.seed = seed
        
        # Initialize random number generator
        if seed is not None:
            np.random.seed(seed)
        
        # Generate masks
        self.input_mask = self._generate_input_mask()
        self.recurrent_mask = self._generate_recurrent_mask()
        self.output_mask = self._generate_output_mask()
    
    def _generate_input_mask(self) -> np.ndarray:
        """
        Generate the input mask.
        
        Returns:
            Input mask of shape (input_dim, units)
        """
        mask = np.zeros((self.input_dim, self.units))
        
        # Determine the number of connections per input
        connections_per_input = max(int(self.units * (1.0 - self.sparsity_level)), 1)
        
        # Connect each input to a random subset of units
        for i in range(self.input_dim):
            target_indices = np.random.choice(
                self.units, size=connections_per_input, replace=False
            )
            mask[i, target_indices] = 1.0
        
        return mask
    
    def _generate_recurrent_mask(self) -> np.ndarray:
        """
        Generate the recurrent mask.
        
        Returns:
            Recurrent mask of shape (units, units)
        """
        mask = np.zeros((self.units, self.units))
        
        # Determine the number of connections per unit
        connections_per_unit = max(int(self.units * (1.0 - self.sparsity_level)), 1)
        
        # Connect each unit to a random subset of units
        for i in range(self.units):
            target_indices = np.random.choice(
                self.units, size=connections_per_unit, replace=False
            )
            mask[i, target_indices] = 1.0
        
        return mask
    
    def _generate_output_mask(self) -> np.ndarray:
        """
        Generate the output mask.
        
        Returns:
            Output mask of shape (units, output_dim)
        """
        mask = np.zeros((self.units, self.output_dim))
        
        # Determine the number of connections per output
        connections_per_output = max(int(self.units * (1.0 - self.sparsity_level)), 1)
        
        # Connect each output to a random subset of units
        for i in range(self.output_dim):
            source_indices = np.random.choice(
                self.units, size=connections_per_output, replace=False
            )
            mask[source_indices, i] = 1.0
        
        return mask
    
    def get_input_mask(self) -> np.ndarray:
        """
        Get the input mask.
        
        Returns:
            Input mask of shape (input_dim, units)
        """
        return self.input_mask
    
    def get_recurrent_mask(self) -> np.ndarray:
        """
        Get the recurrent mask.
        
        Returns:
            Recurrent mask of shape (units, units)
        """
        return self.recurrent_mask
    
    def get_output_mask(self) -> np.ndarray:
        """
        Get the output mask.
        
        Returns:
            Output mask of shape (units, output_dim)
        """
        return self.output_mask
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the NCP wiring.
        
        Returns:
            Dictionary containing the configuration
        """
        config = super().get_config()
        config.update({
            "sparsity_level": self.sparsity_level,
            "seed": self.seed,
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NCP':
        """
        Create an NCP wiring from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            NCP wiring
        """
        return cls(**config)