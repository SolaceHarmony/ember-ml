"""
Neural Circuit Policy (NCP) wiring.

This module provides the NCP class, which implements a neural circuit policy
using a wiring configuration.
"""

from typing import Optional, Dict, Any

from ember_ml import ops
from ember_ml.nn.tensor import EmberTensor, zeros, convert_to_tensor, cast, expand_dims, tensor_scatter_nd_update, maximum, int32
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
            ops.set_seed(seed)
        
        # Generate masks
        self.input_mask = self._generate_input_mask()
        self.recurrent_mask = self._generate_recurrent_mask()
        self.output_mask = self._generate_output_mask()
    
    def _generate_input_mask(self) -> EmberTensor:
        """
        Generate the input mask.
        
        Returns:
            Input mask of shape (input_dim, units)
        """
        # Create a tensor of zeros
        mask = ops.zeros((self.input_dim, self.units))
        
        # Determine the number of connections per input
        connections_per_input = maximum(
            cast(ops.multiply(self.units, 1.0 - self.sparsity_level), int32),
            convert_to_tensor(1)
        )
        
        # Connect each input to a random subset of units
        for i in range(self.input_dim):
            # Generate random indices
            indices = ops.random_permutation(self.units)
            # Take the first connections_per_input indices
            target_indices = indices[:ops.cast(connections_per_input, int32)]
            # Set the mask values to 1.0 at these indices
            for idx in target_indices:
                mask_idx = ops.convert_to_tensor([i, idx], dtype=int32)
                mask = ops.tensor_scatter_nd_update(
                    mask, 
                    ops.expand_dims(mask_idx, 0), 
                    ops.convert_to_tensor([1.0])
                )
        
        return EmberTensor(mask)
    
    def _generate_recurrent_mask(self) -> EmberTensor:
        """
        Generate the recurrent mask.
        
        Returns:
            Recurrent mask of shape (units, units)
        """
        # Create a tensor of zeros
        mask = ops.zeros((self.units, self.units))
        
        # Determine the number of connections per unit
        connections_per_unit = maximum(
            cast(ops.multiply(self.units, 1.0 - self.sparsity_level), int32),
            convert_to_tensor(1)
        )
        
        # Connect each unit to a random subset of units
        for i in range(self.units):
            # Generate random indices
            indices = ops.random_permutation(self.units)
            # Take the first connections_per_unit indices
            target_indices = indices[:ops.cast(connections_per_unit, int32)]
            # Set the mask values to 1.0 at these indices
            for idx in target_indices:
                mask_idx = ops.convert_to_tensor([i, idx], dtype=int32)
                mask = ops.tensor_scatter_nd_update(
                    mask, 
                    ops.expand_dims(mask_idx, 0), 
                    ops.convert_to_tensor([1.0])
                )
        
        return EmberTensor(mask)
    
    def _generate_output_mask(self) -> EmberTensor:
        """
        Generate the output mask.
        
        Returns:
            Output mask of shape (units, output_dim)
        """
        # Create a tensor of zeros
        mask = ops.zeros((self.units, self.output_dim))
        
        # Determine the number of connections per output
        connections_per_output = maximum(
            cast(ops.multiply(self.units, 1.0 - self.sparsity_level), int32),
            convert_to_tensor(1)
        )
        
        # Connect each output to a random subset of units
        for i in range(self.output_dim):
            # Generate random indices
            indices = ops.random_permutation(self.units)
            # Take the first connections_per_output indices
            source_indices = indices[:ops.cast(connections_per_output, int32)]
            # Set the mask values to 1.0 at these indices
            for idx in source_indices:
                mask_idx = ops.convert_to_tensor([idx, i], dtype=int32)
                mask = ops.tensor_scatter_nd_update(
                    mask, 
                    ops.expand_dims(mask_idx, 0), 
                    ops.convert_to_tensor([1.0])
                )
        
        return EmberTensor(mask)
    
    def get_input_mask(self) -> EmberTensor:
        """
        Get the input mask.
        
        Returns:
            Input mask of shape (input_dim, units)
        """
        return self.input_mask
    
    def get_recurrent_mask(self) -> EmberTensor:
        """
        Get the recurrent mask.
        
        Returns:
            Recurrent mask of shape (units, units)
        """
        return self.recurrent_mask
    
    def get_output_mask(self) -> EmberTensor:
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