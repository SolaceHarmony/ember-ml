"""
Auto Neural Circuit Policy (AutoNCP) wiring.

This module provides the AutoNCP class, which is a convenience wrapper
around the NCPWiring class that automatically configures the wiring.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Union, List

from ember_ml import ops
from ember_ml.nn.wirings.ncp_wiring import NCPWiring

class AutoNCP(NCPWiring):
    """
    Auto Neural Circuit Policy (AutoNCP) wiring.
    
    This class is a convenience wrapper around the NCPWiring class that
    automatically configures the wiring based on the number of units
    and outputs.
    """
    
    def __init__(
        self,
        units: int,
        output_size: int,
        sparsity_level: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        Initialize an AutoNCP wiring.
        
        Args:
            units: Number of units in the circuit
            output_size: Number of output dimensions
            sparsity_level: Sparsity level for the connections (default: 0.5)
            seed: Random seed for reproducibility
        """
        if output_size >= units - 2:
            raise ValueError(
                f"Output size must be less than the number of units-2 (given {units} units, {output_size} output size)"
            )
        if sparsity_level < 0.1 or sparsity_level > 1.0:
            raise ValueError(
                f"Sparsity level must be between 0.0 and 0.9 (given {sparsity_level})"
            )
        
        # Calculate the number of inter and command neurons
        density_level = 1.0 - sparsity_level
        inter_and_command_neurons = units - output_size
        command_neurons = max(int(0.4 * inter_and_command_neurons), 1)
        inter_neurons = inter_and_command_neurons - command_neurons
        
        # Calculate the fanout and fanin parameters
        sensory_fanout = max(int(inter_neurons * density_level), 1)
        inter_fanout = max(int(command_neurons * density_level), 1)
        recurrent_command_synapses = max(int(command_neurons * density_level * 2), 1)
        motor_fanin = max(int(command_neurons * density_level), 1)
        
        # Initialize the NCPWiring
        super().__init__(
            inter_neurons=inter_neurons,
            motor_neurons=output_size,
            sensory_neurons=0,  # No sensory neurons in AutoNCP
            sparsity_level=sparsity_level,
            seed=seed,
        )
        
        # Store the AutoNCP-specific parameters
        self.units = units
        self.output_size = output_size
        self.sparsity_level = sparsity_level
        self.seed = seed
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the AutoNCP wiring.
        
        Returns:
            Dictionary containing the configuration
        """
        config = super().get_config()
        config.update({
            "units": self.units,
            "output_size": self.output_size,
            "sparsity_level": self.sparsity_level,
            "seed": self.seed,
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AutoNCP':
        """
        Create an AutoNCP wiring from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            AutoNCP wiring
        """
        # Extract the AutoNCP-specific parameters
        units = config.pop("units")
        output_size = config.pop("output_size")
        sparsity_level = config.pop("sparsity_level", 0.5)
        seed = config.pop("seed", None)
        
        # Create the AutoNCP wiring
        return cls(
            units=units,
            output_size=output_size,
            sparsity_level=sparsity_level,
            seed=seed,
        )