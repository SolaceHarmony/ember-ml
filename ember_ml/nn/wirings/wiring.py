"""
Base Wiring class for neural circuit policies.

This module provides the base Wiring class that defines the interface
for all wiring configurations.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Union

from ember_ml import ops

class Wiring:
    """
    Base class for all wiring configurations.
    
    Wiring configurations define the connectivity patterns between neurons
    in neural circuit policies. They specify which neurons are connected to
    which other neurons, and with what weights.
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
        Initialize a wiring configuration.
        
        Args:
            units: Number of units in the circuit
            output_dim: Number of output dimensions (default: units)
            input_dim: Number of input dimensions (default: units)
            sparsity_level: Sparsity level for the connections (default: 0.5)
            seed: Random seed for reproducibility
        """
        self.units = units
        self.output_dim = output_dim if output_dim is not None else units
        self.input_dim = input_dim if input_dim is not None else units
        self.sparsity_level = sparsity_level
        self.seed = seed
        
        # Initialize masks
        self._input_mask = None
        self._recurrent_mask = None
        self._output_mask = None
        
    def build(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the wiring configuration.
        
        This method should be overridden by all subclasses to implement
        the specific wiring pattern.
        
        Returns:
            Tuple of (input_mask, recurrent_mask, output_mask)
        """
        raise NotImplementedError("Subclasses must implement build method")
    
    def get_input_mask(self) -> np.ndarray:
        """
        Get the input mask.
        
        The input mask determines which input dimensions are connected to
        which neurons in the circuit.
        
        Returns:
            Input mask as a numpy array
        """
        if self._input_mask is None:
            self._input_mask, self._recurrent_mask, self._output_mask = self.build()
        
        # If the mask is an EmberTensor, return its data property
        if hasattr(self._input_mask, 'data'):
            return self._input_mask.data
        return self._input_mask
    
    def get_recurrent_mask(self) -> np.ndarray:
        """
        Get the recurrent mask.
        
        The recurrent mask determines which neurons in the circuit are
        connected to which other neurons.
        
        Returns:
            Recurrent mask as a numpy array
        """
        if self._recurrent_mask is None:
            self._input_mask, self._recurrent_mask, self._output_mask = self.build()
        
        # If the mask is an EmberTensor, return its data property
        if hasattr(self._recurrent_mask, 'data'):
            return self._recurrent_mask.data
        return self._recurrent_mask
    
    def get_output_mask(self) -> np.ndarray:
        """
        Get the output mask.
        
        The output mask determines which neurons in the circuit contribute
        to which output dimensions.
        
        Returns:
            Output mask as a numpy array
        """
        if self._output_mask is None:
            self._input_mask, self._recurrent_mask, self._output_mask = self.build()
        
        # If the mask is an EmberTensor, return its data property
        if hasattr(self._output_mask, 'data'):
            return self._output_mask.data
        return self._output_mask
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the wiring.
        
        Returns:
            Dictionary containing the configuration
        """
        return {
            "units": self.units,
            "output_dim": self.output_dim,
            "input_dim": self.input_dim,
            "sparsity_level": self.sparsity_level,
            "seed": self.seed
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Wiring':
        """
        Create a wiring configuration from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Wiring configuration
        """
        return cls(**config)