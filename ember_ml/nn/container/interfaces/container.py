"""
Container interfaces.

This module defines the abstract interfaces for container operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Union, Tuple, Dict

from ember_ml.ops.tensor import EmberTensor

# Type aliases
Tensor = EmberTensor


class ContainerInterfaces(ABC):
    """Abstract interfaces for container operations."""
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through a container.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def add(self, layer: Any) -> None:
        """
        Add a layer to the container.
        
        Args:
            layer: Layer to add
        """
        pass
    
    @abstractmethod
    def build(self, input_shape: Union[List[int], Tuple[int, ...]]) -> None:
        """
        Build the container for a specific input shape.
        
        Args:
            input_shape: Shape of the input tensor
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the container.
        
        Returns:
            Dictionary containing the configuration
        """
        pass
    
    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the state dictionary of the container.
        
        Returns:
            Dictionary containing the state
        """
        pass
    
    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state dictionary into the container.
        
        Args:
            state_dict: Dictionary containing the state
        """
        pass


__all__ = [
    'ContainerInterfaces',
]