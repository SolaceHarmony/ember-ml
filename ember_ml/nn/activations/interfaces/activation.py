"""
Activation function interfaces.

This module defines the abstract interfaces for activation functions used in neural networks.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union, List, Tuple

from ember_ml.ops.tensor import EmberTensor

# Type aliases
Tensor = EmberTensor


class ActivationInterface(ABC):
    """Abstract interface for activation functions used in neural networks."""
    
    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply the activation function.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with activation applied
        """
        pass
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the activation function.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with activation applied
        """
        pass