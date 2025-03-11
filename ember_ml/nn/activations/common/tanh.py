"""
Tanh activation function implementation.

This module provides a backend-agnostic implementation of the tanh activation function
using the ops abstraction layer.
"""

from typing import Any, Optional, Union, List, Tuple

from ember_ml import ops
from ember_ml.ops.tensor import EmberTensor
from ember_ml.nn.activations.interfaces.activation import ActivationInterface

# Type aliases
Tensor = EmberTensor


class Tanh(ActivationInterface):
    """
    Applies the Hyperbolic Tangent (Tanh) function element-wise.
    
    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Args:
        None
    """
    
    def __init__(self):
        """Initialize Tanh activation function."""
        pass
        
    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply tanh activation function.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with tanh activation applied
        """
        return ops.tanh(x)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of tanh activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with tanh activation applied
        """
        return self.__call__(x)