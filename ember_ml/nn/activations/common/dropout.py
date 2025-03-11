"""
Dropout activation function implementation.

This module provides a backend-agnostic implementation of the dropout activation function
using the ops abstraction layer.
"""

from typing import Any, Optional, Union, List, Tuple

from ember_ml import ops
from ember_ml.ops.tensor import EmberTensor
from ember_ml.nn.activations.interfaces.activation import ActivationInterface

# Type aliases
Tensor = EmberTensor


class Dropout(ActivationInterface):
    """
    Applies Dropout to the input.
    
    During training, randomly zeroes some of the elements of the input tensor with probability p.
    
    Args:
        rate: Probability of an element to be zeroed. Default: 0.5
        training: If True, applies dropout. If False, returns the input unchanged.
    """
    
    def __init__(self, rate: float = 0.5, training: bool = True):
        """
        Initialize Dropout activation function.
        
        Args:
            rate: Probability of an element to be zeroed. Default: 0.5
            training: If True, applies dropout. If False, returns the input unchanged.
        """
        self.rate = rate
        self.training = training
        
    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply dropout activation function.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with dropout applied
        """
        if not self.training or self.rate == 0.0:
            return x
            
        # Create a random mask
        mask = ops.greater_equal(
            ops.random_uniform(ops.shape(x)),
            ops.convert_to_tensor(self.rate)
        )
        
        # Apply mask and scale
        scale = ops.convert_to_tensor(1.0 / (1.0 - self.rate))
        return ops.multiply(ops.where(mask, x, ops.zeros_like(x)), scale)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of dropout activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with dropout applied
        """
        return self.__call__(x)