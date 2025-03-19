"""
Sigmoid activation function implementation.

This module provides a backend-agnostic implementation of the sigmoid activation function
using the ops abstraction layer.
"""

from ember_ml import ops
from ember_ml.nn.tensor import EmberTensor
from ember_ml.nn.activations.interfaces.activation import ActivationInterface


class Sigmoid(ActivationInterface):
    """
    Applies the Sigmoid function element-wise.
    
    Sigmoid(x) = 1 / (1 + exp(-x))
    
    Args:
        None
    """
    
    def __init__(self):
        """Initialize Sigmoid activation function."""
        pass
        
    def __call__(self, x: EmberTensor) -> EmberTensor:
        """
        Apply sigmoid activation function.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with sigmoid activation applied
        """
        return ops.sigmoid(x)
        
    def forward(self, x: EmberTensor) -> EmberTensor:
        """
        Forward pass of sigmoid activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with sigmoid activation applied
        """
        return self.__call__(x)