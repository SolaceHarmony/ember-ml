"""
Softmax activation function implementation.

This module provides a backend-agnostic implementation of the softmax activation function
using the ops abstraction layer.
"""

from typing import Any, Optional, Union, List, Tuple

from ember_ml import ops
from ember_ml.ops.tensor import EmberTensor
from ember_ml.nn.activations.interfaces.activation import ActivationInterface

# Type aliases
Tensor = EmberTensor


class Softmax(ActivationInterface):
    """
    Applies the Softmax function element-wise.
    
    Softmax(x)_i = exp(x_i) / sum_j(exp(x_j))
    
    Args:
        axis: The axis along which to apply softmax
    """
    
    def __init__(self, axis: int = -1):
        """
        Initialize Softmax activation function.
        
        Args:
            axis: The axis along which to apply softmax
        """
        self.axis = axis
        
    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply softmax activation function.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with softmax activation applied
        """
        exp_x = ops.exp(x)
        return ops.divide(exp_x, ops.sum(exp_x, axis=self.axis, keepdims=True))
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of softmax activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with softmax activation applied
        """
        return self.__call__(x)