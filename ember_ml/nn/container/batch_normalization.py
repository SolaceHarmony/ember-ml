"""
Batch normalization module for ember_ml.

This module provides a backend-agnostic implementation of batch normalization
that works with any backend (NumPy, PyTorch, MLX).
"""

from typing import Optional, Union, Tuple, Any

from ember_ml import ops
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn import tensor
class BatchNormalization(Module):
    """
    Batch normalization layer.
    
    Normalizes the activations of the previous layer at each batch, i.e. applies a
    transformation that maintains the mean activation close to 0 and the activation
    standard deviation close to 1.
    
    Attributes:
        epsilon: Small constant for numerical stability
        momentum: Momentum for moving averages
        gamma: Scale parameter
        beta: Shift parameter
        moving_mean: Running mean
        moving_var: Running variance
        initialized: Whether the layer has been initialized
    """
    
    def __init__(self, epsilon: float = 1e-5, momentum: float = 0.9):
        """
        Initialize a batch normalization layer.
        
        Args:
            epsilon: Small constant for numerical stability
            momentum: Momentum for moving averages
        """
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = None  # Scale parameter
        self.beta = None   # Shift parameter
        self.moving_mean = None
        self.moving_var = None
        self.initialized = False
    
    def forward(self, x: Any) -> Any:
        """
        Forward pass through the layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Get input shape
        input_shape = tensor.shape(x)
        
        # Initialize parameters if not already done
        if not self.initialized:
            self.gamma = Parameter(tensor.ones((input_shape[-1],)))
            self.beta = Parameter(tensor.zeros((input_shape[-1],)))
            self.moving_mean = tensor.zeros((input_shape[-1],))
            self.moving_var = tensor.ones((input_shape[-1],))
            self.initialized = True
        
        # Training mode: use batch statistics
        # Compute batch mean and variance
        batch_mean = ops.mean(x, axis=(0, 1))
        batch_var = ops.var(x, axis=(0, 1))
        
        # Update moving statistics
        self.moving_mean = ops.add(
            ops.multiply(self.momentum, self.moving_mean),
            ops.multiply(1 - self.momentum, batch_mean)
        )
        self.moving_var = ops.add(
            ops.multiply(self.momentum, self.moving_var),
            ops.multiply(1 - self.momentum, batch_var)
        )
        
        # Normalize
        x_norm = ops.divide(
            ops.subtract(x, batch_mean),
            ops.sqrt(ops.add(batch_var, self.epsilon))
        )
        
        # Scale and shift
        return ops.add(ops.multiply(self.gamma, x_norm), self.beta)
    
    def __repr__(self) -> str:
        """Return a string representation of the layer."""
        return f"BatchNormalization(epsilon={self.epsilon}, momentum={self.momentum})"