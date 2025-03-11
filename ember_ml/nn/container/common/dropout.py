"""
Dropout module for ember_ml.

This module provides a backend-agnostic implementation of a dropout layer
that works with any backend (NumPy, PyTorch, MLX).
"""

from typing import Optional, Any

from ember_ml import ops
from ember_ml.nn.modules.base_module import BaseModule as Module
from ember_ml.nn.container.interfaces import DropoutInterface

class Dropout(Module, DropoutInterface):
    """
    Dropout layer.
    
    Applies dropout to the input tensor during training, which helps prevent overfitting.
    
    Attributes:
        rate: Fraction of the input units to drop (between 0 and 1)
        seed: Optional random seed for reproducibility
    """
    
    def __init__(self, rate: float, seed: Optional[int] = None):
        """
        Initialize a dropout layer.
        
        Args:
            rate: Fraction of the input units to drop (between 0 and 1)
            seed: Optional random seed for reproducibility
        """
        super().__init__()
        self.rate = rate
        self.seed = seed
    
    def forward(self, x: Any, training: bool = False) -> Any:
        """
        Forward pass through the layer.
        
        Args:
            x: Input tensor
            training: Whether to apply dropout (True) or return the input unchanged (False)
            
        Returns:
            Output tensor with dropout applied (if training is True)
        """
        if not training or self.rate == 0:
            return x
        
        # Ensure rate is between 0 and 1
        rate = max(0., min(1., self.rate))
        
        # Create a random mask
        if self.seed is not None:
            ops.set_seed(self.seed)
        
        # Generate a mask with 1s where we keep the values and 0s where we drop them
        keep_prob = 1.0 - rate
        mask = ops.random_uniform(ops.shape(x), 0.0, 1.0) < keep_prob
        
        # Convert mask to the same dtype as x
        mask = ops.cast(mask, ops.dtype(x))
        
        # Apply the mask and scale the output
        output = ops.multiply(x, mask) / keep_prob
        
        return output
    
    def __repr__(self) -> str:
        """Return a string representation of the layer."""
        return f"Dropout(rate={self.rate})"