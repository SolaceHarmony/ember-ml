"""
Activation functions for emberharmony.

This module provides backend-agnostic implementations of common activation
functions that work with any backend (NumPy, PyTorch, MLX).
"""

from ember_ml import backend as K
from ember_ml.nn.module import Module

class ReLU(Module):
    """
    Applies the Rectified Linear Unit (ReLU) function element-wise:
    ReLU(x) = max(0, x)
    
    Args:
        inplace: If True, modify the input tensor in-place (not supported in all backends)
    """
    
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x):
        """
        Forward pass of the ReLU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with ReLU activation applied
        """
        return K.relu(x)
    
    def __repr__(self):
        return f"ReLU(inplace={self.inplace})"

class Sigmoid(Module):
    """
    Applies the Sigmoid function element-wise:
    Sigmoid(x) = 1 / (1 + exp(-x))
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Forward pass of the Sigmoid activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Sigmoid activation applied
        """
        return K.sigmoid(x)
    
    def __repr__(self):
        return "Sigmoid()"

class Tanh(Module):
    """
    Applies the Hyperbolic Tangent (Tanh) function element-wise:
    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Forward pass of the Tanh activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Tanh activation applied
        """
        return K.tanh(x)
    
    def __repr__(self):
        return "Tanh()"

class Softmax(Module):
    """
    Applies the Softmax function to an n-dimensional input tensor.
    
    The Softmax function is defined as:
    Softmax(x_i) = exp(x_i) / sum(exp(x_j))
    
    Args:
        dim: Dimension along which Softmax will be computed (default: -1)
    """
    
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        """
        Forward pass of the Softmax activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Softmax activation applied
        """
        return K.softmax(x, axis=self.dim)
    
    def __repr__(self):
        return f"Softmax(dim={self.dim})"