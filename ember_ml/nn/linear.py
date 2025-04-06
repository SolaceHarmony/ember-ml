"""
Linear layer implementation for ember_ml.

This module provides a backend-agnostic implementation of a fully connected
(linear) layer that works with any backend (NumPy, PyTorch, MLX).
"""

from typing import Optional, Union, Tuple

from ember_ml import backend as K
from ember_ml.nn.modules import Module, Parameter

class Linear(Module):
    """
    Applies a linear transformation to the incoming data: y = x @ W.T + b
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If set to False, the layer will not learn an additive bias
        device: Device to place the parameters on
        dtype: Data type of the parameters
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[str] = None,
        dtype: Optional[any] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using Kaiming initialization (He initialization)
        # This is a good default for layers followed by ReLU
        std = (2.0 / in_features) ** 0.5
        weight_data = K.random_normal(
            (out_features, in_features),
            mean=0.0,
            stddev=std,
            dtype=dtype,
            device=device
        )
        self.weight = Parameter(weight_data)
        
        if bias:
            # Initialize bias to zeros
            bias_data = K.zeros(out_features, dtype=dtype, device=device)
            self.bias = Parameter(bias_data)
        else:
            self.bias = None
    
    def forward(self, x):
        """
        Forward pass of the linear layer.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Ensure x is a tensor
        x = K.convert_to_tensor(x)
        
        # Compute the linear transformation
        output = K.matmul(x, K.transpose(self.weight))
        
        # Add bias if present
        if self.bias is not None:
            output = K.add(output, self.bias)
        
        return output
    
    def extra_repr(self) -> str:
        """Return a string with extra information."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"