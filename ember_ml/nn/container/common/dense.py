"""
Dense (fully connected) module for ember_ml.

This module provides a backend-agnostic implementation of a dense (fully connected)
layer that works with any backend (NumPy, PyTorch, MLX).
"""

from typing import Optional, Any

from ember_ml import ops
from ember_ml.nn.modules.base_module import BaseModule as Module, Parameter
from ember_ml.nn.container.interfaces import DenseInterface
from ember_ml.nn import tensor
class Dense(Module, DenseInterface):
    """
    Dense (fully connected) layer.
    
    Implements the operation: output = activation(dot(input, kernel) + bias)
    
    Attributes:
        units: Dimensionality of the output space
        activation: Activation function to use
        kernel: Weight matrix
        bias: Bias vector
        initialized: Whether the layer has been initialized
    """
    
    def __init__(self, units: int, activation: Optional[str] = None, use_bias: bool = True):
        """
        Initialize a dense layer.
        
        Args:
            units: Dimensionality of the output space
            activation: Activation function to use
            use_bias: Whether to use bias
        """
        super().__init__()
        self.units = units
        self.activation_name = activation
        self.activation = None
        if activation is not None:
            self.activation = ops.get_activation(activation)
        self.use_bias = use_bias
        self.kernel = None
        self.bias = None
        self.initialized = False
    
    def forward(self, x: Any) -> Any:
        """
        Forward pass through the layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Get input shape
        input_shape = tensor.shape(x)
        
        # Initialize parameters if not already done
        if not self.initialized:
            # Last dimension is the input dimension
            input_dim = input_shape[-1]
            
            # Initialize weights using Glorot uniform initialization
            self.kernel = Parameter(tensor.random_uniform(
                (input_dim, self.units),
                minval=-ops.sqrt(6.0 / (input_dim + self.units)),
                maxval=ops.sqrt(6.0 / (input_dim + self.units))
            ))
            
            if self.use_bias:
                self.bias = Parameter(tensor.zeros((self.units,)))
            
            self.initialized = True
        
        # Reshape input if needed
        original_shape = input_shape
        if len(input_shape) > 2:
            # Flatten all dimensions except the last one
            x = tensor.reshape(x, (-1, input_shape[-1]))
        
        # Linear transformation
        output = ops.matmul(x, self.kernel)
        
        if self.use_bias:
            output = ops.add(output, self.bias)
        
        # Apply activation if specified
        if self.activation is not None:
            output = self.activation(output)
        
        # Reshape output if needed
        if len(original_shape) > 2:
            output_shape = list(original_shape[:-1]) + [self.units]
            output = tensor.reshape(output, output_shape)
        
        return output
    
    def __repr__(self) -> str:
        """Return a string representation of the layer."""
        return f"Dense(units={self.units}, activation={self.activation_name})"