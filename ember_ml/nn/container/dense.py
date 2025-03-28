"""
Dense (fully connected) module for ember_ml.

This module provides a backend-agnostic implementation of a dense (fully connected)
layer that works with any backend (NumPy, PyTorch, MLX).
"""

from typing import Optional, Union, Tuple, Any, Callable

from ember_ml import ops
from ember_ml.nn.modules import Module, Parameter

class Dense(Module):
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
    
    def __init__(self, units: int, activation: Optional[str] = None):
        """
        Initialize a dense layer.
        
        Args:
            units: Dimensionality of the output space
            activation: Activation function to use
        """
        super().__init__()
        self.units = units
        self.activation = activation
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
        input_shape = ops.shape(x)
        
        # Initialize parameters if not already done
        if not self.initialized:
            # Last dimension is the input dimension
            input_dim = input_shape[-1]
            
            # Initialize weights and bias using Glorot uniform initialization
            self.kernel = Parameter(ops.random_uniform(
                (input_dim, self.units),
                minval=-ops.sqrt(6.0 / (input_dim + self.units)),
                maxval=ops.sqrt(6.0 / (input_dim + self.units))
            ))
            self.bias = Parameter(ops.zeros((self.units,)))
            self.initialized = True
        
        # Reshape input if needed
        original_shape = input_shape
        if len(input_shape) > 2:
            # Flatten all dimensions except the last one
            x = ops.reshape(x, (-1, input_shape[-1]))
        
        # Linear transformation
        output = ops.add(ops.matmul(x, self.kernel), self.bias)
        
        # Apply activation if specified
        if self.activation is not None:
            if self.activation == "tanh":
                output = ops.tanh(output)
            elif self.activation == "sigmoid":
                output = ops.sigmoid(output)
            elif self.activation == "relu":
                output = ops.relu(output)
            else:
                # Try to get the activation function from ops
                activation_fn = getattr(ops, self.activation, None)
                if activation_fn is not None:
                    output = activation_fn(output)
                else:
                    raise ValueError(f"Unknown activation function: {self.activation}")
        
        # Reshape output if needed
        if len(original_shape) > 2:
            output_shape = list(original_shape[:-1]) + [self.units]
            output = ops.reshape(output, output_shape)
        
        return output
    
    def __repr__(self) -> str:
        """Return a string representation of the layer."""
        return f"Dense(units={self.units}, activation={self.activation})"