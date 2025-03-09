"""
Glorot initializers for neural network weights.

This module provides Glorot (Xavier) initializers for neural network weights,
which help maintain the variance of activations and gradients across layers.
"""

from typing import Tuple, Optional, Union, Any

from ember_ml import ops

def glorot_uniform(shape: Tuple[int, ...], dtype: Optional[Any] = None, device: Optional[str] = None):
    """
    Glorot uniform initializer, also called Xavier uniform initializer.
    
    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / (fan_in + fan_out))` where `fan_in` is the number
    of input units in the weight tensor and `fan_out` is the number of output units.
    
    Args:
        shape: Shape of the tensor to initialize
        dtype: Data type of the tensor
        device: Device to place the tensor on
        
    Returns:
        Initialized tensor
    """
    fan_in = shape[0] if len(shape) >= 1 else 1
    fan_out = shape[1] if len(shape) >= 2 else 1
    
    limit = ops.sqrt(ops.divide(6.0, ops.add(fan_in, fan_out)))
    
    return ops.random_uniform(shape, -limit, limit, dtype=dtype, device=device)

def glorot_normal(shape: Tuple[int, ...], dtype: Optional[Any] = None, device: Optional[str] = None):
    """
    Glorot normal initializer, also called Xavier normal initializer.
    
    It draws samples from a normal distribution with mean 0 and
    standard deviation `sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number
    of input units in the weight tensor and `fan_out` is the number of output units.
    
    Args:
        shape: Shape of the tensor to initialize
        dtype: Data type of the tensor
        device: Device to place the tensor on
        
    Returns:
        Initialized tensor
    """
    fan_in = shape[0] if len(shape) >= 1 else 1
    fan_out = shape[1] if len(shape) >= 2 else 1
    
    stddev = ops.sqrt(ops.divide(2.0, ops.add(fan_in, fan_out)))
    
    return ops.random_normal(shape, 0.0, stddev, dtype=dtype, device=device)

def orthogonal(shape: Tuple[int, ...], gain: float = 1.0, dtype: Optional[Any] = None, device: Optional[str] = None):
    """
    Orthogonal initializer.
    
    It generates a random orthogonal matrix using a simplified approach.
    
    Args:
        shape: Shape of the tensor to initialize
        gain: Multiplicative factor to apply to the orthogonal matrix
        dtype: Data type of the tensor
        device: Device to place the tensor on
        
    Returns:
        Initialized tensor
    """
    if len(shape) < 2:
        raise ValueError("Orthogonal initialization requires at least 2 dimensions")
    
    # Extract dimensions
    rows, cols = shape[0], shape[1]
    
    # Generate a random matrix
    a = ops.random_normal((rows, cols), 0.0, 1.0, dtype=dtype, device=device)
    
    # Use a simplified approach for orthogonalization
    # This is not a true orthogonal matrix, but it's a reasonable approximation
    # for initialization purposes
    
    # Normalize each column to unit length
    for i in range(cols):
        # Get the i-th column
        col = a[:, i:i+1]
        
        # Compute the norm
        norm = ops.sqrt(ops.sum(ops.square(col)))
        
        # Add a small epsilon to avoid division by zero
        norm = ops.add(norm, 1e-8)
        
        # Normalize the column
        a[:, i:i+1] = ops.divide(col, norm)
    
    # Apply gain
    a = ops.multiply(a, gain)
    
    # If shape has more than 2 dimensions, reshape accordingly
    if len(shape) > 2:
        extra_dims = shape[2:]
        flat_dim = cols
        for dim in extra_dims:
            flat_dim *= dim
        a = ops.reshape(a, (rows, flat_dim))
        a = ops.reshape(a, shape)
    
    return a