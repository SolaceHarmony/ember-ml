"""
Implementation of the power operation for different backends.

This file provides implementations of the power operation for NumPy, PyTorch, and MLX backends.
"""

def numpy_power(x, y):
    """
    Element-wise power operation for NumPy backend.
    
    Parameters
    ----------
    x : tensor
        Base tensor.
    y : tensor or scalar
        Exponent tensor or scalar.
    
    Returns
    -------
    tensor
        Element-wise power of x raised to y.
    """
    import numpy as np
    return np.power(x, y)

def torch_power(x, y):
    """
    Element-wise power operation for PyTorch backend.
    
    Parameters
    ----------
    x : tensor
        Base tensor.
    y : tensor or scalar
        Exponent tensor or scalar.
    
    Returns
    -------
    tensor
        Element-wise power of x raised to y.
    """
    import torch
    # Ensure x is a PyTorch tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Handle different types of y
    if isinstance(y, torch.Tensor):
        return torch.pow(x, y)
    elif isinstance(y, (int, float)):
        return torch.pow(x, y)
    else:
        # Convert y to a PyTorch tensor
        y = torch.tensor(y, dtype=torch.float32)
        return torch.pow(x, y)

def mlx_power(x, y):
    """
    Element-wise power operation for MLX backend.
    
    Parameters
    ----------
    x : tensor
        Base tensor.
    y : tensor or scalar
        Exponent tensor or scalar.
    
    Returns
    -------
    tensor
        Element-wise power of x raised to y.
    """
    import mlx.core as mx
    return mx.power(x, y)