"""
Implementation of random operations for different backends.

This file provides implementations of random operations for NumPy, PyTorch, and MLX backends
with consistent seeding behavior across all backends.
"""

def numpy_random_normal(shape, mean=0.0, stddev=1.0, seed=None):
    """
    Generate random values from a normal distribution using NumPy backend.
    
    Parameters
    ----------
    shape : tuple of ints
        Shape of the output tensor.
    mean : float or tensor, optional
        Mean of the normal distribution. Default: 0.0.
    stddev : float or tensor, optional
        Standard deviation of the normal distribution. Default: 1.0.
    seed : int, optional
        Random seed. Default: None.
    
    Returns
    -------
    tensor
        Tensor of random values from a normal distribution.
    """
    import numpy as np
    if seed is not None:
        rng = np.random.RandomState(seed)
        return rng.normal(mean, stddev, shape)
    else:
        return np.random.normal(mean, stddev, shape)

def torch_random_normal(shape, mean=0.0, stddev=1.0, seed=None):
    """
    Generate random values from a normal distribution using PyTorch backend.
    
    Parameters
    ----------
    shape : tuple of ints
        Shape of the output tensor.
    mean : float or tensor, optional
        Mean of the normal distribution. Default: 0.0.
    stddev : float or tensor, optional
        Standard deviation of the normal distribution. Default: 1.0.
    seed : int, optional
        Random seed. Default: None.
    
    Returns
    -------
    tensor
        Tensor of random values from a normal distribution.
    """
    import torch
    if seed is not None:
        # Set the seed for reproducibility
        torch.manual_seed(seed)
    return torch.normal(mean, stddev, shape)

def mlx_random_normal(shape, mean=0.0, stddev=1.0, seed=None):
    """
    Generate random values from a normal distribution using MLX backend.
    
    Parameters
    ----------
    shape : tuple of ints
        Shape of the output tensor.
    mean : float or tensor, optional
        Mean of the normal distribution. Default: 0.0.
    stddev : float or tensor, optional
        Standard deviation of the normal distribution. Default: 1.0.
    seed : int, optional
        Random seed. Default: None.
    
    Returns
    -------
    tensor
        Tensor of random values from a normal distribution.
    """
    import mlx.core as mx
    key = None
    if seed is not None:
        key = mx.random.key(seed)
    return mx.random.normal(shape, mean=mean, std=stddev, key=key)

def numpy_random_uniform(shape, minval=0.0, maxval=1.0, seed=None):
    """
    Generate random values from a uniform distribution using NumPy backend.
    
    Parameters
    ----------
    shape : tuple of ints
        Shape of the output tensor.
    minval : float or tensor, optional
        Lower bound of the uniform distribution. Default: 0.0.
    maxval : float or tensor, optional
        Upper bound of the uniform distribution. Default: 1.0.
    seed : int, optional
        Random seed. Default: None.
    
    Returns
    -------
    tensor
        Tensor of random values from a uniform distribution.
    """
    import numpy as np
    if seed is not None:
        rng = np.random.RandomState(seed)
        return rng.uniform(minval, maxval, shape)
    else:
        return np.random.uniform(minval, maxval, shape)

def torch_random_uniform(shape, minval=0.0, maxval=1.0, seed=None):
    """
    Generate random values from a uniform distribution using PyTorch backend.
    
    Parameters
    ----------
    shape : tuple of ints
        Shape of the output tensor.
    minval : float or tensor, optional
        Lower bound of the uniform distribution. Default: 0.0.
    maxval : float or tensor, optional
        Upper bound of the uniform distribution. Default: 1.0.
    seed : int, optional
        Random seed. Default: None.
    
    Returns
    -------
    tensor
        Tensor of random values from a uniform distribution.
    """
    import torch
    if seed is not None:
        # Set the seed for reproducibility
        torch.manual_seed(seed)
    return torch.empty(shape).uniform_(minval, maxval)

def mlx_random_uniform(shape, minval=0.0, maxval=1.0, seed=None):
    """
    Generate random values from a uniform distribution using MLX backend.
    
    Parameters
    ----------
    shape : tuple of ints
        Shape of the output tensor.
    minval : float or tensor, optional
        Lower bound of the uniform distribution. Default: 0.0.
    maxval : float or tensor, optional
        Upper bound of the uniform distribution. Default: 1.0.
    seed : int, optional
        Random seed. Default: None.
    
    Returns
    -------
    tensor
        Tensor of random values from a uniform distribution.
    """
    import mlx.core as mx
    key = None
    if seed is not None:
        key = mx.random.key(seed)
    return mx.random.uniform(shape, minval=minval, maxval=maxval, key=key)