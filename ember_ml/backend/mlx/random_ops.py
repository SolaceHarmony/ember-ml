"""
MLX implementation of random operations.

This module provides MLX implementations of random operations.
"""

import mlx.core as mx
from typing import Union, Sequence, Optional, Tuple, Any, List

# Type aliases
ArrayLike = Union[mx.array, float, int, list, tuple]
Shape = Union[int, Sequence[int]]
DType = Any

def random_normal(shape: Shape, mean: float = 0.0, stddev: float = 1.0,
                 dtype: DType = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array with random values from a normal distribution.
    
    Args:
        shape: Shape of the array
        mean: Mean of the normal distribution
        stddev: Standard deviation of the normal distribution
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array with random normal values
    """
    # Convert shape to a list
    if isinstance(shape, (list, tuple)):
        shape_list = list(shape)
    else:
        # If shape is an integer, convert it to a list with one element
        shape_list = [shape]
    
    # Use the correct signature for mx.random.normal
    return mx.random.normal(shape=shape_list, dtype=dtype, loc=mean, scale=stddev)

def random_uniform(shape: Shape, minval: float = 0.0, maxval: float = 1.0,
                  dtype: DType = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array with random values from a uniform distribution.
    
    Args:
        shape: Shape of the array
        minval: Minimum value
        maxval: Maximum value
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array with random uniform values
    """
    # Convert shape to a list
    if isinstance(shape, (list, tuple)):
        shape_list = list(shape)
    else:
        # If shape is an integer, convert it to a list with one element
        shape_list = [shape]
    
    # Use the correct signature for mx.random.uniform
    return mx.random.uniform(low=minval, high=maxval, shape=shape_list, dtype=dtype)

def random_binomial(shape: Shape, p: float = 0.5,
                   dtype: DType = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array with random values from a binomial distribution.
    
    Args:
        shape: Shape of the array
        p: Probability of success
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array with random binomial values
    """
    # Convert shape to a list
    if isinstance(shape, (list, tuple)):
        shape_list = list(shape)
    else:
        # If shape is an integer, convert it to a list with one element
        shape_list = [shape]
    
    return mx.array(mx.random.bernoulli(p, shape_list), dtype=dtype)

def random_permutation(x: Union[int, ArrayLike]) -> mx.array:
    """
    Randomly permute a sequence or return a permuted range.
    
    Args:
        x: If x is an integer, randomly permute np.arange(x).
           If x is an array, make a copy and shuffle the elements randomly.
        
    Returns:
        Permuted array
    """
    if isinstance(x, int):
        # Create a range and permute it
        arr = mx.arange(x)
        indices = mx.random.permutation(x)
        return arr[indices]
    else:
        # Convert to MLX array if needed
        arr = mx.array(x)
        indices = mx.random.permutation(arr.shape[0])
        return arr[indices]

def set_random_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    mx.random.seed(seed)


class MLXRandomOps:
    """MLX implementation of random operations."""
    
    def __init__(self):
        """Initialize MLX random operations."""
        self._current_seed = None
    
    def random_normal(self, shape, mean=0.0, stddev=1.0, dtype=None, device=None):
        """Create a tensor with random values from a normal distribution."""
        return random_normal(shape, mean=mean, stddev=stddev, dtype=dtype, device=device)
    
    def random_uniform(self, shape, minval=0.0, maxval=1.0, dtype=None, device=None):
        """Create a tensor with random values from a uniform distribution."""
        return random_uniform(shape, minval=minval, maxval=maxval, dtype=dtype, device=device)
    
    def random_binomial(self, shape, p=0.5, dtype=None, device=None):
        """Create a tensor with random values from a binomial distribution."""
        return random_binomial(shape, p=p, dtype=dtype, device=device)
    
    def random_permutation(self, x):
        """Randomly permute a sequence or return a permuted range."""
        return random_permutation(x)
    
    def set_seed(self, seed):
        """Set the random seed for reproducibility."""
        self._current_seed = seed
        set_random_seed(seed)
    
    def get_seed(self):
        """Get the current random seed."""
        return self._current_seed