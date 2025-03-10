"""
MLX implementation of random operations.

This module provides MLX implementations of random operations.
"""

import mlx.core as mx
from typing import Union, Sequence, Optional, Any

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

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    mx.random.seed(seed)


def get_seed() -> Optional[int]:
    """
    Get the current random seed.
    
    Returns:
        Current random seed (None if not set)
    """
    # MLX doesn't provide a way to get the current seed
    return None


def random_categorical(logits: Any, num_samples: int,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Draw samples from a categorical distribution.
    
    Args:
        logits: 2D tensor with unnormalized log probabilities
        num_samples: Number of samples to draw
        dtype: Optional data type
        device: Optional device
    
    Returns:
        MLX array with random categorical values
    """
    # Convert to MLX array if needed
    from ember_ml.backend.mlx.tensor_ops import convert_to_tensor
    logits_tensor = convert_to_tensor(logits)
    
    # MLX's categorical function takes num_samples parameter
    result = mx.random.categorical(logits=logits_tensor, num_samples=num_samples)
    
    if dtype is not None:
        result = result.astype(dtype)
    
    return result


def random_exponential(shape: Shape, scale: float = 1.0,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Generate random values from an exponential distribution.
    
    Args:
        shape: Shape of the output array
        scale: Scale parameter
        dtype: Optional data type
        device: Optional device
    
    Returns:
        MLX array with random values from an exponential distribution
    """
    # Convert shape to sequence if it's an int
    if isinstance(shape, int):
        shape = (shape,)
    
    # Generate uniform random values
    u = mx.random.uniform(shape=shape)
    
    # Transform to exponential distribution
    # Exponential distribution: f(x) = (1/scale) * exp(-x/scale)
    # Can be sampled by taking -scale * ln(U) where U is uniform(0,1)
    # Avoid log(0) by using 1-u instead of u
    scale_tensor = mx.array(scale)
    result = mx.multiply(mx.negative(scale_tensor), mx.log(mx.subtract(mx.array(1.0), u)))
    
    if dtype is not None:
        result = result.astype(dtype)
    
    return result


def random_gamma(shape: Shape, alpha: float = 1.0, beta: float = 1.0,
                dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Generate random values from a gamma distribution.
    
    Args:
        shape: Shape of the output array
        alpha: Shape parameter
        beta: Scale parameter
        dtype: Optional data type
        device: Optional device
    
    Returns:
        MLX array with random values from a gamma distribution
    """
    # Convert shape to sequence if it's an int
    if isinstance(shape, int):
        shape = (shape,)
    
    if alpha <= 0:
        raise ValueError("Alpha parameter must be positive")
    
    # For alpha = 1, gamma is equivalent to exponential
    if alpha == 1.0:
        return random_exponential(shape, scale=beta, dtype=dtype, device=device)
    
    # For integer alpha, we can use the sum of exponentials
    if isinstance(alpha, int) and alpha > 1:
        result = mx.zeros(shape)
        for _ in range(alpha):
            result = mx.add(result, random_exponential(shape, scale=beta, dtype=None, device=device))
        
        if dtype is not None:
            result = result.astype(dtype)
        
        return result
    
    # For non-integer alpha, we use the Marsaglia and Tsang method
    # This is a simplified version that works for alpha > 1
    # For alpha < 1, we would need a more complex algorithm
    d = mx.subtract(mx.array(alpha), mx.divide(mx.array(1.0), mx.array(3.0)))
    c = mx.divide(mx.array(1.0), mx.sqrt(mx.multiply(mx.array(9.0), d)))
    
    result = mx.zeros(shape)
    # Use boolean type without the underscore
    valid_samples = mx.zeros(shape, dtype=bool)
    
    # Keep generating until all samples are valid
    while not mx.all(valid_samples):
        # Generate standard normal samples
        z = mx.random.normal(shape=shape)
        
        # Calculate v = (1 + c*z)^3
        v = mx.power(mx.add(mx.array(1.0), mx.multiply(c, z)), mx.array(3.0))
        
        # Filter out invalid samples (v <= 0)
        v_valid = mx.greater(v, mx.array(0.0))
        
        # Calculate log acceptance ratio
        u = mx.random.uniform(shape=shape)
        log_accept = mx.add(
            mx.add(
                mx.multiply(mx.array(0.5), mx.square(z)),
                d
            ),
            mx.subtract(
                mx.negative(mx.multiply(d, v)),
                mx.multiply(d, mx.log(v))
            )
        )
        
        # Accept samples where log(u) < log_accept
        accept = mx.less(mx.log(u), log_accept)
        
        # Update valid samples and result
        new_valid = mx.logical_and(
            mx.logical_and(v_valid, accept),
            mx.logical_not(valid_samples)
        )
        result = mx.where(new_valid, mx.multiply(mx.multiply(d, v), beta), result)
        valid_samples = mx.logical_or(valid_samples, new_valid)
    
    if dtype is not None:
        result = result.astype(dtype)
    
    return result


def random_poisson(shape: Shape, lam: float = 1.0,
                  dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Generate random values from a Poisson distribution.
    
    Args:
        shape: Shape of the output array
        lam: Rate parameter
        dtype: Optional data type
        device: Optional device
    
    Returns:
        MLX array with random values from a Poisson distribution
    """
    # Convert shape to sequence if it's an int
    if isinstance(shape, int):
        shape = (shape,)
    
    # Convert lambda to MLX array if it's a scalar
    if isinstance(lam, (int, float)):
        lam_array = mx.full(shape, lam)
    else:
        from ember_ml.backend.mlx.tensor_ops import convert_to_tensor
        lam_array = convert_to_tensor(lam)
    
    # Initialize counts and time accumulators
    counts = mx.zeros(shape, dtype=mx.int32)
    times = mx.zeros(shape)
    
    # Generate exponential waiting times until exceeding 1.0
    # This is based on the fact that Poisson process events have
    # exponentially distributed inter-arrival times
    while not mx.all(mx.greater_equal(times, mx.array(1.0))):
        # Generate exponential random variables with rate lambda
        exp_samples = mx.divide(
            mx.negative(mx.log(mx.random.uniform(shape=shape))),
            lam_array
        )
        # Add to accumulated times
        new_times = mx.add(times, exp_samples)
        # Increment counts where we haven't exceeded 1.0 yet
        counts = mx.where(
            mx.less(new_times, mx.array(1.0)),
            mx.add(counts, mx.array(1)),
            counts
        )
        times = new_times
    
    if dtype is not None and dtype != mx.int32:
        counts = counts.astype(dtype)
    
    return counts


def shuffle(x: Any) -> mx.array:
    """
    Randomly shuffle an MLX array along the first dimension.
    
    Args:
        x: Input array
    
    Returns:
        Shuffled MLX array
    """
    from ember_ml.backend.mlx.tensor_ops import convert_to_tensor
    x_tensor = convert_to_tensor(x)
    
    # Get the shape of the tensor
    shape = x_tensor.shape
    
    # If the tensor is empty or has only one element, return it as is
    if shape[0] <= 1:
        return x_tensor
    
    # Generate random indices
    indices = mx.random.permutation(shape[0])
    
    # Gather along the first dimension
    return mx.take(x_tensor, indices, axis=0)


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
    
    def random_gamma(self, shape, alpha=1.0, beta=1.0, dtype=None, device=None):
        """Generate random values from a gamma distribution."""
        return random_gamma(shape, alpha=alpha, beta=beta, dtype=dtype, device=device)
    
    def random_poisson(self, shape, lam=1.0, dtype=None, device=None):
        """Generate random values from a Poisson distribution."""
        return random_poisson(shape, lam=lam, dtype=dtype, device=device)
    
    def random_exponential(self, shape, scale=1.0, dtype=None, device=None):
        """Generate random values from an exponential distribution."""
        return random_exponential(shape, scale=scale, dtype=dtype, device=device)
    
    def random_categorical(self, logits, num_samples, dtype=None, device=None):
        """Draw samples from a categorical distribution."""
        return random_categorical(logits, num_samples, dtype=dtype, device=device)
    
    def random_permutation(self, x, dtype=None, device=None):
        """Randomly permute a sequence or return a permuted range."""
        return random_permutation(x)
    
    def shuffle(self, x):
        """Randomly shuffle a tensor along the first dimension."""
        return shuffle(x)
    
    def set_seed(self, seed):
        """Set the random seed for reproducibility."""
        self._current_seed = seed
        set_seed(seed)
    
    def get_seed(self):
        """Get the current random seed."""
        return self._current_seed