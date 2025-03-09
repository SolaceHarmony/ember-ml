"""
NumPy random operations for EmberHarmony.

This module provides NumPy implementations of random operations.
"""

import numpy as np
from typing import Optional, Union, Any, Sequence

# Type aliases
ArrayLike = Union[np.ndarray, float, int, list, tuple]
Shape = Union[int, Sequence[int]]
DType = Union[np.dtype, str, None]

# Import from config
from ember_ml.backend.numpy.config import _current_seed


def random_normal(shape: Shape, mean: float = 0.0, stddev: float = 1.0,
                 dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array with random values from a normal distribution.
    
    Args:
        shape: Shape of the array
        mean: Mean of the normal distribution
        stddev: Standard deviation of the normal distribution
        dtype: Optional data type (default: float32)
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array with random normal values
    """
    return np.random.normal(mean, stddev, size=shape).astype(dtype)


def random_uniform(shape: Shape, minval: float = 0.0, maxval: float = 1.0,
                  dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array with random values from a uniform distribution.
    
    Args:
        shape: Shape of the array
        minval: Minimum value
        maxval: Maximum value
        dtype: Optional data type (default: float32)
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array with random uniform values
    """
    return np.random.uniform(minval, maxval, size=shape).astype(dtype)


def random_binomial(shape: Shape, p: float = 0.5,
                   dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array with random values from a binomial distribution.
    
    Args:
        shape: Shape of the array
        p: Probability of success
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array with random binomial values
    """
    return np.random.binomial(1, p, size=shape).astype(dtype)


def random_exponential(shape: Shape, scale: float = 1.0,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array with random values from an exponential distribution.
    
    Args:
        shape: Shape of the array
        scale: Scale parameter
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array with random exponential values
    """
    return np.random.exponential(scale, size=shape).astype(dtype)


def random_gamma(shape: Shape, alpha: float = 1.0, beta: float = 1.0,
                dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array with random values from a gamma distribution.
    
    Args:
        shape: Shape of the array
        alpha: Shape parameter
        beta: Scale parameter
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array with random gamma values
    """
    return np.random.gamma(alpha, beta, size=shape).astype(dtype)


def random_poisson(shape: Shape, lam: float = 1.0,
                  dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array with random values from a Poisson distribution.
    
    Args:
        shape: Shape of the array
        lam: Rate parameter
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array with random Poisson values
    """
    return np.random.poisson(lam, size=shape).astype(dtype)


def random_categorical(logits: Any, num_samples: int,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Draw samples from a categorical distribution.
    
    Args:
        logits: 2D tensor with unnormalized log probabilities
        num_samples: Number of samples to draw
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array with random categorical values
    """
    # Convert logits to probabilities
    logits = np.asarray(logits)
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(np.subtract(logits, max_logits))
    sum_exp_logits = np.sum(exp_logits, axis=-1, keepdims=True)
    probs = np.divide(exp_logits, sum_exp_logits)
    
    # Draw samples
    samples = np.zeros((probs.shape[0], num_samples), dtype=np.int64)
    for i in range(probs.shape[0]):
        samples[i] = np.random.choice(probs.shape[1], size=num_samples, p=probs[i])
    
    if dtype is not None:
        samples = samples.astype(dtype)
    
    return samples


def random_permutation(n: int, dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Randomly permute a sequence of integers from 0 to n-1.
    
    Args:
        n: Upper bound for the range of integers to permute
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array with a random permutation of integers from 0 to n-1
    """
    perm = np.random.permutation(n)
    if dtype is not None:
        perm = perm.astype(dtype)
    return perm


def shuffle(x: Any) -> np.ndarray:
    """
    Randomly shuffle a NumPy array along its first dimension.
    
    Args:
        x: Input array
        
    Returns:
        Shuffled NumPy array
    """
    x = np.asarray(x)
    indices = np.random.permutation(len(x))
    return x[indices]


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    global _current_seed
    _current_seed = seed  # type: ignore
    np.random.seed(seed)


def set_random_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility (alias for set_seed).
    
    Args:
        seed: Random seed
    """
    set_seed(seed)


def get_seed() -> Optional[int]:
    """
    Get the current random seed.
    
    Returns:
        Current random seed or None if not set
    """
    return _current_seed


class NumpyRandomOps:
    """NumPy implementation of random operations."""
    
    def random_normal(self, shape, mean=0.0, stddev=1.0, dtype=None, device=None):
        """Create a tensor with random values from a normal distribution."""
        return random_normal(shape, mean=mean, stddev=stddev, dtype=dtype, device=device)
    
    def random_uniform(self, shape, minval=0.0, maxval=1.0, dtype=None, device=None):
        """Create a tensor with random values from a uniform distribution."""
        return random_uniform(shape, minval=minval, maxval=maxval, dtype=dtype, device=device)
    
    def random_binomial(self, shape, p=0.5, dtype=None, device=None):
        """Create a tensor with random values from a binomial distribution."""
        return random_binomial(shape, p=p, dtype=dtype, device=device)
    
    def random_exponential(self, shape, scale=1.0, dtype=None, device=None):
        """Create a tensor with random values from an exponential distribution."""
        return random_exponential(shape, scale=scale, dtype=dtype, device=device)
    
    def random_gamma(self, shape, alpha=1.0, beta=1.0, dtype=None, device=None):
        """Create a tensor with random values from a gamma distribution."""
        return random_gamma(shape, alpha=alpha, beta=beta, dtype=dtype, device=device)
    
    def random_poisson(self, shape, lam=1.0, dtype=None, device=None):
        """Create a tensor with random values from a Poisson distribution."""
        return random_poisson(shape, lam=lam, dtype=dtype, device=device)
    
    def random_categorical(self, logits, num_samples, dtype=None, device=None):
        """Draw samples from a categorical distribution."""
        return random_categorical(logits, num_samples, dtype=dtype, device=device)
    
    def random_permutation(self, n, dtype=None, device=None):
        """Randomly permute a sequence of integers from 0 to n-1."""
        return random_permutation(n, dtype=dtype, device=device)
    
    def shuffle(self, x):
        """Randomly shuffle a tensor along its first dimension."""
        return shuffle(x)
    
    def set_seed(self, seed):
        """Set the random seed for reproducibility."""
        set_seed(seed)
    
    def set_random_seed(self, seed):
        """Set the random seed for reproducibility (alias for set_seed)."""
        set_random_seed(seed)
    
    def get_seed(self):
        """Get the current random seed."""
        return get_seed()