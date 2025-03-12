"""
NumPy random operations for ember_ml.

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


def get_seed() -> Optional[int]:
    """
    Get the current random seed.
    
    Returns:
        Current random seed or None if not set
    """
    return _current_seed


def random_lognormal(shape: Shape, mean: float = 0.0, stddev: float = 1.0,
                     dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
    """
    Generate random values from a log-normal distribution.
    
    Args:
        shape: Shape of the output array
        mean: Mean of the underlying normal distribution
        stddev: Standard deviation of the underlying normal distribution
        dtype: Optional data type
        device: Optional device
    
    Returns:
        NumPy array with random values from a log-normal distribution
    """
    normal_samples = random_normal(shape, mean=mean, stddev=stddev, dtype=dtype, device=device)
    lognormal_samples = np.exp(normal_samples)
    return lognormal_samples


def random_multinomial(n: int, pvals: ArrayLike, size: Optional[Shape] = None,
                       dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
    """
    Generate random values from a multinomial distribution.
    
    Args:
        n: Number of trials
        pvals: Probabilities of each outcome
        size: Shape of the output array
        dtype: Optional data type
        device: Optional device
    
    Returns:
        NumPy array with random values from a multinomial distribution
    """
    pvals_tensor = np.asarray(pvals)
    multinomial_samples = np.random.multinomial(n, pvals_tensor, size=size)
    if dtype is not None:
        multinomial_samples = multinomial_samples.astype(dtype)
    return multinomial_samples


def random_geometric(p: float, size: Optional[Shape] = None,
                     dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
    """
    Generate random values from a geometric distribution.
    
    Args:
        p: Probability of success
        size: Shape of the output array
        dtype: Optional data type
        device: Optional device
    
    Returns:
        NumPy array with random values from a geometric distribution
    """
    u = np.random.uniform(size=size)
    log_result = np.log(u)
    log_one_minus_p = np.log(1.0 - p)
    geometric_samples = np.floor(log_result / log_one_minus_p) + 1
    if dtype is not None:
        geometric_samples = geometric_samples.astype(dtype)
    return geometric_samples


class NumpyRandomOps:
    """NumPy implementation of random operations."""
    
    def random_normal(self, shape: Shape, mean: float = 0.0, stddev: float = 1.0,
                      dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
        """Create a tensor with random values from a normal distribution."""
        return random_normal(shape, mean=mean, stddev=stddev, dtype=dtype, device=device)
    
    def random_uniform(self, shape: Shape, minval: float = 0.0, maxval: float = 1.0,
                       dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
        """Create a tensor with random values from a uniform distribution."""
        return random_uniform(shape, minval=minval, maxval=maxval, dtype=dtype, device=device)
    
    def random_binomial(self, shape: Shape, p: float = 0.5,
                        dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
        """Create a tensor with random values from a binomial distribution."""
        return random_binomial(shape, p=p, dtype=dtype, device=device)
    
    def random_exponential(self, shape: Shape, scale: float = 1.0,
                           dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
        """Create a tensor with random values from an exponential distribution."""
        return random_exponential(shape, scale=scale, dtype=dtype, device=device)
    
    def random_gamma(self, shape: Shape, alpha: float = 1.0, beta: float = 1.0,
                     dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
        """Create a tensor with random values from a gamma distribution."""
        return random_gamma(shape, alpha=alpha, beta=beta, dtype=dtype, device=device)
    
    def random_poisson(self, shape: Shape, lam: float = 1.0,
                       dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
        """Create a tensor with random values from a Poisson distribution."""
        return random_poisson(shape, lam=lam, dtype=dtype, device=device)
    
    def random_categorical(self, logits: Any, num_samples: int,
                           dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
        """Draw samples from a categorical distribution."""
        return random_categorical(logits, num_samples, dtype=dtype, device=device)
    
    def random_permutation(self, n: int, dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
        """Randomly permute a sequence of integers from 0 to n-1."""
        return random_permutation(n, dtype=dtype, device=device)
    
    def shuffle(self, x: Any) -> np.ndarray:
        """Randomly shuffle a tensor along the first dimension."""
        return shuffle(x)
    
    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        set_seed(seed)
    
    def get_seed(self) -> Optional[int]:
        """Get the current random seed."""
        return get_seed()
    
    def random_lognormal(self, shape: Shape, mean: float = 0.0, stddev: float = 1.0,
                         dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
        """Generate random values from a log-normal distribution."""
        return random_lognormal(shape, mean=mean, stddev=stddev, dtype=dtype, device=device)
    
    def random_multinomial(self, n: int, pvals: ArrayLike, size: Optional[Shape] = None,
                           dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
        """Generate random values from a multinomial distribution."""
        return random_multinomial(n, pvals, size=size, dtype=dtype, device=device)
    
    def random_geometric(self, p: float, size: Optional[Shape] = None,
                         dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
        """Generate random values from a geometric distribution."""
        return random_geometric(p, size=size, dtype=dtype, device=device)
