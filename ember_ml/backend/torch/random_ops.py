"""
PyTorch random operations for EmberHarmony.

This module provides PyTorch implementations of random operations.
"""

import torch
from typing import Optional, Union, Sequence, Any, List

# Import from tensor_ops
from ember_ml.backend.torch.tensor_ops import convert_to_tensor, ArrayLike, Shape, DType
from ember_ml.backend.torch.config import DEFAULT_DEVICE


def random_normal(shape: Shape, mean: float = 0.0, stddev: float = 1.0, 
                 dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor with random values from a normal distribution.
    
    Args:
        shape: Shape of the tensor
        mean: Mean of the normal distribution
        stddev: Standard deviation of the normal distribution
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor with random normal values
    """
    if device is None:
        device = DEFAULT_DEVICE
    return torch.normal(mean, stddev, size=shape, dtype=dtype, device=device)


def random_uniform(shape: Shape, minval: float = 0.0, maxval: float = 1.0,
                   dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor with random values from a uniform distribution.
    
    Args:
        shape: Shape of the tensor
        minval: Minimum value
        maxval: Maximum value
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor with random uniform values
    """
    if device is None:
        device = DEFAULT_DEVICE
        
    # Generate random values between 0 and 1
    rand_tensor = torch.rand(shape, dtype=dtype, device=device)
    
    # Calculate the range using torch.subtract instead of direct subtraction
    maxval_tensor = convert_to_tensor(maxval)
    minval_tensor = convert_to_tensor(minval)
    range_tensor = torch.subtract(maxval_tensor, minval_tensor)
    
    # Scale the random values to the desired range
    scaled_tensor = torch.multiply(rand_tensor, range_tensor)
    
    # Shift the values to start at minval
    min_tensor = convert_to_tensor(minval)
    result_tensor = torch.add(scaled_tensor, min_tensor)
    
    return result_tensor


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_random_seed(seed: int) -> None:
    """
    Alias for set_seed for backward compatibility.
    
    Args:
        seed: Random seed
    """
    set_seed(seed)


def random_binomial(shape: Shape, p: float = 0.5, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Generate random values from a binomial distribution.
    
    Args:
        shape: Shape of the output tensor
        p: Probability of success
        dtype: Optional data type
        device: Optional device
        
    Returns:
        PyTorch tensor with random values from a binomial distribution
    """
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
    
    if device is None:
        device = DEFAULT_DEVICE
        
    # Generate random values
    result = torch.bernoulli(torch.full(shape, p, device=device))
    
    # Convert to the specified data type
    if dtype is not None:
        result = result.to(dtype=dtype)
        
    return result


def random_permutation(x: Union[int, ArrayLike], dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Generate a random permutation.
    
    Args:
        x: If an integer, randomly permute integers from 0 to x-1.
           If a tensor, randomly permute along the first axis.
        dtype: Optional data type
        device: Optional device
        
    Returns:
        Tensor with random permutation
    """
    if device is None:
        device = DEFAULT_DEVICE
        
    if isinstance(x, int):
        # Generate random permutation using PyTorch's randperm function
        perm = torch.randperm(x, device=device)
        
        # Convert to the specified data type
        if dtype is not None:
            perm = perm.to(dtype=dtype)
            
        return perm
    else:
        # If x is a tensor, permute along the first axis
        x_tensor = convert_to_tensor(x)
        
        # Get the shape of the tensor
        shape = x_tensor.shape
        
        # If the tensor is empty or has only one element, return it as is
        if shape[0] <= 1:
            return x_tensor
        
        # Generate random indices
        indices = torch.randperm(shape[0], device=x_tensor.device)
        
        # Gather along the first dimension
        return x_tensor[indices]


def random_categorical(logits: Any, num_samples: int, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Draw samples from a categorical distribution.
    
    Args:
        logits: 2D tensor with unnormalized log probabilities
        num_samples: Number of samples to draw
        dtype: Optional data type
        device: Optional device
        
    Returns:
        Tensor with random categorical values
    """
    # Convert to PyTorch tensor if needed
    logits_tensor = convert_to_tensor(logits)
    
    if device is None:
        device = DEFAULT_DEVICE
    else:
        # Move to the specified device
        logits_tensor = logits_tensor.to(device=device)
        
    # Convert to probabilities
    probs = torch.softmax(logits_tensor, dim=-1)
    
    # Sample from the categorical distribution
    samples = torch.multinomial(probs, num_samples, replacement=True)
    
    # Convert to the specified data type
    if dtype is not None:
        samples = samples.to(dtype=dtype)
        
    return samples


def random_exponential(shape: Shape, scale: float = 1.0, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Generate random values from an exponential distribution.
    
    Args:
        shape: Shape of the output tensor
        scale: Scale parameter
        dtype: Optional data type
        device: Optional device
        
    Returns:
        PyTorch tensor with random values from an exponential distribution
    """
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
    
    if device is None:
        device = DEFAULT_DEVICE
        
    # Generate uniform random values
    u = torch.rand(shape, dtype=dtype, device=device)
    
    # Transform to exponential distribution
    # Exponential distribution: f(x) = (1/scale) * exp(-x/scale)
    # Can be sampled by taking -scale * ln(U) where U is uniform(0,1)
    # Avoid log(0) by using 1-u instead of u
    # Replace direct operators with torch functions
    one_tensor = convert_to_tensor(1.0, device=device)
    one_minus_u = torch.subtract(one_tensor, u)
    log_result = torch.log(one_minus_u)
    scale_tensor = convert_to_tensor(scale, device=device)
    scaled_result = torch.multiply(scale_tensor, log_result)
    result = torch.negative(scaled_result)
    
    return result


def random_gamma(shape: Shape, alpha: float = 1.0, beta: float = 1.0, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Generate random values from a gamma distribution.
    
    Args:
        shape: Shape of the output tensor
        alpha: Shape parameter
        beta: Scale parameter
        dtype: Optional data type
        device: Optional device
        
    Returns:
        PyTorch tensor with random values from a gamma distribution
    """
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
    
    if device is None:
        device = DEFAULT_DEVICE
        
    # PyTorch's gamma distribution uses rate parameter (1/beta)
    # Replace direct division with torch.divide
    one_tensor = convert_to_tensor(1.0, device=device)
    beta_tensor = convert_to_tensor(beta, device=device)
    rate = torch.divide(one_tensor, beta_tensor)
    
    # Create a gamma distribution
    gamma_dist = torch.distributions.gamma.Gamma(alpha, rate)
    
    # Sample from the distribution
    result = gamma_dist.sample(shape)
    
    # Convert to the specified data type
    if dtype is not None:
        result = result.to(dtype=dtype)
        
    # Move to the specified device
    if device is not None:
        result = result.to(device=device)
        
    return result


def random_poisson(shape: Shape, lam: float = 1.0, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Generate random values from a Poisson distribution.
    
    Args:
        shape: Shape of the output tensor
        lam: Rate parameter
        dtype: Optional data type
        device: Optional device
        
    Returns:
        PyTorch tensor with random values from a Poisson distribution
    """
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
    
    if device is None:
        device = DEFAULT_DEVICE
        
    # Create a tensor filled with the rate parameter
    rate_tensor = torch.full(shape, lam, dtype=dtype, device=device)
    
    # Sample from the Poisson distribution
    result = torch.poisson(rate_tensor)
    
    return result


def shuffle(x: Any) -> torch.Tensor:
    """
    Randomly shuffle a tensor along the first dimension.
    
    Args:
        x: Input tensor
        
    Returns:
        Shuffled tensor
    """
    x_tensor = convert_to_tensor(x)
    
    # Get the shape of the tensor
    shape = x_tensor.shape
    
    # If the tensor is empty or has only one element, return it as is
    if shape[0] <= 1:
        return x_tensor
    
    # Generate random indices
    indices = torch.randperm(shape[0], device=x_tensor.device)
    
    # Gather along the first dimension
    return x_tensor[indices]


class TorchRandomOps:
    """PyTorch implementation of random operations."""
    
    def random_normal(self, shape, mean=0.0, stddev=1.0, dtype=None, device=None):
        """Generate random values from a normal distribution."""
        return random_normal(shape, mean=mean, stddev=stddev, dtype=dtype, device=device)
    
    def random_uniform(self, shape, minval=0.0, maxval=1.0, dtype=None, device=None):
        """Generate random values from a uniform distribution."""
        return random_uniform(shape, minval=minval, maxval=maxval, dtype=dtype, device=device)
    
    def random_binomial(self, shape, p=0.5, dtype=None, device=None):
        """Generate random values from a binomial distribution."""
        return random_binomial(shape, p=p, dtype=dtype, device=device)
    
    def random_permutation(self, x):
        """Generate a random permutation."""
        return random_permutation(x)
    
    def set_random_seed(self, seed):
        """Set the random seed for reproducibility."""
        return set_random_seed(seed)