"""PyTorch tensor random operations."""

import torch
from typing import Union, Optional, Sequence, Any, List, Tuple
from torch import Size

from ember_ml.backend.torch.tensor.dtype import TorchDType
from ember_ml.backend.torch.tensor.ops.utility import convert_to_tensor

# Type aliases
Shape = Sequence[int]
TensorLike = Any
DType = Any

def random_normal(shape: Shape, mean: float = 0.0, stddev: float = 1.0, 
                  dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor with random values from a normal distribution.
    
    Args:
        shape: The shape of the tensor
        mean: The mean of the normal distribution
        stddev: The standard deviation of the normal distribution
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor with random values from a normal distribution
    """
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    if isinstance(shape, int):
        shape = (shape,)
    
    # Create a tensor with random values from a normal distribution
    tensor = torch.normal(mean=mean, std=stddev, size=shape, device=device)
    
    # Cast to the specified dtype if needed
    if torch_dtype is not None:
        tensor = tensor.to(torch_dtype)
    
    return tensor

def random_uniform(shape: Shape, minval: float = 0.0, maxval: float = 1.0,
                   dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a torch array with random values from a uniform distribution.
    
    Args:
        shape: Shape of the array
        minval: Minimum value
        maxval: Maximum value
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Torch array with random uniform values
    """
    # Convert shape to a list
    if isinstance(shape, (list, tuple)):
        shape_list = list(shape)
    else:
        # If shape is an integer, convert it to a list with one element
        shape_list = [shape]
    
    # Handle string dtype values
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    # Generate random values between 0 and 1
    rand_tensor = torch.rand(shape_list, device=device)
    
    # Scale to the desired range
    range_tensor = torch.tensor(maxval, device=device)
    min_tensor = torch.tensor(minval, device=device)
    range_diff = torch.sub(range_tensor, min_tensor)
    
    # Scale and shift
    result = torch.add(torch.mul(rand_tensor, range_diff), min_tensor)
    
    # Cast to the specified dtype if needed
    if torch_dtype is not None:
        result = result.to(torch_dtype)
        
    return result

def random_binomial(shape: Shape, p: float = 0.5,
                    dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor with random values from a binomial distribution.
    
    Args:
        shape: Shape of the tensor
        p: Probability of success
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor with random values from a binomial distribution
    """
    # Handle string dtype values
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
    
    if device is None:
        from ember_ml.backend.torch.config import DEFAULT_DEVICE
        device = DEFAULT_DEVICE
    
    # Generate random values
    result = torch.bernoulli(torch.full(shape, p, device=device))
    
    # Convert to the specified data type
    if torch_dtype is not None:
        result = result.to(torch_dtype)
    
    return result

def random_gamma(shape: Shape, alpha: float = 1.0, beta: float = 1.0,
                 dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Generate random values from a gamma distribution.
    
    Args:
        shape: Shape of the output tensor
        alpha: Shape parameter
        beta: Scale parameter
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        PyTorch tensor with random values from a gamma distribution
    """
    # Handle string dtype values
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
    
    if device is None:
        from ember_ml.backend.torch.config import DEFAULT_DEVICE
        device = DEFAULT_DEVICE
    
    # PyTorch's gamma distribution uses rate parameter (1/beta)
    one_tensor = torch.tensor(1.0, device=device)
    beta_tensor = torch.tensor(beta, device=device)
    rate = torch.divide(one_tensor, beta_tensor)
    
    # Create a gamma distribution
    gamma_dist = torch.distributions.gamma.Gamma(alpha, rate)
    
    # Sample from the distribution
    # Convert shape to Size object for PyTorch
    result = gamma_dist.sample(Size(shape))
    
    # Convert to the specified data type
    if torch_dtype is not None:
        result = result.to(torch_dtype)
    
    # Move to the specified device
    if device is not None:
        result = result.to(device=device)
    
    return result

def random_exponential(shape: Shape, scale: float = 1.0,
                       dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Generate random values from an exponential distribution.
    
    Args:
        shape: Shape of the output tensor
        scale: Scale parameter
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        PyTorch tensor with random values from an exponential distribution
    """
    # Handle string dtype values
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
    
    if device is None:
        from ember_ml.backend.torch.config import DEFAULT_DEVICE
        device = DEFAULT_DEVICE
    
    # Generate uniform random values
    u = torch.rand(shape, device=device)
    
    # Transform to exponential distribution
    # Exponential distribution: f(x) = (1/scale) * exp(-x/scale)
    # Can be sampled by taking -scale * ln(U) where U is uniform(0,1)
    # Avoid log(0) by using 1-u instead of u
    one_tensor = torch.tensor(1.0, device=device)
    one_minus_u = torch.subtract(one_tensor, u)
    log_result = torch.log(one_minus_u)
    scale_tensor = torch.tensor(scale, device=device)
    scaled_result = torch.multiply(scale_tensor, log_result)
    result = torch.negative(scaled_result)
    
    # Convert to the specified data type
    if torch_dtype is not None:
        result = result.to(torch_dtype)
    
    return result

def random_poisson(shape: Shape, lam: float = 1.0,
                   dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Generate random values from a Poisson distribution.
    
    Args:
        shape: Shape of the output tensor
        lam: Rate parameter
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        PyTorch tensor with random values from a Poisson distribution
    """
    # Handle string dtype values
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
    
    if device is None:
        from ember_ml.backend.torch.config import DEFAULT_DEVICE
        device = DEFAULT_DEVICE
    
    # Create a tensor filled with the rate parameter
    rate_tensor = torch.full(shape, lam, device=device)
    
    # Sample from the Poisson distribution
    result = torch.poisson(rate_tensor)
    
    # Convert to the specified data type
    if torch_dtype is not None:
        result = result.to(torch_dtype)
    
    return result

def random_categorical(data: TensorLike, num_samples: int,
                       dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Draw samples from a categorical distribution.
    
    Args:
        data: 2D tensor with unnormalized log probabilities
        num_samples: Number of samples to draw
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor with random categorical values
    """
    # Handle string dtype values
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    # Convert to PyTorch tensor if needed
    logits_tensor = convert_to_tensor(data)
    
    if device is None:
        from ember_ml.backend.torch.config import DEFAULT_DEVICE
        device = DEFAULT_DEVICE
    else:
        # Move to the specified device
        logits_tensor = logits_tensor.to(device=device)
    
    # Convert to probabilities
    probs = torch.softmax(logits_tensor, dim=-1)
    
    # Sample from the categorical distribution
    samples = torch.multinomial(probs, num_samples, replacement=True)
    
    # Convert to the specified data type
    if torch_dtype is not None:
        samples = samples.to(torch_dtype)
    
    return samples

def random_permutation(data: Union[int, TensorLike],
                       dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Generate a random permutation.
    
    Args:
        data: If an integer, randomly permute integers from 0 to data-1.
              If a tensor, randomly permute along the first axis.
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor with random permutation
    """
    if device is None:
        from ember_ml.backend.torch.config import DEFAULT_DEVICE
        device = DEFAULT_DEVICE
    
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    if isinstance(data, int):
        # Generate random permutation using PyTorch's randperm function
        perm = torch.randperm(data, device=device)
        
        # Convert to the specified data type
        if torch_dtype is not None:
            perm = perm.to(torch_dtype)
        
        return perm
    else:
        # If data is a tensor, permute along the first axis
        tensor = convert_to_tensor(data)
        
        # Get the shape of the tensor
        shape = tensor.shape
        
        # If the tensor is empty or has only one element, return it as is
        if shape[0] <= 1:
            return tensor
        
        # Generate random indices
        indices = torch.randperm(shape[0], device=tensor.device)
        
        # Gather along the first dimension
        return tensor[indices]

def shuffle(data: TensorLike) -> torch.Tensor:
    """
    Randomly shuffle a tensor along the first dimension.
    
    Args:
        data: Input tensor
        
    Returns:
        Shuffled tensor
    """
    tensor = convert_to_tensor(data)
    
    # Get the shape of the tensor
    shape = tensor.shape
    
    # If the tensor is empty or has only one element, return it as is
    if shape[0] <= 1:
        return tensor
    
    # Generate random indices
    indices = torch.randperm(shape[0], device=tensor.device)
    
    # Gather along the first dimension
    return tensor[indices]

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_seed() -> Optional[int]:
    """
    Get the current random seed.
    
    Returns:
        Current random seed or None if not set
    """
    # PyTorch doesn't provide a way to get the current seed
    return None