"""PyTorch tensor random operations."""

import torch
from typing import Union, Optional, Sequence, Any, List, Tuple

from ember_ml.backend.torch.tensor.dtype import TorchDType

# Type aliases
Shape = Union[int, Sequence[int]]

def random_normal(tensor_obj, shape, mean=0.0, stddev=1.0, dtype=None, device=None):
    """
    Create a tensor with random values from a normal distribution.
    
    Args:
        tensor_obj: TorchTensor instance
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

def random_uniform(tensor_obj, shape, minval=0.0, maxval=1.0, dtype=None, device=None):
    """
    Create a torch array with random values from a uniform distribution.
    
    Args:
        tensor_obj: TorchTensor instance
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
    if isinstance(dtype, str):
        dtype = TorchDType().from_dtype_str(dtype)
    # Handle EmberDtype objects
    elif dtype is not None and hasattr(dtype, 'name'):
        dtype = TorchDType().from_dtype_str(dtype.name)
    
    # Generate random values between 0 and 1
    rand_tensor = torch.rand(shape_list, device=device)
    
    # Scale to the desired range
    range_tensor = torch.tensor(maxval, device=device)
    min_tensor = torch.tensor(minval, device=device)
    range_diff = torch.sub(range_tensor, min_tensor)
    
    # Scale and shift
    result = torch.add(torch.mul(rand_tensor, range_diff), min_tensor)
    
    # Cast to the specified dtype if needed
    if dtype is not None:
        result = result.to(dtype)
        
    return result

def random_binomial(tensor_obj, shape, p=0.5, dtype=None, device=None):
    """
    Create a tensor with random values from a binomial distribution.
    
    Args:
        tensor_obj: TorchTensor instance
        shape: Shape of the tensor
        p: Probability of success
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor with random values from a binomial distribution
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        dtype = TorchDType().from_dtype_str(dtype)
    
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
    
    if device is None:
        from ember_ml.backend.torch.config import DEFAULT_DEVICE
        device = DEFAULT_DEVICE
    
    # Generate random values
    result = torch.bernoulli(torch.full(shape, p, device=device))
    
    # Convert to the specified data type
    if dtype is not None:
        result = result.to(dtype)
    
    return result

def random_gamma(tensor_obj, shape, alpha=1.0, beta=1.0, dtype=None, device=None):
    """
    Generate random values from a gamma distribution.
    
    Args:
        tensor_obj: TorchTensor instance
        shape: Shape of the output tensor
        alpha: Shape parameter
        beta: Scale parameter
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        PyTorch tensor with random values from a gamma distribution
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        dtype = TorchDType().from_dtype_str(dtype)
    
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
    from torch import Size
    result = gamma_dist.sample(Size(shape))
    
    # Convert to the specified data type
    if dtype is not None:
        result = result.to(dtype)
    
    # Move to the specified device
    if device is not None:
        result = result.to(device=device)
    
    return result

def random_exponential(tensor_obj, shape, scale=1.0, dtype=None, device=None):
    """
    Generate random values from an exponential distribution.
    
    Args:
        tensor_obj: TorchTensor instance
        shape: Shape of the output tensor
        scale: Scale parameter
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        PyTorch tensor with random values from an exponential distribution
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        dtype = TorchDType().from_dtype_str(dtype)
    
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
    if dtype is not None:
        result = result.to(dtype)
    
    return result

def random_poisson(tensor_obj, shape, lam=1.0, dtype=None, device=None):
    """
    Generate random values from a Poisson distribution.
    
    Args:
        tensor_obj: TorchTensor instance
        shape: Shape of the output tensor
        lam: Rate parameter
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        PyTorch tensor with random values from a Poisson distribution
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        dtype = TorchDType().from_dtype_str(dtype)
    
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
    if dtype is not None:
        result = result.to(dtype)
    
    return result

def random_categorical(tensor_obj, logits, num_samples, dtype=None, device=None):
    """
    Draw samples from a categorical distribution.
    
    Args:
        tensor_obj: TorchTensor instance
        logits: 2D tensor with unnormalized log probabilities
        num_samples: Number of samples to draw
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor with random categorical values
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        dtype = TorchDType().from_dtype_str(dtype)
    
    # Convert to PyTorch tensor if needed
    logits_tensor = tensor_obj.convert_to_tensor(logits)
    
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
    if dtype is not None:
        samples = samples.to(dtype)
    
    return samples

def random_permutation(tensor_obj, x, dtype=None, device=None):
    """
    Generate a random permutation.
    
    Args:
        tensor_obj: TorchTensor instance
        x: If an integer, randomly permute integers from 0 to x-1.
           If a tensor, randomly permute along the first axis.
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor with random permutation
    """
    if device is None:
        from ember_ml.backend.torch.config import DEFAULT_DEVICE
        device = DEFAULT_DEVICE
    
    if isinstance(x, int):
        # Generate random permutation using PyTorch's randperm function
        perm = torch.randperm(x, device=device)
        
        # Convert to the specified data type
        if dtype is not None:
            if isinstance(dtype, str):
                dtype = TorchDType().from_dtype_str(dtype)
            perm = perm.to(dtype)
        
        return perm
    else:
        # If x is a tensor, permute along the first axis
        x_tensor = tensor_obj.convert_to_tensor(x)
        
        # Get the shape of the tensor
        shape = x_tensor.shape
        
        # If the tensor is empty or has only one element, return it as is
        if shape[0] <= 1:
            return x_tensor
        
        # Generate random indices
        indices = torch.randperm(shape[0], device=x_tensor.device)
        
        # Gather along the first dimension
        return x_tensor[indices]

def shuffle(tensor_obj, x):
    """
    Randomly shuffle a tensor along the first dimension.
    
    Args:
        tensor_obj: TorchTensor instance
        x: Input tensor
        
    Returns:
        Shuffled tensor
    """
    x_tensor = tensor_obj.convert_to_tensor(x)
    
    # Get the shape of the tensor
    shape = x_tensor.shape
    
    # If the tensor is empty or has only one element, return it as is
    if shape[0] <= 1:
        return x_tensor
    
    # Generate random indices
    indices = torch.randperm(shape[0], device=x_tensor.device)
    
    # Gather along the first dimension
    return x_tensor[indices]

def set_seed(tensor_obj, seed):
    """
    Set the random seed for reproducibility.
    
    Args:
        tensor_obj: TorchTensor instance
        seed: Random seed
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_seed(tensor_obj):
    """
    Get the current random seed.
    
    Args:
        tensor_obj: TorchTensor instance
    
    Returns:
        Current random seed or None if not set
    """
    # PyTorch doesn't provide a way to get the current seed
    return None