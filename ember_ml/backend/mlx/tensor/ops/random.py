"""MLX tensor random operations."""

import mlx.core as mx
from typing import Union, Optional, Sequence, Any, List, Tuple

from ember_ml.backend.mlx.types import Shape, TensorLike, DType
from ember_ml.backend.mlx.tensor.ops.utility import _create_new_tensor # Import helper

# Create single instances to reuse throughout the module
# DTypeHandler instance removed, logic moved to helper/local

def random_normal(shape: Shape, mean: float = 0.0, stddev: float = 1.0,
                 dtype: Optional[DType] = None, device: Optional[str] = None) -> 'mx.array':
    """
    Create a tensor with random values from a normal distribution.
    
    Args:
        shape: Shape of the tensor
        mean: Mean of the normal distribution
        stddev: Standard deviation of the normal distribution
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array with random normal values
    """
    # Use the helper function, passing mx.random.normal and its specific args
    return _create_new_tensor(mx.random.normal, dtype=dtype, device=device, shape=shape, loc=mean, scale=stddev)

def random_uniform(shape: Shape, minval: float = 0.0, maxval: float = 1.0,
                  dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create a tensor with random values from a uniform distribution.
    
    Args:
        shape: Shape of the tensor
        minval: Minimum value
        maxval: Maximum value
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array with random uniform values
    """
    # Use the helper function, passing mx.random.uniform and its specific args
    # Note: MLX uniform takes low, high, shape.
    return _create_new_tensor(mx.random.uniform, dtype=dtype, device=device, shape=shape, low=minval, high=maxval)

def random_binomial(shape: Shape, p: float = 0.5,
                   dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create a tensor with random values from a binomial distribution.
    
    Args:
        shape: Shape of the tensor
        p: Probability of success
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array with random binomial values
    """
    # Use the helper function, passing mx.random.bernoulli and its specific args
    # Pass probability p via kwargs. Helper handles dtype/device.
    return _create_new_tensor(mx.random.bernoulli, dtype=dtype, device=device, shape=shape, p=p)

def random_exponential(shape: Shape, scale: float = 1.0,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Generate random values from an exponential distribution.
    
    Args:
        shape: Shape of the output array
        scale: Scale parameter
        dtype: Optional data type
        device: Ignored for MLX backend
    
    Returns:
        MLX array with random values from an exponential distribution
    """
    # MLX doesn't have direct exponential. Sample uniform and transform: -scale * log(1-U)
    # Use helper for uniform sampling first.
    u = _create_new_tensor(mx.random.uniform, shape=shape, dtype=dtype, device=device) # Use target dtype for intermediate if specified

    # Perform transformation using mx ops
    scale_tensor = mx.array(scale) # Scale doesn't need helper conversion
    # Ensure 1.0 has compatible dtype with u if u's dtype was specified
    one_tensor = mx.array(1.0, dtype=u.dtype)
    # Clamp to avoid log(0)
    log_input = mx.maximum(mx.subtract(one_tensor, u), mx.array(1e-9, dtype=u.dtype))
    result = mx.multiply(mx.negative(scale_tensor), mx.log(log_input))

    # The result should already have the correct dtype from u or inference
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
        device: Ignored for MLX backend
    
    Returns:
        MLX array with random values from a gamma distribution
    """
    if alpha <= 0:
        raise ValueError("Alpha parameter must be positive")
    # Use the helper function, passing mx.random.gamma and its specific args
    # Note: mx.random.gamma takes shape_param (alpha) and scale (beta)
    return _create_new_tensor(mx.random.gamma, dtype=dtype, device=device, shape=shape, shape_param=alpha, scale=beta)

def random_poisson(shape: Shape, lam: float = 1.0,
                  dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Generate random values from a Poisson distribution.
    
    Args:
        shape: Shape of the output array
        lam: Rate parameter
        dtype: Optional data type
        device: Ignored for MLX backend
    
    Returns:
        MLX array with random values from a Poisson distribution
    """
    # Use the helper function, passing mx.random.poisson and its specific args
    return _create_new_tensor(mx.random.poisson, dtype=dtype, device=device, shape=shape, lam=lam)

def random_categorical(logits: TensorLike, num_samples: int,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Draw samples from a categorical distribution.
    
    Args:
        logits: 2D tensor with unnormalized log probabilities
        num_samples: Number of samples to draw
        dtype: Optional data type
        device: Ignored for MLX backend
    
    Returns:
        MLX array with random categorical values
    """
    # Import here to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor


    logits_tensor = MLXTensor().convert_to_tensor(logits)
    
    # MLX's categorical function takes num_samples parameter
    result = mx.random.categorical(logits=logits_tensor, num_samples=num_samples)
    
    # Validate dtype
    from ember_ml.backend.mlx.tensor.ops.utility import _validate_and_get_mlx_dtype
    mlx_dtype = _validate_and_get_mlx_dtype(dtype)
    if mlx_dtype is not None:
        result = result.astype(mlx_dtype)
    
    return result

def random_permutation(x: Union[int, TensorLike], dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Randomly permute a sequence or return a permuted range.
    
    Args:
        x: If x is an integer, randomly permute mx.arange(x).
           If x is an array, make a copy and shuffle the elements randomly.
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        Permuted array
    """
    if isinstance(x, int):
        # Create a range and permute it
        arr = mx.arange(x)
        indices = mx.random.permutation(x)
        return arr[indices]
    else:
        # Import here to avoid circular imports
        from ember_ml.backend.mlx.tensor import MLXTensor
        arr = MLXTensor().convert_to_tensor(x)
        indices = mx.random.permutation(arr.shape[0])
        return arr[indices]

def shuffle(x: TensorLike) -> mx.array:
    """
    Randomly shuffle an MLX array along the first dimension.
    
    Args:
        x: Input array
    
    Returns:
        Shuffled MLX array
    """
    # Import here to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor

    x_tensor = MLXTensor().convert_to_tensor(x)
    
    # Get the shape of the tensor
    shape = x_tensor.shape
    
    # If the tensor is empty or has only one element, return it as is
    if shape[0] <= 1:
        return x_tensor
    
    # Generate random indices
    indices = mx.random.permutation(shape[0])
    
    # Gather along the first dimension
    return mx.take(x_tensor, indices, axis=0)

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    mx.random.seed(seed)

def get_seed() -> Any:
    """
    Get the current random seed.
    
    Returns:
        Current random seed (None if not set)
    """
    # MLX doesn't provide a way to get the current seed
    return None