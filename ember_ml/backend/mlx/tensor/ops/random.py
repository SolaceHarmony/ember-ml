"""MLX tensor random operations."""

import mlx.core as mx
from typing import Union, Optional, Sequence, Any, List, Tuple

from ember_ml.backend.mlx.tensor.dtype import MLXDType, DType

# Type aliases
Shape = Union[int, Sequence[int]]

def _validate_dtype(dtype_cls: MLXDType, dtype: Optional[DType]) -> Optional[Any]:
    """
    Validate and convert dtype to MLX format.
    
    Args:
        dtype_cls: MLXDType instance for conversions
        dtype: Input dtype to validate
        
    Returns:
        Validated MLX dtype or None
    """
    if dtype is None:
        return None
    
    # Handle string dtypes
    if isinstance(dtype, str):
        return dtype_cls.from_dtype_str(dtype)
        
    # Handle EmberDType objects
    if hasattr(dtype, 'name'):
        return dtype_cls.from_dtype_str(str(dtype.name))
        
    # If it's already an MLX dtype, return as is
    if isinstance(dtype, type(mx.float32)):
        return dtype
        
    raise ValueError(f"Invalid dtype: {dtype}")

def random_normal(tensor_obj, shape: Shape, mean: float = 0.0, stddev: float = 1.0,
                 dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create a tensor with random values from a normal distribution.
    
    Args:
        tensor_obj: MLXTensor instance
        shape: Shape of the tensor
        mean: Mean of the normal distribution
        stddev: Standard deviation of the normal distribution
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array with random normal values
    """
    # Convert shape to a sequence if it's an int
    if isinstance(shape, int):
        shape = (shape,)
    
    # Validate dtype
    mlx_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    
    # Use MLX's normal function
    return mx.random.normal(shape=shape, loc=mean, scale=stddev, dtype=mlx_dtype)

def random_uniform(tensor_obj, shape: Shape, minval: float = 0.0, maxval: float = 1.0,
                  dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create a tensor with random values from a uniform distribution.
    
    Args:
        tensor_obj: MLXTensor instance
        shape: Shape of the tensor
        minval: Minimum value
        maxval: Maximum value
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array with random uniform values
    """
    # Convert shape to a sequence if it's an int
    if isinstance(shape, int):
        shape = (shape,)
    
    # Validate dtype
    mlx_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    
    # Use MLX's uniform function
    return mx.random.uniform(shape=shape, low=minval, high=maxval, dtype=mlx_dtype)

def random_binomial(tensor_obj, shape: Shape, p: float = 0.5,
                   dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create a tensor with random values from a binomial distribution.
    
    Args:
        tensor_obj: MLXTensor instance
        shape: Shape of the tensor
        p: Probability of success
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array with random binomial values
    """
    # Convert shape to a sequence if it's an int
    if isinstance(shape, int):
        shape = (shape,)
    
    # Validate dtype
    mlx_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    
    # Use MLX's bernoulli function
    result = mx.random.bernoulli(p=p, shape=shape)
    
    # Convert to the specified dtype if needed
    if mlx_dtype is not None:
        result = result.astype(mlx_dtype)
    
    return result

def random_exponential(tensor_obj, shape: Shape, scale: float = 1.0,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Generate random values from an exponential distribution.
    
    Args:
        tensor_obj: MLXTensor instance
        shape: Shape of the output array
        scale: Scale parameter
        dtype: Optional data type
        device: Ignored for MLX backend
    
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
    
    # Validate dtype
    mlx_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    if mlx_dtype is not None:
        result = result.astype(mlx_dtype)
    
    return result

def random_gamma(tensor_obj, shape: Shape, alpha: float = 1.0, beta: float = 1.0,
                dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Generate random values from a gamma distribution.
    
    Args:
        tensor_obj: MLXTensor instance
        shape: Shape of the output array
        alpha: Shape parameter
        beta: Scale parameter
        dtype: Optional data type
        device: Ignored for MLX backend
    
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
        return random_exponential(tensor_obj, shape, scale=beta, dtype=dtype, device=device)
    
    # For integer alpha, we can use the sum of exponentials
    if isinstance(alpha, int) and alpha > 1:
        result = mx.zeros(shape)
        for _ in range(alpha):
            result = mx.add(result, random_exponential(tensor_obj, shape, scale=beta, dtype=None, device=device))
        
        # Validate dtype
        mlx_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
        if mlx_dtype is not None:
            result = result.astype(mlx_dtype)
        
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
    
    # Validate dtype
    mlx_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    if mlx_dtype is not None:
        result = result.astype(mlx_dtype)
    
    return result

def random_poisson(tensor_obj, shape: Shape, lam: float = 1.0,
                  dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Generate random values from a Poisson distribution.
    
    Args:
        tensor_obj: MLXTensor instance
        shape: Shape of the output array
        lam: Rate parameter
        dtype: Optional data type
        device: Ignored for MLX backend
    
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
        lam_array = tensor_obj.convert_to_tensor(lam)
    
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
    
    # Validate dtype
    mlx_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    if mlx_dtype is not None and mlx_dtype != mx.int32:
        counts = counts.astype(mlx_dtype)
    
    return counts

def random_categorical(tensor_obj, logits: Any, num_samples: int,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Draw samples from a categorical distribution.
    
    Args:
        tensor_obj: MLXTensor instance
        logits: 2D tensor with unnormalized log probabilities
        num_samples: Number of samples to draw
        dtype: Optional data type
        device: Ignored for MLX backend
    
    Returns:
        MLX array with random categorical values
    """
    # Convert to MLX array if needed
    logits_tensor = tensor_obj.convert_to_tensor(logits)
    
    # MLX's categorical function takes num_samples parameter
    result = mx.random.categorical(logits=logits_tensor, num_samples=num_samples)
    
    # Validate dtype
    mlx_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    if mlx_dtype is not None:
        result = result.astype(mlx_dtype)
    
    return result

def random_permutation(tensor_obj, x: Union[int, Any], dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Randomly permute a sequence or return a permuted range.
    
    Args:
        tensor_obj: MLXTensor instance
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
        # Convert to MLX array if needed
        arr = tensor_obj.convert_to_tensor(x)
        indices = mx.random.permutation(arr.shape[0])
        return arr[indices]

def shuffle(tensor_obj, x: Any) -> mx.array:
    """
    Randomly shuffle an MLX array along the first dimension.
    
    Args:
        tensor_obj: MLXTensor instance
        x: Input array
    
    Returns:
        Shuffled MLX array
    """
    x_tensor = tensor_obj.convert_to_tensor(x)
    
    # Get the shape of the tensor
    shape = x_tensor.shape
    
    # If the tensor is empty or has only one element, return it as is
    if shape[0] <= 1:
        return x_tensor
    
    # Generate random indices
    indices = mx.random.permutation(shape[0])
    
    # Gather along the first dimension
    return mx.take(x_tensor, indices, axis=0)

def set_seed(tensor_obj, seed: int) -> None:
    """
    Set the random seed for reproducibility.
    
    Args:
        tensor_obj: MLXTensor instance
        seed: Random seed
    """
    mx.random.seed(seed)

def get_seed(tensor_obj) -> Optional[int]:
    """
    Get the current random seed.
    
    Args:
        tensor_obj: MLXTensor instance
    
    Returns:
        Current random seed (None if not set)
    """
    # MLX doesn't provide a way to get the current seed
    return None