"""NumPy tensor random operations."""

import numpy as np
from typing import Union, Optional, Sequence, Any, List, Tuple

from ember_ml.backend.numpy.tensor.dtype import NumpyDType

# Type aliases
Shape = Union[int, Sequence[int]]
DType = Any

def _validate_dtype(dtype_cls: NumpyDType, dtype: Optional[DType]) -> Optional[Any]:
    """
    Validate and convert dtype to NumPy format.
    
    Args:
        dtype_cls: NumpyDType instance for conversions
        dtype: Input dtype to validate
        
    Returns:
        Validated NumPy dtype or None
    """
    if dtype is None:
        return None
    
    # Handle string dtypes
    if isinstance(dtype, str):
        return dtype_cls.from_dtype_str(dtype)
        
    # Handle EmberDType objects
    if hasattr(dtype, 'name'):
        return dtype_cls.from_dtype_str(str(dtype.name))
        
    # If it's already a NumPy dtype, return as is
    if isinstance(dtype, np.dtype) or dtype in [np.float32, np.float64, np.int32, np.int64, 
                                               np.bool_, np.int8, np.int16, np.uint8, 
                                               np.uint16, np.uint32, np.uint64, np.float16]:
        return dtype
        
    raise ValueError(f"Invalid dtype: {dtype}")

def random_normal(tensor_obj, shape: Shape, mean: float = 0.0, stddev: float = 1.0,
                 dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a tensor with random values from a normal distribution.
    
    Args:
        tensor_obj: NumpyTensor instance
        shape: The shape of the tensor
        mean: The mean of the normal distribution
        stddev: The standard deviation of the normal distribution
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        Tensor with random values from a normal distribution
    """
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
    
    # Validate and convert dtype
    numpy_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    
    # Create a tensor with random values from a normal distribution
    tensor = np.random.normal(loc=mean, scale=stddev, size=shape)
    
    # Cast to the specified dtype if needed
    if numpy_dtype is not None:
        tensor = tensor.astype(numpy_dtype)
    
    return tensor

def random_uniform(tensor_obj, shape: Shape, minval: float = 0.0, maxval: float = 1.0,
                  dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a tensor with random values from a uniform distribution.
    
    Args:
        tensor_obj: NumpyTensor instance
        shape: The shape of the tensor
        minval: Minimum value
        maxval: Maximum value
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        Tensor with random values from a uniform distribution
    """
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
    
    # Validate and convert dtype
    numpy_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    
    # Create a tensor with random values from a uniform distribution
    tensor = np.random.uniform(low=minval, high=maxval, size=shape)
    
    # Cast to the specified dtype if needed
    if numpy_dtype is not None:
        tensor = tensor.astype(numpy_dtype)
    
    return tensor

def random_binomial(tensor_obj, shape: Shape, p: float = 0.5,
                   dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a tensor with random values from a binomial distribution.
    
    Args:
        tensor_obj: NumpyTensor instance
        shape: The shape of the tensor
        p: Probability of success
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        Tensor with random values from a binomial distribution
    """
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
    
    # Validate and convert dtype
    numpy_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    
    # Create a tensor with random values from a binomial distribution
    tensor = np.random.binomial(n=1, p=p, size=shape)
    
    # Cast to the specified dtype if needed
    if numpy_dtype is not None:
        tensor = tensor.astype(numpy_dtype)
    
    return tensor

def random_gamma(tensor_obj, shape: Shape, alpha: float = 1.0, beta: float = 1.0,
                dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Generate random values from a gamma distribution.
    
    Args:
        tensor_obj: NumpyTensor instance
        shape: Shape of the output array
        alpha: Shape parameter
        beta: Scale parameter
        dtype: Optional data type
        device: Ignored for NumPy backend
    
    Returns:
        NumPy array with random values from a gamma distribution
    """
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
    
    # Validate and convert dtype
    numpy_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    
    # Create a tensor with random values from a gamma distribution
    tensor = np.random.gamma(shape=alpha, scale=beta, size=shape)
    
    # Cast to the specified dtype if needed
    if numpy_dtype is not None:
        tensor = tensor.astype(numpy_dtype)
    
    return tensor

def random_exponential(tensor_obj, shape: Shape, scale: float = 1.0,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Generate random values from an exponential distribution.
    
    Args:
        tensor_obj: NumpyTensor instance
        shape: Shape of the output array
        scale: Scale parameter
        dtype: Optional data type
        device: Ignored for NumPy backend
    
    Returns:
        NumPy array with random values from an exponential distribution
    """
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
    
    # Validate and convert dtype
    numpy_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    
    # Create a tensor with random values from an exponential distribution
    tensor = np.random.exponential(scale=scale, size=shape)
    
    # Cast to the specified dtype if needed
    if numpy_dtype is not None:
        tensor = tensor.astype(numpy_dtype)
    
    return tensor

def random_poisson(tensor_obj, shape: Shape, lam: float = 1.0,
                  dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Generate random values from a Poisson distribution.
    
    Args:
        tensor_obj: NumpyTensor instance
        shape: Shape of the output array
        lam: Rate parameter
        dtype: Optional data type
        device: Ignored for NumPy backend
    
    Returns:
        NumPy array with random values from a Poisson distribution
    """
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
    
    # Validate and convert dtype
    numpy_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    
    # Create a tensor with random values from a Poisson distribution
    tensor = np.random.poisson(lam=lam, size=shape)
    
    # Cast to the specified dtype if needed
    if numpy_dtype is not None:
        tensor = tensor.astype(numpy_dtype)
    
    return tensor

def random_categorical(tensor_obj, logits: Any, num_samples: int,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Draw samples from a categorical distribution.
    
    Args:
        tensor_obj: NumpyTensor instance
        logits: 2D tensor with unnormalized log probabilities
        num_samples: Number of samples to draw
        dtype: Optional data type
        device: Ignored for NumPy backend
    
    Returns:
        NumPy array with random categorical values
    """
    # Validate and convert dtype
    numpy_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    
    # Convert to NumPy array if needed
    logits_array = tensor_obj.convert_to_tensor(logits)
    
    # Convert to probabilities
    # Use np.subtract instead of - operator
    max_logits = np.max(logits_array, axis=-1, keepdims=True)
    exp_logits = np.exp(np.subtract(logits_array, max_logits))
    
    # Use np.divide instead of / operator
    sum_exp_logits = np.sum(exp_logits, axis=-1, keepdims=True)
    probs = np.divide(exp_logits, sum_exp_logits)
    
    # Sample from the categorical distribution
    samples = np.zeros((logits_array.shape[0], num_samples), dtype=np.int64)
    for i in range(logits_array.shape[0]):
        samples[i] = np.random.choice(logits_array.shape[1], size=num_samples, p=probs[i])
    
    # Cast to the specified dtype if needed
    if numpy_dtype is not None:
        samples = samples.astype(numpy_dtype)
    
    return samples

def random_permutation(tensor_obj, x: Union[int, Any], dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Randomly permute a sequence or return a permuted range.
    
    Args:
        tensor_obj: NumpyTensor instance
        x: If x is an integer, randomly permute np.arange(x).
           If x is an array, make a copy and shuffle the elements randomly.
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        Permuted array
    """
    # Validate and convert dtype
    numpy_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    
    if isinstance(x, int):
        # Create a range and permute it
        perm = np.random.permutation(x)
        
        # Cast to the specified dtype if needed
        if numpy_dtype is not None:
            perm = perm.astype(numpy_dtype)
        
        return perm
    else:
        # Convert to NumPy array if needed
        x_array = tensor_obj.convert_to_tensor(x)
        
        # Get the shape of the array
        shape = x_array.shape
        
        # If the array is empty or has only one element, return it as is
        if shape[0] <= 1:
            return x_array
        
        # Generate random indices
        indices = np.random.permutation(shape[0])
        
        # Gather along the first dimension
        return x_array[indices]

def shuffle(tensor_obj, x: Any) -> np.ndarray:
    """
    Randomly shuffle a NumPy array along its first dimension.
    
    Args:
        tensor_obj: NumpyTensor instance
        x: Input array
        
    Returns:
        Shuffled NumPy array
    """
    # Convert to NumPy array if needed
    x_array = tensor_obj.convert_to_tensor(x)
    
    # Get the shape of the array
    shape = x_array.shape
    
    # If the array is empty or has only one element, return it as is
    if shape[0] <= 1:
        return x_array
    
    # Generate random indices
    indices = np.random.permutation(shape[0])
    
    # Gather along the first dimension
    return x_array[indices]

def set_seed(tensor_obj, seed: int) -> None:
    """
    Set the random seed for reproducibility.
    
    Args:
        tensor_obj: NumpyTensor instance
        seed: Random seed
    """
    np.random.seed(seed)

def get_seed(tensor_obj) -> Optional[int]:
    """
    Get the current random seed.
    
    Args:
        tensor_obj: NumpyTensor instance
    
    Returns:
        Current random seed or None if not set
    """
    # NumPy doesn't provide a way to get the current seed
    return None