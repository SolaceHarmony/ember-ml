"""MLX tensor creation operations."""

import mlx.core as mx
from typing import Union, Optional, Sequence, Any

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

def zeros(tensor_obj, shape: Shape, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array of zeros.
    
    Args:
        tensor_obj: MLXTensor instance
        shape: Shape of the array
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array of zeros
    """
    mlx_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    return mx.zeros(shape, dtype=mlx_dtype)

def ones(tensor_obj, shape: Shape, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array of ones.
    
    Args:
        tensor_obj: MLXTensor instance
        shape: Shape of the array
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array of ones
    """
    mlx_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    return mx.ones(shape, dtype=mlx_dtype)

def eye(tensor_obj, n: int, m: Optional[int] = None, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX identity matrix.
    
    Args:
        tensor_obj: MLXTensor instance
        n: Number of rows
        m: Number of columns (default: n)
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX identity matrix of shape (n, m)
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        dtype = MLXDType().from_dtype_str(dtype)

    if m is None:
        m = n
    return mx.eye(n, m, dtype=dtype)

def zeros_like(tensor_obj, tensor, dtype=None, device=None):
    """
    Create an MLX array of zeros with the same shape as the input.
    
    Args:
        tensor_obj: MLXTensor instance
        tensor: Input array
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array of zeros with the same shape as tensor
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        dtype = MLXDType().from_dtype_str(dtype)
    
    tensor_array = tensor_obj.convert_to_tensor(tensor)
    # MLX zeros_like doesn't accept dtype parameter
    if dtype is None:
        return mx.zeros_like(tensor_array)
    else:
        # Create zeros with the same shape but specified dtype
        return mx.zeros(tensor_array.shape, dtype=dtype)

def ones_like(tensor_obj, tensor, dtype=None, device=None):
    """
    Create an MLX array of ones with the same shape as the input.
    
    Args:
        tensor_obj: MLXTensor instance
        tensor: Input array
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array of ones with the same shape as tensor
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        dtype = MLXDType().from_dtype_str(dtype)

    tensor_array = tensor_obj.convert_to_tensor(tensor)
    # MLX ones_like doesn't accept dtype parameter
    if dtype is None:
        return mx.ones_like(tensor_array)
    else:
        # Create ones with the same shape but specified dtype
        return mx.ones(tensor_array.shape, dtype=dtype)

def full(tensor_obj, shape, fill_value, dtype=None, device=None):
    """
    Create an MLX array filled with a scalar value.

    Args:
        tensor_obj: MLXTensor instance
        shape: Shape of the array
        fill_value: Value to fill the array with
        dtype: Optional data type
        device: Ignored for MLX backend

    Returns:
        MLX array filled with the specified value
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        dtype = MLXDType().from_dtype_str(dtype)

    return mx.full(shape, fill_value, dtype=dtype)

def full_like(tensor_obj, tensor, fill_value, dtype=None, device=None):
    """
    Create an MLX array filled with a scalar value with the same shape as the input.

    Args:
        tensor_obj: MLXTensor instance
        tensor: Input array
        fill_value: Value to fill the array with
        dtype: Optional data type
        device: Ignored for MLX backend

    Returns:
        MLX array filled with the specified value with the same shape as tensor
    """
    tensor_array = tensor_obj.convert_to_tensor(tensor)

    # Handle string dtype values
    if isinstance(dtype, str):
        dtype = MLXDType().from_dtype_str(dtype)

    # If dtype is None, use the dtype of the input array
    if dtype is None:
        dtype = tensor_array.dtype

    # Create a full array with the same shape as the input
    return mx.full(tensor_array.shape, fill_value, dtype=dtype)

def arange(tensor_obj, start, stop=None, step=1, dtype=None, device=None):
    """
    Create an MLX array with evenly spaced values within a given interval.

    Args:
        tensor_obj: MLXTensor instance
        start: Start of interval (inclusive)
        stop: End of interval (exclusive)
        step: Spacing between values
        dtype: Optional data type
        device: Ignored for MLX backend

    Returns:
        MLX array with evenly spaced values
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        dtype = MLXDType().from_dtype_str(dtype)
    # Handle EmberDtype objects
    elif dtype is not None and hasattr(dtype, 'name') and hasattr(dtype, 'ember_dtype'):
        dtype = MLXDType().from_dtype_str(dtype.name)

    if stop is None:
        # If only one argument is provided, it's the stop value
        return mx.arange(start=0, stop=start, step=step, dtype=dtype)
    return mx.arange(start=start, stop=stop, step=step, dtype=dtype)

def linspace(tensor_obj, start, stop, num, dtype=None, device=None):
    """
    Create an MLX array with evenly spaced values within a given interval.

    Args:
        tensor_obj: MLXTensor instance
        start: Start of interval (inclusive)
        stop: End of interval (inclusive)
        num: Number of values to generate
        dtype: Optional data type
        device: Ignored for MLX backend

    Returns:
        MLX array with evenly spaced values
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        dtype = MLXDType().from_dtype_str(dtype)
    # Handle EmberDtype objects
    elif dtype is not None and hasattr(dtype, 'name') and hasattr(dtype, 'ember_dtype'):
        dtype = MLXDType().from_dtype_str(dtype.name)

    return mx.linspace(start=start, stop=stop, num=num, dtype=dtype)