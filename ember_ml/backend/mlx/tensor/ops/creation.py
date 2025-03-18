"""MLX tensor creation operations."""

import mlx.core as mx
from typing import Union, Optional, Any

from ember_ml.backend.mlx.tensor.dtype import MLXDType
from ember_ml.backend.mlx.tensor.tensor import MLXTensor
from ember_ml.backend.mlx.config import ShapeLike, TensorLike, DType, ScalarLike, Scalar

# Create single instances to reuse throughout the module
Tensor = MLXTensor()
DTypeHandler = MLXDType()

def _validate_dtype(dtype: Optional[DType]) -> Optional[Any]:
    """
    Validate and convert dtype to MLX format.
    
    Args:
        dtype: Input dtype to validate
        
    Returns:
        Validated MLX dtype or None
    """
    if dtype is None:
        return None
    
    # Handle string dtypes
    if isinstance(dtype, str):
        return DTypeHandler.from_dtype_str(dtype)
        
    # Handle EmberDType objects
    if hasattr(dtype, 'name'):
        return DTypeHandler.from_dtype_str(str(dtype.name)) # type: ignore
        
    # If it's already an MLX dtype, return as is
    if isinstance(dtype, type(mx.float32)):
        return dtype
        
    raise ValueError(f"Invalid dtype: {dtype}")

def zeros(shape: ShapeLike, dtype: Optional[DType]=None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array of zeros.
    
    Args:
        shape: Shape of the array
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array of zeros
    """
    mlx_dtype = _validate_dtype(dtype)
    return mx.zeros(shape, dtype=mlx_dtype)

def ones(shape: ShapeLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array of ones.
    
    Args:
        shape: Shape of the array
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array of ones
    """
    mlx_dtype = _validate_dtype(dtype)
    return mx.ones(shape, dtype=mlx_dtype)

def eye(n: int, m: Optional[int] = None, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX identity matrix.
    
    Args:
        n: Number of rows
        m: Number of columns (default: n)
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX identity matrix of shape (n, m)
    """
    # Handle string dtype values
    mlx_dtype = _validate_dtype(dtype)

    if m is None:
        m = n
    return mx.eye(n, m, dtype=mlx_dtype)

def zeros_like(tensor: TensorLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array of zeros with the same shape as the input.
    
    Args:
        tensor: Input array
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array of zeros with the same shape as tensor
    """
    mlx_dtype = _validate_dtype(dtype)
    
    tensor_array = Tensor.convert_to_tensor(tensor)
    # MLX zeros_like doesn't accept dtype parameter
    if dtype is None:
        return mx.zeros_like(tensor_array)
    else:
        # Create zeros with the same shape but specified dtype
        return mx.zeros(tensor_array.shape, dtype=mlx_dtype)

def ones_like(tensor: TensorLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array of ones with the same shape as the input.
    
    Args:
        tensor: Input array
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array of ones with the same shape as tensor
    """
    mlx_dtype = _validate_dtype(dtype)

    tensor_array = Tensor.convert_to_tensor(tensor)
    # MLX ones_like doesn't accept dtype parameter
    if dtype is None:
        return mx.ones_like(tensor_array)
    else:
        # Create ones with the same shape but specified dtype
        return mx.ones(tensor_array.shape, dtype=mlx_dtype)

def full(shape: ShapeLike, fill_value: ScalarLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array filled with a scalar value.

    Args:
        shape: Shape of the array
        fill_value: Value to fill the array with
        dtype: Optional data type
        device: Ignored for MLX backend

    Returns:
        MLX array filled with the specified value
    """
    mlx_dtype = _validate_dtype(dtype)
    return mx.full(shape=shape, vals=fill_value, dtype=mlx_dtype)

def full_like(tensor: TensorLike, fill_value: ScalarLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array filled with a scalar value with the same shape as the input.

    Args:
        tensor: Input array
        fill_value: Value to fill the array with
        dtype: Optional data type
        device: Ignored for MLX backend

    Returns:
        MLX array filled with the specified value with the same shape as tensor
    """
    tensor_array = Tensor.convert_to_tensor(tensor)
    mlx_dtype = _validate_dtype(dtype)

    # If dtype is None, use the dtype of the input array
    if mlx_dtype is None:
        mlx_dtype = tensor_array.dtype

    # Create a full array with the same shape as the input
    return mx.full(tensor_array.shape, fill_value, dtype=mlx_dtype)

def arange(start: Scalar, stop: Optional[Scalar] = None, step: int = 1, 
          dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array with evenly spaced values within a given interval.

    Args:
        start: Start of interval (inclusive)
        stop: End of interval (exclusive)
        step: Spacing between values
        dtype: Optional data type
        device: Ignored for MLX backend

    Returns:
        MLX array with evenly spaced values
    """
    mlx_dtype = _validate_dtype(dtype)

    if stop is None:
        # If only one argument is provided, it's the stop value
        return mx.arange(start=0, stop=start, step=step, dtype=mlx_dtype)
    return mx.arange(start=start, stop=stop, step=step, dtype=mlx_dtype)

def linspace(start: Union[int, float], stop: Union[int, float], num: Optional[int], 
            dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array with evenly spaced values within a given interval.

    Args:
        start: Start of interval (inclusive)
        stop: End of interval (inclusive)
        num: Number of values to generate
        dtype: Optional data type
        device: Ignored for MLX backend

    Returns:
        MLX array with evenly spaced values
    """
    if dtype:
        mlx_dtype = _validate_dtype(dtype)
        return mx.linspace(start=start, stop=stop, num=num, dtype=mlx_dtype)
    return mx.linspace(start=start, stop=stop, num=num)