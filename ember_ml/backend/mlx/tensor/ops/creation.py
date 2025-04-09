"""MLX tensor creation operations."""

from typing import Any, List, Optional, Union

import mlx.core as mx
import numpy as np

from ember_ml.backend.mlx.tensor.dtype import MLXDType
from ember_ml.backend.mlx.types import DType, TensorLike, Shape, ShapeLike, ScalarLike

def zeros(shape: 'Shape', dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create an MLX array of zeros."""
    # Validate dtype
    dtype_cls = MLXDType()
    mlx_dtype = dtype_cls.from_dtype_str(dtype) if dtype else None
    
    # Create zeros array with the specified shape and dtype
    return mx.zeros(shape, dtype=mlx_dtype)

def ones(shape: 'Shape', dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create an MLX array of ones."""
    # Validate dtype
    dtype_cls = MLXDType()
    mlx_dtype = dtype_cls.from_dtype_str(dtype) if dtype else None
    
    # Create ones array with the specified shape and dtype
    return mx.ones(shape, dtype=mlx_dtype)

def zeros_like(tensor: 'TensorLike', dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create an MLX array of zeros with the same shape as the input."""
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor as Tensor
    tensor_array = Tensor().convert_to_tensor(tensor)
    
    # Get shape of input tensor
    shape = tensor_array.shape
    
    # Create zeros array with the same shape
    return zeros(shape, dtype, device)

def ones_like(tensor: 'TensorLike', dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create an MLX array of ones with the same shape as the input."""
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor as Tensor
    tensor_array = Tensor().convert_to_tensor(tensor)
    
    # Get shape of input tensor
    shape = tensor_array.shape
    
    # Create ones array with the same shape
    return ones(shape, dtype, device)

def full(shape: 'ShapeLike', fill_value: 'ScalarLike', dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create a tensor filled with a scalar value."""
    # Handle scalar shape case
    if isinstance(shape, (int, np.integer)):
        shape = (shape,)
        
    # Validate dtype
    dtype_cls = MLXDType()
    mlx_dtype = dtype_cls.from_dtype_str(dtype) if dtype else None
    
    # Create array of the specified shape filled with fill_value
    return mx.full(shape, fill_value, dtype=mlx_dtype)

def full_like(tensor: 'TensorLike', fill_value: 'ScalarLike', dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create a tensor filled with fill_value with the same shape as input."""
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor as Tensor
    tensor_array = Tensor().convert_to_tensor(tensor)
    
    # Get shape of input tensor
    shape = tensor_array.shape
    
    # Create array with the same shape filled with fill_value
    return full(shape, fill_value, dtype, device)

def eye(n: int, m: Optional[int] = None, dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create an identity matrix."""
    # If m is not specified, use n
    if m is None:
        m = n
    
    # Validate dtype
    dtype_cls = MLXDType()
    mlx_dtype = dtype_cls.from_dtype_str(dtype) if dtype else None
    
    # Create identity matrix
    return mx.eye(n, m, dtype=mlx_dtype)

def arange(start: ScalarLike, stop: ScalarLike = None, step: int = 1,
          dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create a sequence of numbers.
    
    Args:
        start: Starting value (inclusive)
        stop: Ending value (exclusive)
        step: Step size
        dtype: Data type of the output
        device: Device to place the output on
        
    Returns:
        A tensor with values from start to stop with step size
    """
    # Validate dtype
    dtype_cls = MLXDType()
    mlx_dtype = dtype_cls.from_dtype_str(dtype) if dtype else None
    
    # Handle single argument case
    if stop is None:
        stop = start
        start = 0
    
    # Convert tensor inputs to Python scalars if needed
    if hasattr(start, 'item'):
        start = float(start.item())
    if hasattr(stop, 'item'):
        stop = float(stop.item())
    
    # Create sequence
    return mx.arange(start, stop, step, dtype=mlx_dtype)

def linspace(start: Union[int, float], stop: Union[int, float], num: int,
            dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create evenly spaced numbers over a specified interval."""
    # Validate dtype
    dtype_cls = MLXDType()
    mlx_dtype = dtype_cls.from_dtype_str(dtype) if dtype else None
    
    # Create evenly spaced sequence
    return mx.linspace(start, stop, num, dtype=mlx_dtype)