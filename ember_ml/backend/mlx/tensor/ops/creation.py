"""MLX tensor creation operations."""

from typing import Any, List, Optional, Union

import mlx.core as mx
import numpy as np

from ember_ml.backend.mlx.tensor.dtype import MLXDType
from ember_ml.backend.mlx.types import DType, TensorLike, Shape, ShapeLike, ScalarLike

def zeros(shape: 'Shape', dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create an MLX array of zeros."""
    # Validate dtype
    from ember_ml.backend.mlx.tensor.ops.utility import _create_new_tensor
    x = _create_new_tensor(mx.zeros, dtype, device,shape=shape)
    # Create zeros array with the specified shape and dtype
    return x

def ones(shape: 'Shape', dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create an MLX array of ones."""
    # Validate dtype
    from ember_ml.backend.mlx.tensor.ops.utility import _create_new_tensor
    x = _create_new_tensor(mx.ones, dtype, device,shape=shape)
    
    # Create ones array with the specified shape and dtype
    return x

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
        
    from ember_ml.backend.mlx.tensor.ops.utility import _create_new_tensor
    x = _create_new_tensor(mx.full, dtype, device,shape=shape)
    
    # Create array of the specified shape filled with fill_value
    return x

def full_like(tensor: 'TensorLike', fill_value: 'ScalarLike', dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create a tensor filled with fill_value with the same shape as input."""
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor as Tensor
    tensor_array = Tensor().convert_to_tensor(tensor)
    
    # Get shape of input tensor
    shape = tensor_array.shape
    
    # Create array with the same shape filled with fill_value
    return full(shape, fill_value, dtype, device)


def eye(n, m=None, dtype=None, device=None):
    """Create a 2D tensor with ones on the diagonal and zeros elsewhere.

    Args:
        n: int, number of rows.
        m: int, optional, number of columns. If None, defaults to n.
        dtype: data type of the returned tensor.
        device: device on which to place the created tensor.

    Returns:
        A 2D tensor with ones on the diagonal and zeros elsewhere.
    """
    if m is None:
        m = n
    
    # Create a zeros tensor with shape (n, m)
    from ember_ml.backend.mlx.tensor.ops.utility import _create_new_tensor
    x = _create_new_tensor(mx.zeros, dtype, device, shape=(n, m))
    
    # Set diagonal elements to 1
    min_dim = min(n, m)
    indices = mx.arange(min_dim)
    from ember_ml.backend.mlx.tensor.ops import scatter
    x = scatter(x, (indices, indices), mx.ones(min_dim, dtype=x.dtype))
    
    return x

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

    
    # Handle single argument case
    if stop is None:
        stop = start
        start = 0
    
    # Convert tensor inputs to Python scalars if needed
    if hasattr(start, 'item'):
        start = float(start.item())
    if hasattr(stop, 'item'):
        stop = float(stop.item())

    from ember_ml.backend.mlx.tensor.ops.utility import _create_new_tensor
    x = _create_new_tensor(mx.arange, dtype=dtype,start=start, stop=stop, step=step)
    # Create sequence
    return x

def linspace(start: Union[int, float], stop: Union[int, float], num: int,
            dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create evenly spaced numbers over a specified interval."""
    # Validate dtype
    from ember_ml.backend.mlx.tensor import MLXTensor
    start_tensor = MLXTensor().convert_to_tensor(start)
    stop_tensor = MLXTensor().convert_to_tensor(stop)
    num_tensor = MLXTensor().convert_to_tensor(num)
    from ember_ml.backend.mlx.tensor.ops.utility import _create_new_tensor
    x = _create_new_tensor(mx.linspace, dtype=dtype,start=start_tensor, stop=stop_tensor, num=num_tensor)
    
    # Create evenly spaced sequence
    return x