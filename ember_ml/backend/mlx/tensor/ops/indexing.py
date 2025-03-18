"""MLX tensor indexing operations."""

from typing import Any, List, Literal, Optional, Sequence, Union, cast, Protocol, TypeVar, runtime_checkable

import mlx.core as mx
import numpy as np

from ember_ml.backend.mlx.types import (
    DType, TensorLike, Shape, DimSize, Axis
)

# Create an instance of MLXTensor for conversion

__all__ = [
    'slice_tensor',
    'slice_update',
    'tensor_scatter_nd_update',
    'gather',
    'scatter',
    'scatter_add',
    'scatter_max',
    'scatter_min',
    'scatter_mean',
    'scatter_softmax',
    'slice'  # Alias for backward compatibility
]

def slice_tensor(tensor: TensorLike, starts: Shape, sizes: Any) -> mx.array:
    """Extract a slice from a tensor."""
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    
    # Create axes as a list of integers
    axes = list(range(len(starts)))
    
    # Use MLX's slice function with proper types
    return mx.slice(tensor_array, mx.array(starts), axes, sizes)

def gather(tensor: Any, indices: Any, axis: int = 0) -> mx.array:
    """Gather slices from a tensor along an axis."""
    # Convert inputs to MLX arrays
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    indices_array = Tensor.convert_to_tensor(indices)
    
    # Ensure indices are integers
    indices_int = indices_array.astype(mx.int32)
    
    # Use take operation for gathering
    return mx.take(tensor_array, indices_int, axis=axis)

def tensor_scatter_nd_update(tensor: Any, indices: Any, updates: Any) -> mx.array:
    """Update tensor elements at given indices."""
    # Convert inputs to MLX arrays
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    indices_array = Tensor.convert_to_tensor(indices)
    updates_array = Tensor.convert_to_tensor(updates)
    
    # Create a copy of the tensor
    result = mx.array(tensor_array)
    
    # Convert indices to integer lists for safe indexing
    if indices_array.ndim == 1:
        indices_list = [_to_int_list(indices_array)]
    else:
        indices_list = [_to_int_list(idx) for idx in indices_array]
    
    # Update the tensor using slice_update
    for i, idx in enumerate(indices_list):
        axes = list(range(len(idx)))
        result = mx.slice_update(result, updates_array[i], mx.array(idx), axes)
    
    return result

def slice_update(tensor: Any, slices: Any, updates: Optional[Any] = None) -> mx.array:
    """Update a tensor at specific indices."""
    # Convert inputs to MLX arrays
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    slices_array = Tensor.convert_to_tensor(slices)
    
    # Convert slices to integer list
    indices_list = _to_int_list(slices_array)
    
    # Create axes as list of integers
    axes = list(range(len(indices_list)))
    
    # If updates is None, return slice of tensor
    if updates is None:
        # Create a size list of ones matching the shape
        ones_list = [1] * len(axes)
        return mx.slice(tensor_array, mx.array(indices_list), axes, ones_list)
    
    # Convert updates to MLX array
    updates_array = Tensor.convert_to_tensor(updates)
    
    # Create a copy of the tensor
    result = mx.array(tensor_array)
    
    # Update the tensor using slice_update with proper axes
    return mx.slice_update(result, updates_array, mx.array(indices_list), axes)

def scatter(data: Any, indices: Any, dim_size: Optional[Union[int, mx.array]] = None,
           aggr: str = "add", axis: int = 0) -> mx.array:
    """Scatter values into a new tensor."""
    # Convert inputs to MLX arrays
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    data_array = Tensor.convert_to_tensor(data)
    indices_array = Tensor.convert_to_tensor(indices)
    
    # Ensure indices are integers
    indices_int = indices_array.astype(mx.int32)
    
    # Handle dim_size
    computed_dim_size = (_safe_int_conversion(indices_int.shape[0]) 
                        if dim_size is None 
                        else _safe_int_conversion(dim_size))
    
    return scatter_op(data_array, indices_int, computed_dim_size, axis, aggr)

def scatter_op(src: mx.array, index: mx.array, dim_size: int,
              axis: int, op: str) -> mx.array:
    """Helper function for scatter operations."""
    # Initialize output tensor based on operation
    if op == "add":
        out = mx.zeros((dim_size,), dtype=src.dtype)
    elif op in ["max", "softmax"]:
        out = mx.full((dim_size,), -float('inf'), dtype=src.dtype)
    elif op == "min":
        out = mx.full((dim_size,), float('inf'), dtype=src.dtype)
    else:
        raise ValueError(f"Unknown operation: {op}")
    
    # Convert indices to integer lists
    index_list = _to_int_list(index)
    
    # Perform scatter operation
    for i, idx in enumerate(index_list):
        idx_array = mx.array([idx])
        val_i = src[i]
        
        if op == "add":
            current = out[idx]
            out = mx.slice_update(out, current + val_i, idx_array, [0])
        elif op == "max":
            current = out[idx]
            out = mx.slice_update(out, mx.maximum(current, val_i), idx_array, [0])
        elif op == "min":
            current = out[idx]
            out = mx.slice_update(out, mx.minimum(current, val_i), idx_array, [0])
    
    return out

def scatter_add(src: Any, index: Any, dim_size: int, axis: int = 0) -> mx.array:
    """Scatter values using addition."""
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    src_array = Tensor.convert_to_tensor(src)
    index_array = Tensor.convert_to_tensor(index)
    return scatter_op(src_array, index_array, _safe_int_conversion(dim_size), axis, "add")

def scatter_max(src: Any, index: Any, dim_size: int, axis: int = 0) -> mx.array:
    """Scatter values using maximum."""
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    src_array = Tensor.convert_to_tensor(src)
    index_array = Tensor.convert_to_tensor(index)
    return scatter_op(src_array, index_array, _safe_int_conversion(dim_size), axis, "max")

def scatter_min(src: Any, index: Any, dim_size: int, axis: int = 0) -> mx.array:
    """Scatter values using minimum."""
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    src_array = Tensor.convert_to_tensor(src)
    index_array = Tensor.convert_to_tensor(index)
    return scatter_op(src_array, index_array, _safe_int_conversion(dim_size), axis, "min")

def scatter_mean(values: Any, index: Any, dim_size: int, axis: int = 0) -> mx.array:
    """Scatter values and compute mean."""
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    values_array = Tensor.convert_to_tensor(values)
    index_array = Tensor.convert_to_tensor(index)
    dim_size_int = _safe_int_conversion(dim_size)
    
    # First compute sum
    sum_result = scatter_op(values_array, index_array, dim_size_int, axis, "add")
    
    # Then compute count
    ones = mx.ones_like(values_array)
    count = scatter_op(ones, index_array, dim_size_int, axis, "add")
    
    # Avoid division by zero
    count = mx.where(count == 0, mx.ones_like(count), count)
    
    # Compute mean
    return mx.divide(sum_result, count)

def scatter_softmax(values: Any, index: Any, dim_size: int, axis: int = 0) -> mx.array:
    """Scatter values and compute softmax."""
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    values_array = Tensor.convert_to_tensor(values)
    index_array = Tensor.convert_to_tensor(index)
    dim_size_int = _safe_int_conversion(dim_size)
    
    # First compute max for numerical stability
    max_vals = scatter_op(values_array, index_array, dim_size_int, axis, "max")
    
    # Compute exp(x - max)
    exp_vals = mx.exp(values_array - max_vals)
    
    # Sum exp values
    sum_exp = scatter_op(exp_vals, index_array, dim_size_int, axis, "add")
    
    # Compute softmax
    return mx.divide(exp_vals, sum_exp)

# Alias for backward compatibility
slice = slice_tensor