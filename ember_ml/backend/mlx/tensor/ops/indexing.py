"""MLX tensor indexing operations."""

import mlx.core as mx

from typing import Union, Optional, Literal, TYPE_CHECKING
from builtins import slice as py_slice
from ember_ml.backend.mlx.types import (
    TensorLike, Shape, ShapeLike
)

def slice_tensor(tensor: TensorLike, starts: Shape, sizes: Shape) -> mx.array:
    """
    Extract a slice from a tensor.
    
    Args:
        data: Input tensor
        starts: Starting indices for each dimension
        sizes: Size of the slice in each dimension. A value of -1 means "all remaining elements in this dimension"
        
    Returns:
        Sliced tensor
    """
    
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    
    # Create a list of slice objects for each dimension
    slice_objects = []
    for i, (start, size) in enumerate(zip(starts, sizes)):
        # Convert to tensor to avoid precision-reducing casts
        start_tensor = mx.array(start, dtype=mx.int64)
        if size == -1:
            # -1 means "all remaining elements in this dimension"
            # Use Python's built-in slice function, not our slice_tensor function
            slice_obj = py_slice(start_tensor.item(), None)
            slice_objects.append(slice_obj)
        else:
            # Convert size to tensor to avoid precision-reducing casts
            size_tensor = mx.array(size, dtype=mx.int64)
            end_tensor = mx.add(start_tensor, size_tensor)
            # Use Python's built-in slice function, not our slice_tensor function
            slice_obj = py_slice(start_tensor.item(), end_tensor.item())
            slice_objects.append(slice_obj)
    
    # Extract the slice
    return tensor_array[tuple(slice_objects)]

# Alias for slice_tensor to match MLX naming
slice = slice_tensor


def gather(tensor: TensorLike, indices: TensorLike, axis: int = 0) -> mx.array:
    """
    Gather slices from a tensor along an axis.
    
    Args:
        data: Input tensor
        indices: Indices of slices to gather
        axis: Axis along which to gather
        
    Returns:
        Gathered tensor
    """
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    indices_array = Tensor.convert_to_tensor(indices)
    
    # Ensure indices are integers
    indices_int = indices_array.astype(mx.int32)

    # Use take operation for gathering
    return mx.take(tensor_array, indices_int, axis=axis)

def tensor_scatter_nd_update(tensor: TensorLike, indices: TensorLike, updates: TensorLike) -> mx.array:
    """
    Update tensor elements at given indices.
    
    Args:
        tensor: Input tensor to update
        indices: N-dimensional indices where to update values
        updates: Values to insert at the specified indices
        
    Returns:
        Updated tensor
    """
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
        indices_list = [indices_array.tolist()]
    else:
        # Handle multi-dimensional indices
        if len(indices_array.shape) > 1:
            # Convert each index to a list
            indices_list = [tuple(idx.tolist()) for idx in indices_array]
        else:
            indices_list = [indices_array.tolist()]
    
    # Update the tensor using our slice_update function
    for i, idx in enumerate(indices_list):
        result = slice_update(result, idx, updates_array[i])
    
    return result

def slice_update(tensor: TensorLike, slices: TensorLike, updates: Optional[TensorLike] = None) -> mx.array:
    """Update a tensor at specific indices."""
    # Convert inputs to MLX arrays
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    # Handle the case where slices is an integer (single index)
    if isinstance(slices, int) or (hasattr(slices, "item") and isinstance(slices.item(), int)):
        # Convert to a list with a single element
        indices_list = [int(slices)]
    else:
        # Convert slices to MLX array and then to list
        slices_array = Tensor.convert_to_tensor(slices)
        indices_list = slices_array.tolist()
        
        # Handle the case where tolist() returns an int
        if isinstance(indices_list, int):
            indices_list = [indices_list]
    
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

def scatter(data: TensorLike, indices: TensorLike, dim_size: Optional[Union[int, mx.array]] = None,
           aggr: Literal["add", "max", "min", "mean", "softmax"] = "add", axis: int = 0) -> mx.array:
    """
    Scatter values into a new tensor.
    
    Args:
        data: Source tensor containing values to scatter
        indices: Indices where to scatter the values
        dim_size: Size of the output tensor along the given axis. If None, uses the maximum index + 1
        aggr: Aggregation method to use for duplicate indices ("add", "max", "mean", "softmax", "min")
        axis: Axis along which to scatter
        
    Returns:
        Tensor with scattered values
    """
    # Convert inputs to MLX arrays
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    data_array = Tensor.convert_to_tensor(data)
    indices_array = Tensor.convert_to_tensor(indices)
    
    # Ensure indices are integers
    indices_int = indices_array.astype(mx.int32)
    
    # Handle dim_size
    if dim_size is None:
        # Determine maximum index + 1 for dimension size
        max_idx = mx.max(indices_int)
        computed_dim_size = int(max_idx.item()) + 1
    else:
        computed_dim_size = int(dim_size)
    
    # Choose appropriate scatter operation based on aggr
    if aggr == "add":
        return scatter_add(data_array, indices_int, computed_dim_size, axis)
    elif aggr == "max":
        return scatter_max(data_array, indices_int, computed_dim_size, axis)
    elif aggr == "min":
        return scatter_min(data_array, indices_int, computed_dim_size, axis)
    elif aggr == "mean":
        return scatter_mean(data_array, indices_int, computed_dim_size, axis)
    elif aggr == "softmax":
        return scatter_softmax(data_array, indices_int, computed_dim_size, axis)
    else:
        raise ValueError(f"Unsupported aggregation method: {aggr}")

def scatter_op(src: mx.array, index: mx.array, dim_size: int,
                axis: int, op: Literal["add", "max", "min", "softmax"]) -> mx.array:
    """Helper function for scatter operations."""
    # Get shape of output tensor
    output_shape = list(src.shape)
    if axis < 0:
        axis = len(output_shape) + axis
    output_shape[axis] = dim_size
    
    # Initialize output tensor based on operation
    if op == "add":
        out = mx.zeros(output_shape, dtype=src.dtype)
    elif op in ["max", "softmax"]:
        out = mx.full(output_shape, -float('inf'), dtype=src.dtype)
    elif op == "min":
        out = mx.full(output_shape, float('inf'), dtype=src.dtype)
    else:
        raise ValueError(f"Unknown operation: {op}")
    
    # Convert indices to integer lists
    index_list = index.tolist()
    if not isinstance(index_list, list):
        index_list = [index_list]
    
    # Create slices for indexing
    for i, idx in enumerate(index_list):
        # Create a slice for the specified axis with the index
        slices = [py_slice(None)] * len(output_shape)
        slices[axis] = idx
        
        # Get the value to scatter
        val_i = src[i]
        
        if op == "add":
            # Get current value and add
            current = out[tuple(slices)]
            # Create an index array with the correct shape for slice_update
            idx_array = mx.array([idx])
            axes = [axis]  # Use the correct axis for slicing
            out = mx.slice_update(out, val_i + current, idx_array, axes)
        elif op == "max":
            # Get current value and take max
            current = out[tuple(slices)]
            idx_array = mx.array([idx])
            axes = [axis]
            out = mx.slice_update(out, mx.maximum(current, val_i), idx_array, axes)
        elif op == "min":
            # Get current value and take min
            current = out[tuple(slices)]
            idx_array = mx.array([idx])
            axes = [axis]
            out = mx.slice_update(out, mx.minimum(current, val_i), idx_array, axes)
    
    return out

def scatter_add(src: TensorLike, index: TensorLike, dim_size: int, axis: int = 0) -> mx.array:
    """Scatter values using addition."""
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    src_array = Tensor.convert_to_tensor(src)
    index_array = Tensor.convert_to_tensor(index)
    return scatter_op(src_array, index_array, int(dim_size), axis, "add")

def scatter_max(src: TensorLike, index: TensorLike, dim_size: int, axis: int = 0) -> mx.array:
    """Scatter values using maximum."""
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    src_array = Tensor.convert_to_tensor(src)
    index_array = Tensor.convert_to_tensor(index)
    return scatter_op(src_array, index_array, int(dim_size), axis, "max")

def scatter_min(src: TensorLike, index: TensorLike, dim_size: int, axis: int = 0) -> mx.array:
    """Scatter values using minimum."""
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    src_array = Tensor.convert_to_tensor(src)
    index_array = Tensor.convert_to_tensor(index)
    return scatter_op(src_array, index_array, int(dim_size), axis, "min")

def scatter_mean(values: TensorLike, index: TensorLike, dim_size: int, axis: int = 0) -> mx.array:
    """Scatter values and compute mean."""
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    values_array = Tensor.convert_to_tensor(values)
    index_array = Tensor.convert_to_tensor(index)
    dim_size_int = int(dim_size)
    
    # First compute sum
    sum_result = scatter_op(values_array, index_array, dim_size_int, axis, "add")
    
    # Then compute count
    ones = mx.ones_like(values_array)
    count = scatter_op(ones, index_array, dim_size_int, axis, "add")
    
    # Avoid division by zero
    count = mx.where(count == 0, mx.ones_like(count), count)
    
    # Compute mean
    return mx.divide(sum_result, count)

def scatter_softmax(values: TensorLike, index: TensorLike, dim_size: int, axis: int = 0) -> mx.array:
    """Scatter values and compute softmax."""
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    values_array = Tensor.convert_to_tensor(values)
    index_array = Tensor.convert_to_tensor(index)
    dim_size_int = int(dim_size)
    
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