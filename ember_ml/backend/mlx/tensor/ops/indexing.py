"""MLX tensor indexing operations."""

import mlx.core as mx
from typing import Any, Sequence, Union, Optional, List, Tuple

def slice_tensor(tensor_obj, tensor: Any, starts: Sequence[int], sizes: Sequence[int]) -> mx.array:
    """
    Extract a slice from a tensor.
    
    Args:
        tensor_obj: MLXTensor instance
        tensor: Input tensor
        starts: Starting indices for each dimension
        sizes: Size of the slice in each dimension
        
    Returns:
        Sliced tensor
    """
    # Convert input to MLX array
    tensor_array = tensor_obj.convert_to_tensor(tensor)
    
    # Convert to MLX arrays
    starts_mx = mx.array(starts, dtype=mx.int32)
    sizes_mx = mx.array(sizes, dtype=mx.int32)
    sizes_mx_list = [int(x) for x in sizes_mx]
    # Create axes as a list of integers
    axes = list(range(len(starts)))
    
    # Use MLX's slice function
    return mx.slice(tensor_array, starts_mx, axes, sizes_mx_list)

def gather(tensor_obj, tensor: Any, indices: Any, axis: int = 0) -> mx.array:
    """
    Gather slices from a tensor along an axis.
    
    Args:
        tensor_obj: MLXTensor instance
        tensor: Input tensor
        indices: Indices of slices to gather
        axis: Axis along which to gather
        
    Returns:
        Gathered tensor
    """
    # Convert inputs to MLX arrays
    tensor_array = tensor_obj.convert_to_tensor(tensor)
    indices_array = tensor_obj.convert_to_tensor(indices)
    
    axis = int(axis)
    # Ensure indices are integers
    indices_int = mx.array(indices_array, dtype=mx.int32)
    
    # Use take along axis
    return mx.take(tensor_array, indices_int, axis=axis)

def tensor_scatter_nd_update(tensor_obj, tensor: Any, indices: Any, updates: Any) -> mx.array:
    """
    Updates values of a tensor at specified indices.
    
    Args:
        tensor_obj: MLXTensor instance
        tensor: Input tensor to update
        indices: Indices at which to update values (N-dimensional indices)
        updates: Values to insert at the specified indices
        
    Returns:
        Updated tensor
    """
    # Convert inputs to MLX arrays
    tensor_array = tensor_obj.convert_to_tensor(tensor)
    indices_array = tensor_obj.convert_to_tensor(indices)
    updates_array = tensor_obj.convert_to_tensor(updates)
    
    # Ensure indices are integers
    indices_int = mx.array(indices_array, dtype=mx.int32)
    
    # Create a copy of the tensor
    result = mx.array(tensor_array)
    
    # Iterate over the indices and apply updates
    for i in range(indices_int.shape[0]):
        # Extract indices for this update
        start_indices = mx.array(indices_int[i].tolist(), dtype=mx.int32)
        
        # Create axes as a list of integers
        axes = list(range(len(start_indices)))
        
        # Create sizes for the update (all ones)
        sizes = mx.ones(len(start_indices), dtype=mx.int32)
        
        # Apply the update using slice_update
        result = mx.slice_update(result, updates_array[i], start_indices, axes)
    
    return result

from typing import Optional

def scatter(tensor_obj, data: Any, indices: Any, dim_size: Optional[int] = None,
            aggr: str = "add", axis: int = 0) -> mx.array:
    """
    Scatter values from data into a new tensor of size dim_size along the given axis.
    
    Args:
        tensor_obj: MLXTensor instance
        data: Source tensor containing values to scatter
        indices: Indices where to scatter the values
        dim_size: Size of the output tensor along the given axis. If None, uses the maximum index + 1
        aggr: Aggregation method to use for duplicate indices ("add", "max", "mean", "softmax", "min")
        axis: Axis along which to scatter
        
    Returns:
        Tensor with scattered values
    """
    # Convert inputs to MLX arrays
    data_array = tensor_obj.convert_to_tensor(data)
    indices_array = tensor_obj.convert_to_tensor(indices)
    
    axis = int(axis)
    # Ensure indices are integers
    indices_int = mx.array(indices_array, dtype=mx.int32)
    
    # Determine output size if not provided
    if dim_size is None:
        # Get the maximum index and add 1
        max_idx = mx.max(indices_int)
        dim_size = int(max_idx) + 1

    if aggr == "add":
        return scatter_add(data_array, indices_int, dim_size, axis)
    elif aggr == "max":
        return scatter_max(data_array, indices_int, dim_size, axis)
    elif aggr == "min":
        return scatter_min(data_array, indices_int, dim_size, axis)
    elif aggr == "mean":
        return scatter_mean(data_array, indices_int, dim_size, axis)
    elif aggr == "softmax":
        return scatter_softmax(data_array, indices_int, dim_size, axis)
    else:
        raise ValueError(f"Unknown aggregation method: {aggr}")

def scatter_add(src: mx.array, index: mx.array, dim_size: int, axis: int = 0) -> mx.array:
    """
    Scatter values from src into a new tensor of size dim_size along the given axis using addition.
    
    Args:
        src: Source tensor containing values to scatter
        index: Indices where to scatter the values
        dim_size: Size of the output tensor along the given axis
        axis: Axis along which to scatter
        
    Returns:
        Tensor with scattered summed values
    """
    out = mx.zeros((dim_size,), dtype=src.dtype)
    for i in range(index.shape[0]):
        idx = int(index[i])
        start_indices = mx.array([idx], dtype=mx.int32)
        val = out[idx] + src[i]
        out = mx.slice_update(out, val, start_indices, [0])
    return out

def scatter_max(src: mx.array, index: mx.array, dim_size: int, axis: int = 0) -> mx.array:
    """
    Scatter values from src into a new tensor of size dim_size along the given axis using maximum.
    
    Args:
        src: Source tensor containing values to scatter
        index: Indices where to scatter the values
        dim_size: Size of the output tensor along the given axis
        axis: Axis along which to scatter
        
    Returns:
        Tensor with scattered maximum values
    """
    out = mx.full((dim_size,), -mx.inf, dtype=src.dtype)
    for i in range(index.shape[0]):
        idx = int(index[i])
        start_indices = mx.array([idx], dtype=mx.int32)
        val = mx.max(out[idx], src[i])
        out = mx.slice_update(out, val, start_indices, [0])
    return out

def scatter_min(src: mx.array, index: mx.array, dim_size: int, axis: int = 0) -> mx.array:
    """
    Scatter values from src into a new tensor of size dim_size along the given axis using minimum.
    
    Args:
        src: Source tensor containing values to scatter
        index: Indices where to scatter the values
        dim_size: Size of the output tensor along the given axis
        axis: Axis along which to scatter
        
    Returns:
        Tensor with scattered minimum values
    """
    out = mx.full((dim_size,), mx.inf, dtype=src.dtype)
    for i in range(index.shape[0]):
        idx = int(index[i])
        start_indices = mx.array([idx], dtype=mx.int32)
        val = mx.min(out[idx], src[i])
        out = mx.slice_update(out, val, start_indices, [0])
    return out

def scatter_mean(tensor_obj, values: Any, index: Any, dim_size: int, axis: int = 0) -> mx.array:
    """
    Scatter values and compute the mean at the specified indices.
    
    Args:
        tensor_obj: MLXTensor instance
        values: Values to scatter
        index: Indices where to scatter the values
        dim_size: Size of the output tensor along the given axis
        axis: Axis along which to scatter
        
    Returns:
        Tensor with scattered mean values
    """
    # Convert inputs to MLX arrays
    values_array = tensor_obj.convert_to_tensor(values)
    index_array = tensor_obj.convert_to_tensor(index)
    
    axis = int(axis)
    # First compute the sum using scatter with "add" aggregation
    sum_result = scatter(tensor_obj, values_array, index_array, dim_size, "add", axis)
    
    # Compute the count of values for each index
    counts = mx.zeros((dim_size,), dtype=mx.int32)
    
    # Count occurrences
    for i in range(index_array.shape[0]):
        # Get the index for this update
        idx = int(index_array[i])
        
        # Add 1 to the count
        counts[idx] += 1
    
    # Avoid division by zero
    counts = mx.where(counts < 1, 1, counts)
    
    # Compute the mean
    return mx.divide(sum_result, counts)

def scatter_softmax(tensor_obj, values: Any, index: Any, dim_size: int, axis: int = 0) -> mx.array:
    """
    Scatter values and compute the softmax at the specified indices.
    
    Args:
        tensor_obj: MLXTensor instance
        values: Values to scatter
        index: Indices where to scatter the values
        dim_size: Size of the output tensor along the given axis
        axis: Axis along which to scatter
        
    Returns:
        Tensor with scattered softmax values
    """
    # Convert inputs to MLX arrays
    values_array = tensor_obj.convert_to_tensor(values)
    index_array = tensor_obj.convert_to_tensor(index)
    
    axis = int(axis)
    # First compute the max for each index
    max_result = scatter(tensor_obj, values_array, index_array, dim_size, "max", axis)
    
    # Compute exp(values - max)
    exp_values = mx.exp(mx.subtract(values_array, max_result))
    
    # Compute the sum of exp values for each index
    sum_result = scatter(tensor_obj, exp_values, index_array, dim_size, "add", axis)
    
    # Compute softmax
    return mx.divide(exp_values, sum_result)

def slice_update(tensor_obj, tensor: Any, slices: Any, updates: Any) -> mx.array:
    """
    Update a tensor at specific indices.
    
    Args:
        tensor_obj: MLXTensor instance
        tensor: Input tensor to update
        slices: List of slice indices
        updates: Values to insert at the specified indices
        
    Returns:
        Updated tensor
    """
    # Convert inputs to MLX arrays
    tensor_array = tensor_obj.convert_to_tensor(tensor)
    updates_array = tensor_obj.convert_to_tensor(updates)
    
    # Create a copy of the input tensor
    result = mx.array(tensor_array)
    
    # Convert slices to start_indices and axes
    start_indices = mx.array(list(map(int, slices)), dtype=mx.int32)
    axes = list(range(len(slices)))
    
    # Apply the update using slice_update
    return mx.slice_update(result, updates_array, start_indices, axes)