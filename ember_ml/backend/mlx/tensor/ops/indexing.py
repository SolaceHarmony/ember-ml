"""MLX tensor indexing operations."""

from typing import Any, Literal, Optional, Sequence, Union

import mlx.core as mx

from ember_ml.backend.mlx.tensor.tensor import MLXTensor
from ember_ml.backend.mlx.config import TensorLike, Shape

# Define the allowed aggregation methods
ScatterAggregations = Literal["add", "max", "mean", "softmax", "min"]

# Create an instance of MLXTensor for conversion
Tensor = MLXTensor()

def slice_tensor(tensor: TensorLike, starts: Shape, sizes: Shape) -> mx.array:
    """
    Extract a slice from a tensor.
    
    Args:
        tensor: Input tensor
        starts: Starting indices for each dimension
        sizes: Size of the slice in each dimension
        
    Returns:
        Sliced tensor
    """
    # Convert input to MLX array
    tensor_array = Tensor.convert_to_tensor(tensor)
    
    # Convert starts to MLX array
    if isinstance(starts, (list, tuple)):
        starts_mx = mx.array(starts, dtype=mx.int32)
    else:
        # Handle scalar case
        starts_mx = mx.array([starts], dtype=mx.int32)

    # Create axes as a tuple of integers
    axes = tuple(range(len(starts_mx)))
    
    # Use MLX's slice function
    return mx.slice(tensor_array, starts_mx, axes, sizes)

def gather(tensor: TensorLike, indices: TensorLike, axis: int = 0) -> mx.array:
    """
    Gather slices from a tensor along an axis.
    
    Args:
        tensor: Input tensor
        indices: Indices of slices to gather
        axis: Axis along which to gather
        
    Returns:
        Gathered tensor
    """
    # Convert inputs to MLX arrays
    tensor_array = Tensor.convert_to_tensor(tensor)
    indices_array = Tensor.convert_to_tensor(indices)
    
    # Convert axis to MLX array
    axis_value = mx.array(axis, dtype=mx.int32)
    
    # Ensure indices are integers
    indices_int = mx.array(indices_array, dtype=mx.int32)
    
    # Use take along axis
    return mx.take(tensor_array, indices_int, axis=axis_value)

def tensor_scatter_nd_update(tensor: TensorLike, indices: TensorLike, updates: TensorLike) -> mx.array:
    """
    Updates values of a tensor at specified indices.
    
    Args:
        tensor: Input tensor to update
        indices: Indices at which to update values (N-dimensional indices)
        updates: Values to insert at the specified indices
        
    Returns:
        Updated tensor
    """
    # Convert inputs to MLX arrays
    tensor_array = Tensor.convert_to_tensor(tensor)
    indices_array = Tensor.convert_to_tensor(indices)
    updates_array = Tensor.convert_to_tensor(updates)
    
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

def scatter(data: TensorLike, indices: TensorLike,
            dim_size: Optional[Union[int, mx.array]] = None,
            aggr: str = "add", axis: Union[int, mx.array] = 0) -> mx.array:
    """
    Scatter values from data into a new tensor of size dim_size along the given axis.
    
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
    data_array = Tensor.convert_to_tensor(data)
    indices_array = Tensor.convert_to_tensor(indices)
    
    # Convert axis to MLX array if it's not already
    if not isinstance(axis, mx.array):
        axis_value = mx.array(axis, dtype=mx.int32)
    else:
        axis_value = axis
    
    # Ensure indices are integers
    indices_int = mx.array(indices_array, dtype=mx.int32)
    
    # Determine output size if not provided
    if dim_size is None:
        # Use shape instead of max + 1 computation
        computed_dim_size = mx.array(indices_array.shape[0], dtype=mx.int32)
    else:
        # Ensure dim_size is an mx.array
        computed_dim_size = mx.array(dim_size, dtype=mx.int32) if not isinstance(dim_size, mx.array) else dim_size

    if aggr == "add":
        return scatter_add(data_array, indices_int, computed_dim_size, axis_value)
    elif aggr == "max":
        return scatter_max(data_array, indices_int, computed_dim_size, axis_value)
    elif aggr == "min":
        return scatter_min(data_array, indices_int, computed_dim_size, axis_value)
    elif aggr == "mean":
        return scatter_mean(data_array, indices_int, computed_dim_size, axis_value)
    elif aggr == "softmax":
        return scatter_softmax(data_array, indices_int, computed_dim_size, axis_value)
    else:
        raise ValueError(f"Unknown aggregation method: {aggr}")

def scatter_add(src: TensorLike, index: TensorLike, 
                dim_size: Union[int, mx.array], axis: Union[int, mx.array] = 0) -> mx.array:
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
    # Convert inputs to MLX arrays
    src_array = Tensor.convert_to_tensor(src)
    index_array = Tensor.convert_to_tensor(index)
    
    # Convert dim_size to mx.array if it's not already
    if not isinstance(dim_size, mx.array):
        dim_size_tensor = mx.array(dim_size, dtype=mx.int32)
    else:
        dim_size_tensor = dim_size
        
    # Initialize output tensor with zeros
    out = mx.zeros((dim_size_tensor,), dtype=src_array.dtype)
    
    for i in range(index_array.shape[0]):
        # Get the index for this update
        idx_tensor = mx.array(index_array[i], dtype=mx.int32)
        
        # Use the tensor directly for indexing
        start_indices = mx.array([idx_tensor], dtype=mx.int32)
        
        # Use mx.add instead of direct + operator
        val = mx.add(out[idx_tensor], src_array[i])
        
        # Update the output tensor
        out = mx.slice_update(out, val, start_indices, [0])
    return out

def scatter_max(src: TensorLike, index: TensorLike,
                dim_size: Union[int, mx.array], axis: Union[int, mx.array] = 0) -> mx.array:
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
    # Convert inputs to MLX arrays
    src_array = Tensor.convert_to_tensor(src)
    index_array = Tensor.convert_to_tensor(index)
    
    # Convert dim_size to mx.array if it's not already
    if not isinstance(dim_size, mx.array):
        dim_size_tensor = mx.array(dim_size, dtype=mx.int32)
    else:
        dim_size_tensor = dim_size
        
    # Initialize output tensor with negative infinity
    out = mx.full((dim_size_tensor,), -mx.inf, dtype=src_array.dtype)
    
    for i in range(index_array.shape[0]):
        # Get the index for this update
        idx_tensor = mx.array(index_array[i], dtype=mx.int32)
        
        # Use the tensor directly for indexing
        start_indices = mx.array([idx_tensor], dtype=mx.int32)
        
        # Compute the maximum
        val = mx.max(out[idx_tensor], src_array[i])
        
        # Update the output tensor
        out = mx.slice_update(out, val, start_indices, [0])
    return out

def scatter_min(src: TensorLike, index: TensorLike, 
                dim_size: Union[int, mx.array], axis: Union[int, mx.array] = 0) -> mx.array:
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
    # Convert inputs to MLX arrays
    src_array = Tensor.convert_to_tensor(src)
    index_array = Tensor.convert_to_tensor(index)
    
    # Convert dim_size to mx.array if it's not already
    if not isinstance(dim_size, mx.array):
        dim_size_tensor = mx.array(dim_size, dtype=mx.int32)
    else:
        dim_size_tensor = dim_size
        
    # Initialize output tensor with infinity
    out = mx.full((dim_size_tensor,), mx.inf, dtype=src_array.dtype)
    
    for i in range(index_array.shape[0]):
        # Get the index for this update
        idx_tensor = mx.array(index_array[i], dtype=mx.int32)
        
        # Use the tensor directly for indexing
        start_indices = mx.array([idx_tensor], dtype=mx.int32)
        
        # Compute the minimum
        val = mx.min(out[idx_tensor], src_array[i])
        
        # Update the output tensor
        out = mx.slice_update(out, val, start_indices, [0])
    return out

def scatter_mean(values: TensorLike, index: TensorLike, 
                 dim_size: Union[int, mx.array], axis: Union[int, mx.array] = 0) -> mx.array:
    """
    Scatter values and compute the mean at the specified indices.
    
    Args:
        values: Values to scatter
        index: Indices where to scatter the values
        dim_size: Size of the output tensor along the given axis
        axis: Axis along which to scatter
        
    Returns:
        Tensor with scattered mean values
    """
    # Convert inputs to MLX arrays
    values_array = Tensor.convert_to_tensor(values)
    index_array = Tensor.convert_to_tensor(index)
    
    # First compute the sum using scatter with "add" aggregation
    sum_result = scatter(values_array, index_array, dim_size, aggr="add", axis=axis)
    
    # Compute the count of values for each index
    counts = mx.zeros((dim_size,), dtype=mx.int32)
    
    # Count occurrences
    for i in range(index_array.shape[0]):
        # Get the index for this update
        idx_tensor = mx.array(index_array[i], dtype=mx.int32)
        
        # Create start indices for the update
        start_indices = mx.array([idx_tensor], dtype=mx.int32)
        
        # Create value to add (1)
        one = mx.array(1, dtype=mx.int32)
        
        # Get current count
        current_count = counts[idx_tensor]
        
        # Add 1 to the count
        new_count = mx.add(current_count, one)
        
        # Update counts
        counts = mx.slice_update(counts, new_count, start_indices, [0])
    
    # Avoid division by zero
    one_tensor = mx.array(1, dtype=mx.int32)
    counts = mx.where(counts < one_tensor, one_tensor, counts)
    
    # Compute the mean
    return mx.divide(sum_result, counts)

def scatter_softmax(values: TensorLike, index: TensorLike,
                    dim_size: Union[int, mx.array], axis: Union[int, mx.array] = 0) -> mx.array:
    """
    Scatter values and compute the softmax at the specified indices.
    
    Args:
        values: Values to scatter
        index: Indices where to scatter the values
        dim_size: Size of the output tensor along the given axis
        axis: Axis along which to scatter
        
    Returns:
        Tensor with scattered softmax values
    """
    # Convert inputs to MLX arrays
    values_array = Tensor.convert_to_tensor(values)
    index_array = Tensor.convert_to_tensor(index)
    
    # Convert axis to MLX array if it's not already
    if not isinstance(axis, mx.array):
        axis_value = mx.array(axis, dtype=mx.int32)
    else:
        axis_value = axis
    
    # First compute the max for each index
    max_result = scatter(values_array, index_array, dim_size, aggr="max", axis=axis)
    
    # Compute exp(values - max)
    exp_values = mx.exp(mx.subtract(values_array, max_result))
    
    # Compute the sum of exp values for each index
    # Convert axis to mx.array if it's not already
    if not isinstance(axis, mx.array):
        axis_tensor = mx.array(axis, dtype=mx.int32)
    else:
        axis_tensor = axis
        
    # Use axis_tensor instead of axis
    sum_result = scatter(exp_values, index_array, dim_size, aggr="add", axis=axis_tensor)
    
    # Compute softmax
    return mx.divide(exp_values, sum_result)

def slice_update(tensor: TensorLike, slices: TensorLike, updates: Optional[TensorLike] = None) -> mx.array:
    """
    Update a tensor at specific indices.
    
    Args:
        tensor: Input tensor to update
        slices: Index or list of indices
        updates: Values to insert at the specified indices. If None, this function
                acts as a getter and returns the slice of the tensor.
        
    Returns:
        Updated tensor or sliced tensor if updates is None
    """
    # Convert inputs to MLX arrays
    tensor_array = Tensor.convert_to_tensor(tensor)
    
    # If updates is None, this is a get operation, not an update
    if updates is None:
        # Create a copy of the input tensor
        result = mx.array(tensor_array)
        
        # Convert slices to start_indices and axes
        # Use mx.array directly to handle different types of indices
        start_indices = mx.array(slices, dtype=mx.int32)
        
        # Create axes as a list of integers
        axes = list(range(start_indices.size))
        
        # Create sizes array with all ones using MLX
        # Use mx.ones directly with the length of axes
        sizes_array = mx.ones(len(axes), dtype=mx.int32)
        
        # Use mx.slice to get the slice with the shape of sizes_array as slice_size
        return mx.slice(result, start_indices, axes, sizes_array.shape)
    
    # Otherwise, this is an update operation
    updates_array = Tensor.convert_to_tensor(updates)
    
    # Create a copy of the input tensor
    result = mx.array(tensor_array)
    
    # Convert slices to start_indices and axes
    # Use mx.array directly to handle different types of indices
    start_indices = mx.array(slices, dtype=mx.int32)
    axes = list(range(start_indices.size))
    
    # Apply the update using slice_update
    return mx.slice_update(result, updates_array, start_indices, axes)

# Monkey-patch slice_tensor as slice for compatibility
slice = slice_tensor