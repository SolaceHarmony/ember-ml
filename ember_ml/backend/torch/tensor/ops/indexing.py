"""PyTorch tensor indexing operations."""

import torch
from typing import Union, Optional, Sequence, Any, List, Tuple, Literal

# Import utility functions
from ember_ml.backend.torch.tensor.ops.utility import convert_to_tensor

# Type aliases
Shape = Sequence[int]
TensorLike = Any

def slice_tensor(data: TensorLike, starts: Sequence[int], sizes: Sequence[int]) -> torch.Tensor:
    """
    Extract a slice from a tensor.
    
    Args:
        data: Input tensor
        starts: Starting indices for each dimension
        sizes: Size of the slice in each dimension. A value of -1 means "all remaining elements in this dimension"
        
    Returns:
        Sliced tensor
    """
    tensor_torch = convert_to_tensor(data)
    
    # Create a list of slice objects for each dimension
    slice_objects = []
    for i, (start, size) in enumerate(zip(starts, sizes)):
        # Convert to tensor to avoid precision-reducing casts
        start_tensor = torch.tensor(start, dtype=torch.long)
        
        if size == -1:
            # -1 means "all remaining elements in this dimension"
            slice_objects.append(slice(start_tensor.item(), None))
        else:
            # Convert size to tensor to avoid precision-reducing casts
            size_tensor = torch.tensor(size, dtype=torch.long)
            end_tensor = torch.add(start_tensor, size_tensor)
            slice_objects.append(slice(start_tensor.item(), end_tensor.item()))
    
    # Extract the slice
    return tensor_torch[tuple(slice_objects)]

# Alias for slice_tensor to match MLX naming
slice = slice_tensor

def slice_update(data: TensorLike, slices: Any, updates: Any) -> torch.Tensor:
    """
    Update a tensor at specific indices.
    
    Args:
        data: Input tensor to update
        slices: List or tuple of slice objects or indices
        updates: Values to insert at the specified indices
        
    Returns:
        Updated tensor
    """
    tensor_torch = convert_to_tensor(data)
    updates_torch = convert_to_tensor(updates)
    
    # Create a copy of the input tensor
    result = tensor_torch.clone()
    
    # Update the tensor at the specified indices
    result[slices] = updates_torch
    
    return result

def gather(data: TensorLike, indices: Any, axis: int = 0) -> torch.Tensor:
    """
    Gather slices from a tensor along an axis.
    
    Args:
        data: Input tensor
        indices: Indices of slices to gather
        axis: Axis along which to gather
        
    Returns:
        Gathered tensor
    """
    tensor_torch = convert_to_tensor(data)
    indices_torch = convert_to_tensor(indices)
    
    # Convert indices to long
    indices_torch = indices_torch.long()
    
    return torch.index_select(tensor_torch, axis, indices_torch)

def tensor_scatter_nd_update(data: TensorLike, indices: Any, updates: Any) -> torch.Tensor:
    """
    Updates values of a tensor at specified indices.
    
    Args:
        data: Input tensor to update
        indices: Indices at which to update values (N-dimensional indices)
        updates: Values to insert at the specified indices
        
    Returns:
        Updated tensor
    """
    # Create a copy of the tensor
    tensor_torch = convert_to_tensor(data)
    indices_torch = convert_to_tensor(indices)
    updates_torch = convert_to_tensor(updates)
    
    # Ensure indices are integers
    indices_torch = indices_torch.long()
    
    # Create a copy of the tensor
    result = tensor_torch.clone()
    
    # Iterate over the indices and apply updates
    for i in range(indices_torch.shape[0]):
        # Extract indices for this update
        idx = []
        for j in range(indices_torch.shape[1]):
            # Get each dimension's index value
            idx.append(indices_torch[i, j].item())
        
        # Apply the update directly using tuple indexing
        result[tuple(idx)] = updates_torch[i]
    
    return result

def scatter(data: TensorLike, indices: Any, dim_size: Optional[int] = None,
            aggr: Literal["add", "max", "mean", "softmax", "min"] = "add", axis: int = 0) -> torch.Tensor:
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
    tensor_torch = convert_to_tensor(data)
    indices_torch = convert_to_tensor(indices).long()
    
    # Determine the output size
    if dim_size is None:
        dim_size = indices_torch.max().item() + 1
    
    # Create output shape
    output_shape = list(tensor_torch.shape)
    output_shape[axis] = dim_size
    
    # Initialize output tensor with zeros
    output = torch.zeros(output_shape, dtype=tensor_torch.dtype, device=tensor_torch.device)
    
    # Apply the appropriate scatter operation based on aggr
    if aggr == "add":
        return scatter_add(data, indices, dim_size, axis)
    elif aggr == "max":
        return scatter_max(data, indices, dim_size, axis)
    elif aggr == "min":
        return scatter_min(data, indices, dim_size, axis)
    elif aggr == "mean":
        return scatter_mean(data, indices, dim_size, axis)
    elif aggr == "softmax":
        return scatter_softmax(data, indices, dim_size, axis)
    else:
        raise ValueError(f"Unsupported aggregation method: {aggr}")

def scatter_add(data: TensorLike, indices: Any, dim_size: Optional[int] = None, axis: int = 0) -> torch.Tensor:
    """
    Scatter-add operation: adds values from data at the indices in the output tensor.
    
    Args:
        data: Source tensor containing values to scatter
        indices: Indices where to scatter the values
        dim_size: Size of the output tensor along the given axis. If None, uses the maximum index + 1
        axis: Axis along which to scatter
        
    Returns:
        Tensor with scattered values (added)
    """
    tensor_torch = convert_to_tensor(data)
    indices_torch = convert_to_tensor(indices).long()
    
    # Determine the output size
    if dim_size is None:
        dim_size = indices_torch.max().item() + 1
    
    # Create output shape
    output_shape = list(tensor_torch.shape)
    output_shape[axis] = dim_size
    
    # Initialize output tensor with zeros
    output = torch.zeros(output_shape, dtype=tensor_torch.dtype, device=tensor_torch.device)
    
    # Use torch.scatter_add_ for the operation
    return output.scatter_add_(axis, indices_torch, tensor_torch)

def scatter_max(data: TensorLike, indices: Any, dim_size: Optional[int] = None, axis: int = 0) -> torch.Tensor:
    """
    Scatter-max operation: takes the maximum of values from data at the indices in the output tensor.
    
    Args:
        data: Source tensor containing values to scatter
        indices: Indices where to scatter the values
        dim_size: Size of the output tensor along the given axis. If None, uses the maximum index + 1
        axis: Axis along which to scatter
        
    Returns:
        Tensor with scattered values (maximum)
    """
    tensor_torch = convert_to_tensor(data)
    indices_torch = convert_to_tensor(indices).long()
    
    # Determine the output size
    if dim_size is None:
        dim_size = indices_torch.max().item() + 1
    
    # Create output shape
    output_shape = list(tensor_torch.shape)
    output_shape[axis] = dim_size
    
    # Initialize output tensor with minimum value
    output = torch.full(output_shape, float('-inf'), dtype=tensor_torch.dtype, device=tensor_torch.device)
    
    # Use torch.scatter to perform the max operation
    result, _ = output.scatter_(axis, indices_torch, tensor_torch, reduce='amax')
    
    # Replace -inf with 0
    result = torch.where(result == float('-inf'), torch.zeros_like(result), result)
    
    return result

def scatter_min(data: TensorLike, indices: Any, dim_size: Optional[int] = None, axis: int = 0) -> torch.Tensor:
    """
    Scatter-min operation: takes the minimum of values from data at the indices in the output tensor.
    
    Args:
        data: Source tensor containing values to scatter
        indices: Indices where to scatter the values
        dim_size: Size of the output tensor along the given axis. If None, uses the maximum index + 1
        axis: Axis along which to scatter
        
    Returns:
        Tensor with scattered values (minimum)
    """
    tensor_torch = convert_to_tensor(data)
    indices_torch = convert_to_tensor(indices).long()
    
    # Determine the output size
    if dim_size is None:
        dim_size = indices_torch.max().item() + 1
    
    # Create output shape
    output_shape = list(tensor_torch.shape)
    output_shape[axis] = dim_size
    
    # Initialize output tensor with maximum value
    output = torch.full(output_shape, float('inf'), dtype=tensor_torch.dtype, device=tensor_torch.device)
    
    # Use torch.scatter to perform the min operation
    result, _ = output.scatter_(axis, indices_torch, tensor_torch, reduce='amin')
    
    # Replace inf with 0
    result = torch.where(result == float('inf'), torch.zeros_like(result), result)
    
    return result

def scatter_mean(data: TensorLike, indices: Any, dim_size: Optional[int] = None, axis: int = 0) -> torch.Tensor:
    """
    Scatter-mean operation: computes the mean of values from data at the indices in the output tensor.
    
    Args:
        data: Source tensor containing values to scatter
        indices: Indices where to scatter the values
        dim_size: Size of the output tensor along the given axis. If None, uses the maximum index + 1
        axis: Axis along which to scatter
        
    Returns:
        Tensor with scattered values (mean)
    """
    tensor_torch = convert_to_tensor(data)
    indices_torch = convert_to_tensor(indices).long()
    
    # Determine the output size
    if dim_size is None:
        dim_size = indices_torch.max().item() + 1
    
    # Create output shape
    output_shape = list(tensor_torch.shape)
    output_shape[axis] = dim_size
    
    # First, compute the sum
    sum_output = torch.zeros(output_shape, dtype=tensor_torch.dtype, device=tensor_torch.device)
    sum_output = sum_output.scatter_add_(axis, indices_torch, tensor_torch)
    
    # Then, count the number of values added to each position
    count_output = torch.zeros(output_shape, dtype=tensor_torch.dtype, device=tensor_torch.device)
    ones = torch.ones_like(tensor_torch)
    count_output = count_output.scatter_add_(axis, indices_torch, ones)
    
    # Compute the mean (avoiding division by zero)
    count_output = torch.where(count_output == 0, torch.ones_like(count_output), count_output)
    mean_output = sum_output / count_output
    
    return mean_output

def scatter_softmax(data: TensorLike, indices: Any, dim_size: Optional[int] = None, axis: int = 0) -> torch.Tensor:
    """
    Scatter-softmax operation: applies softmax to values from data at the indices in the output tensor.
    
    Args:
        data: Source tensor containing values to scatter
        indices: Indices where to scatter the values
        dim_size: Size of the output tensor along the given axis. If None, uses the maximum index + 1
        axis: Axis along which to scatter
        
    Returns:
        Tensor with scattered values (softmax)
    """
    tensor_torch = convert_to_tensor(data)
    indices_torch = convert_to_tensor(indices).long()
    
    # Determine the output size
    if dim_size is None:
        dim_size = indices_torch.max().item() + 1
    
    # Create output shape
    output_shape = list(tensor_torch.shape)
    output_shape[axis] = dim_size
    
    # First, compute the max for numerical stability
    max_output = torch.full(output_shape, float('-inf'), dtype=tensor_torch.dtype, device=tensor_torch.device)
    max_output, _ = max_output.scatter_(axis, indices_torch, tensor_torch, reduce='amax')
    
    # Subtract max for numerical stability
    shifted_data = tensor_torch - max_output.gather(axis, indices_torch)
    
    # Compute exp
    exp_data = torch.exp(shifted_data)
    
    # Sum the exp values
    sum_output = torch.zeros(output_shape, dtype=tensor_torch.dtype, device=tensor_torch.device)
    sum_output = sum_output.scatter_add_(axis, indices_torch, exp_data)
    
    # Compute softmax
    softmax_data = exp_data / sum_output.gather(axis, indices_torch)
    
    # Scatter the softmax values
    result = torch.zeros(output_shape, dtype=tensor_torch.dtype, device=tensor_torch.device)
    result = result.scatter_add_(axis, indices_torch, softmax_data)
    
    return result