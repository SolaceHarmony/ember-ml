"""PyTorch tensor indexing operations."""

import torch

from typing import Sequence, Optional, Literal, Union
from builtins import slice as py_slice
from ember_ml.backend.torch.types import (
    TensorLike, Shape
)

def slice_tensor(tensor: TensorLike, starts: Shape, sizes: Shape) -> torch.Tensor:
    """
    Extract a slice from a tensor.
    
    Args:
        data: Input tensor
        starts: Starting indices for each dimension
        sizes: Size of the slice in each dimension. A value of -1 means "all remaining elements in this dimension"
        
    Returns:
        Sliced tensor
    """
    
    # Convert input to Torch array
    from ember_ml.backend.torch.tensor import TorchTensor
    Tensor = TorchTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    
    # Create a list of slice objects for each dimension
    slice_objects = []
    for i, (start, size) in enumerate(zip(starts, sizes)):
        # Convert to tensor to avoid precision-reducing casts
        start_tensor = torch.tensor(start, dtype=torch.long)
        if size == -1:
            # -1 means "all remaining elements in this dimension"
            # Use Python's built-in slice function, not our slice_tensor function
            slice_obj = py_slice(start_tensor.item(), None)
            slice_objects.append(slice_obj)
        else:
            # Convert size to tensor to avoid precision-reducing casts
            size_tensor = torch.tensor(size, dtype=torch.long)
            end_tensor = torch.add(start_tensor, size_tensor)
            # Use Python's built-in slice function, not our slice_tensor function
            slice_obj = py_slice(start_tensor.item(), end_tensor.item())
            slice_objects.append(slice_obj)
    
    # Extract the slice
    return tensor_array[tuple(slice_objects)]

# Alias for slice_tensor to match Torch naming
slice = slice_tensor

def slice_update(data: TensorLike, slices: TensorLike, updates: Optional[TensorLike] = None) -> torch.Tensor:
    """
    Update a tensor at specific indices.
    
    Args:
        data: Input tensor to update
        slices: List or tuple of slice objects or indices
        updates: Values to insert at the specified indices. If None, returns a slice of the tensor.
        
    Returns:
        Updated tensor or sliced tensor if updates is None
    """
    # Convert inputs to Torch arrays
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    tensor = tensor_ops.convert_to_tensor(data)
    
    # If updates is None, return a slice of the tensor
    if updates is None:
        return tensor[slices].clone()
    
    # Convert updates to tensor
    updates_tensor = tensor_ops.convert_to_tensor(updates)
    
    # Create a copy of the input tensor
    result = tensor.clone()
    
    # Update the tensor at the specified indices
    result[slices] = updates_tensor
    
    return result

def gather(tensor: TensorLike, indices: TensorLike, axis: int = 0) -> torch.Tensor:
    """
    Gather slices from a tensor along an axis.
    
    Args:
        data: Input tensor
        indices: Indices of slices to gather
        axis: Axis along which to gather
        
    Returns:
        Gathered tensor
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    Tensor = TorchTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    indices_array = Tensor.convert_to_tensor(indices)
    
    # Convert indices to long
    indices_array = indices_array.long()
    
    return torch.gather(tensor_array, axis, indices_array)
    

def tensor_scatter_nd_update(data: TensorLike, indices: TensorLike, updates: TensorLike) -> torch.Tensor:
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
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    tensor_torch = tensor_ops.convert_to_tensor(data)
    indices_torch = tensor_ops.convert_to_tensor(indices)
    updates_torch = tensor_ops.convert_to_tensor(updates)
    
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

def scatter(data: TensorLike, indices: TensorLike, dim_size: Optional[int] = None,
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
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    tensor_torch = tensor_ops.convert_to_tensor(data)
    indices_torch = tensor_ops.convert_to_tensor(indices).long()
    
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
        return scatter_add(tensor_torch, indices_torch, dim_size, axis)
    elif aggr == "max":
        return scatter_max(tensor_torch, indices_torch, dim_size, axis)
    elif aggr == "min":
        return scatter_min(tensor_torch, indices_torch, dim_size, axis)
    elif aggr == "mean":
        return scatter_mean(tensor_torch, indices_torch, dim_size, axis)
    elif aggr == "softmax":
        return scatter_softmax(tensor_torch, indices_torch, dim_size, axis)
    else:
        raise ValueError(f"Unsupported aggregation method: {aggr}")

def scatter_add(data: TensorLike, indices: TensorLike, dim_size: Optional[int] = None, axis: int = 0) -> torch.Tensor:
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
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    tensor_torch = tensor_ops.convert_to_tensor(data)
    indices_torch = tensor_ops.convert_to_tensor(indices).long()
    
    # Determine the output size
    if dim_size is None:
        dim_size = indices_torch.max().item() + 1
    
    # Create output shape
    output_shape = list(tensor_torch.shape)
    output_shape[axis] = dim_size
    
    # Initialize output tensor with zeros
    output = torch.zeros(output_shape, dtype=tensor_torch.dtype, device=tensor_torch.device)
    
    # For multi-dimensional tensors, we need to handle scatter differently
    if len(tensor_torch.shape) > 1:
        # Use manual iteration for multi-dimensional tensors
        for i in range(len(indices_torch)):
            idx = indices_torch[i].item()
            
            # Create slice for the specific index
            slice_indices = [py_slice(None)] * len(output_shape)
            slice_indices[axis] = idx
            
            # Get current slice of output tensor
            output_slice = output[tuple(slice_indices)]
            
            # Get corresponding slice of input tensor
            input_slice = tensor_torch[i]
            
            # Add values
            output[tuple(slice_indices)] = output_slice + input_slice
            
        return output
    else:
        # For 1D tensors, we can use torch.scatter_add_ directly
        return output.scatter_add_(axis, indices_torch, tensor_torch)

def scatter_max(data: TensorLike, indices: TensorLike, dim_size: Optional[int] = None, axis: int = 0) -> torch.Tensor:
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
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    tensor_torch = tensor_ops.convert_to_tensor(data)
    indices_torch = tensor_ops.convert_to_tensor(indices).long()
    
    # Determine the output size
    if dim_size is None:
        dim_size = indices_torch.max().item() + 1
    
    # Create output shape
    output_shape = list(tensor_torch.shape)
    output_shape[axis] = dim_size
    
    # Initialize output tensor with minimum value
    output = torch.full(output_shape, float('-inf'), dtype=tensor_torch.dtype, device=tensor_torch.device)
    
    # We can't use the reduce='amax' option as it's not available in all PyTorch versions
    # Instead, manually implement max scatter
    for i in range(tensor_torch.shape[0]):
        idx = indices_torch[i].item()
        value = tensor_torch[i]
        
        # Create slice for the specific index using Python's built-in slice
        slice_indices = [py_slice(None)] * len(output_shape)
        slice_indices[axis] = idx
        
        # Get current value
        current = output[tuple(slice_indices)]
        
        # Update with max
        output[tuple(slice_indices)] = torch.maximum(current, value)
    
    # Replace -inf with 0
    result = torch.where(output == float('-inf'), torch.zeros_like(output), output)
    
    return result

def scatter_min(data: TensorLike, indices: TensorLike, dim_size: Optional[int] = None, axis: int = 0) -> torch.Tensor:
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
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    tensor_torch = tensor_ops.convert_to_tensor(data)
    indices_torch = tensor_ops.convert_to_tensor(indices).long()
    
    # Determine the output size
    if dim_size is None:
        dim_size = indices_torch.max().item() + 1
    
    # Create output shape
    output_shape = list(tensor_torch.shape)
    output_shape[axis] = dim_size
    
    # Initialize output tensor with maximum value
    output = torch.full(output_shape, float('inf'), dtype=tensor_torch.dtype, device=tensor_torch.device)
    
    # We can't use the reduce='amin' option as it's not available in all PyTorch versions
    # Instead, manually implement min scatter
    for i in range(tensor_torch.shape[0]):
        idx = indices_torch[i].item()
        value = tensor_torch[i]
        
        # Create slice for the specific index using Python's built-in slice
        slice_indices = [py_slice(None)] * len(output_shape)
        slice_indices[axis] = idx
        
        # Get current value
        current = output[tuple(slice_indices)]
        
        # Update with min
        output[tuple(slice_indices)] = torch.minimum(current, value)
    
    # Replace inf with 0
    result = torch.where(output == float('inf'), torch.zeros_like(output), output)
    
    return result

def scatter_mean(data: TensorLike, indices: TensorLike, dim_size: Optional[int] = None, axis: int = 0) -> torch.Tensor:
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
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    tensor_torch = tensor_ops.convert_to_tensor(data)
    indices_torch = tensor_ops.convert_to_tensor(indices).long()
    
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

def scatter_softmax(data: TensorLike, indices: TensorLike, dim_size: Optional[int] = None, axis: int = 0) -> torch.Tensor:
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
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    tensor_torch = tensor_ops.convert_to_tensor(data)
    indices_torch = tensor_ops.convert_to_tensor(indices).long()
    
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