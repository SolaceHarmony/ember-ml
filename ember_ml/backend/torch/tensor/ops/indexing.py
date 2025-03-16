"""PyTorch tensor indexing operations."""

import torch
from typing import Union, Optional, Sequence, Any, List, Tuple

# Type aliases
Shape = Union[int, Sequence[int]]

def slice_tensor(tensor_obj, tensor, starts, sizes):
    """
    Extract a slice from a tensor.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: Input tensor
        starts: Starting indices for each dimension
        sizes: Size of the slice in each dimension. A value of -1 means "all remaining elements in this dimension"
        
    Returns:
        Sliced tensor
    """
    tensor_torch = tensor_obj.convert_to_tensor(tensor)
    
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

def slice_update(tensor_obj, tensor, slices, updates):
    """
    Update a tensor at specific indices.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: Input tensor to update
        slices: List or tuple of slice objects or indices
        updates: Values to insert at the specified indices
        
    Returns:
        Updated tensor
    """
    tensor_torch = tensor_obj.convert_to_tensor(tensor)
    updates_torch = tensor_obj.convert_to_tensor(updates)
    
    # Create a copy of the input tensor
    result = tensor_torch.clone()
    
    # Update the tensor at the specified indices
    result[slices] = updates_torch
    
    return result

def gather(tensor_obj, tensor, indices, axis=0):
    """
    Gather slices from a tensor along an axis.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: Input tensor
        indices: Indices of slices to gather
        axis: Axis along which to gather
        
    Returns:
        Gathered tensor
    """
    tensor_torch = tensor_obj.convert_to_tensor(tensor)
    indices_torch = tensor_obj.convert_to_tensor(indices)
    
    # Convert indices to long
    indices_torch = indices_torch.long()
    
    return torch.index_select(tensor_torch, axis, indices_torch)

def tensor_scatter_nd_update(tensor_obj, tensor, indices, updates):
    """
    Updates values of a tensor at specified indices.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: Input tensor to update
        indices: Indices at which to update values (N-dimensional indices)
        updates: Values to insert at the specified indices
        
    Returns:
        Updated tensor
    """
    # Create a copy of the tensor
    tensor_torch = tensor_obj.convert_to_tensor(tensor)
    indices_torch = tensor_obj.convert_to_tensor(indices)
    updates_torch = tensor_obj.convert_to_tensor(updates)
    
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