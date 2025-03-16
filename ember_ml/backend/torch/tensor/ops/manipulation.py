"""PyTorch tensor manipulation operations."""

import torch
import torch.nn.functional as F
from typing import Union, Optional, Sequence, Any, List, Tuple

# Type aliases
Shape = Union[int, Sequence[int]]

def reshape(tensor_obj, tensor, shape):
    """
    Reshape a tensor.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: The tensor to reshape
        shape: The new shape
        
    Returns:
        Reshaped tensor
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = tensor_obj.convert_to_tensor(tensor)
    
    return tensor.reshape(shape)

def transpose(tensor_obj, tensor, axes=None):
    """
    Transpose a tensor.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: The tensor to transpose
        axes: Optional permutation of dimensions
        
    Returns:
        Transposed tensor
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = tensor_obj.convert_to_tensor(tensor)
    
    if axes is None:
        return tensor.t()
    return tensor.permute(axes)

def concatenate(tensor_obj, tensors, axis=0):
    """
    Concatenate tensors along a specified axis.
    
    Args:
        tensor_obj: TorchTensor instance
        tensors: The tensors to concatenate
        axis: The axis along which to concatenate
        
    Returns:
        Concatenated tensor
    """
    # Convert to PyTorch tensors
    torch_tensors = [tensor_obj.convert_to_tensor(t) for t in tensors]
    return torch.cat(torch_tensors, dim=axis)

def stack(tensor_obj, tensors, axis=0):
    """
    Stack tensors along a new axis.
    
    Args:
        tensor_obj: TorchTensor instance
        tensors: The tensors to stack
        axis: The axis along which to stack
        
    Returns:
        Stacked tensor
    """
    # Convert to PyTorch tensors
    torch_tensors = [tensor_obj.convert_to_tensor(t) for t in tensors]
    return torch.stack(torch_tensors, dim=axis)

def split(tensor_obj, tensor, num_or_size_splits, axis=0):
    """
    Split a tensor into sub-tensors.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: The tensor to split
        num_or_size_splits: Number of splits or sizes of each split
        axis: The axis along which to split
        
    Returns:
        List of sub-tensors
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = tensor_obj.convert_to_tensor(tensor)
    
    if isinstance(num_or_size_splits, int):
        # Get the size of the dimension we're splitting along
        dim_size = tensor.shape[axis]
        
        # Check if the dimension is evenly divisible by num_or_size_splits
        # Use torch.remainder instead of % operator
        is_divisible = torch.eq(torch.remainder(torch.tensor(dim_size), torch.tensor(num_or_size_splits)), torch.tensor(0))
        
        if is_divisible:
            # If evenly divisible, use a simple split
            # Use torch.div instead of // operator
            split_size = torch.div(torch.tensor(dim_size), torch.tensor(num_or_size_splits), rounding_mode='trunc')
            # Convert to int to avoid type error
            split_size_int = int(split_size.item())
            return list(torch.split(tensor, split_size_int, dim=axis))
        else:
            # If not evenly divisible, create a list of split sizes
            # Use torch.div instead of // operator
            base_size = torch.div(torch.tensor(dim_size), torch.tensor(num_or_size_splits), rounding_mode='trunc')
            # Use torch.remainder instead of % operator
            remainder = torch.remainder(torch.tensor(dim_size), torch.tensor(num_or_size_splits))
            
            # Create a list where the first 'remainder' chunks have size 'base_size + 1'
            # and the rest have size 'base_size'
            split_sizes = []
            for i in range(num_or_size_splits):
                if i < remainder.item():
                    # Use torch.add instead of + operator
                    split_sizes.append(torch.add(base_size, torch.tensor(1)).item())
                else:
                    split_sizes.append(base_size.item())
            
            return list(torch.split(tensor, split_sizes, dim=axis))
    
    return list(torch.split(tensor, num_or_size_splits, dim=axis))

def expand_dims(tensor_obj, tensor, axis):
    """
    Insert a new axis into a tensor's shape.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: The tensor to expand
        axis: The axis at which to insert the new dimension
        
    Returns:
        Expanded tensor
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = tensor_obj.convert_to_tensor(tensor)
    
    if isinstance(axis, int):
        return tensor.unsqueeze(axis)
    
    # Handle multiple axes
    result = tensor
    for ax in sorted(axis):
        result = result.unsqueeze(ax)
    return result

def squeeze(tensor_obj, tensor, axis=None):
    """
    Remove single-dimensional entries from a tensor's shape.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: The tensor to squeeze
        axis: The axis to remove
        
    Returns:
        Squeezed tensor
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = tensor_obj.convert_to_tensor(tensor)
    
    if axis is None:
        return tensor.squeeze()
    
    if isinstance(axis, int):
        return tensor.squeeze(axis)
    
    # Handle multiple axes
    result = tensor
    for ax in sorted(axis, reverse=True):
        result = result.squeeze(ax)
    return result

def tile(tensor_obj, tensor, reps):
    """
    Construct a tensor by tiling a given tensor.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: Input tensor
        reps: Number of repetitions along each dimension
        
    Returns:
        Tiled tensor
    """
    tensor_torch = tensor_obj.convert_to_tensor(tensor)
    return tensor_torch.repeat(reps)

def pad(tensor_obj, tensor, paddings, constant_values=0):
    """
    Pad a tensor with a constant value.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: Input tensor
        paddings: Sequence of sequences of integers specifying the padding for each dimension
                Each inner sequence should contain two integers: [pad_before, pad_after]
        constant_values: Value to pad with
        
    Returns:
        Padded tensor
    """
    tensor_torch = tensor_obj.convert_to_tensor(tensor)
    
    # Convert paddings to the format expected by torch.nn.functional.pad
    # PyTorch expects (pad_left, pad_right, pad_top, pad_bottom, ...)
    # We need to reverse the order and flatten
    pad_list = []
    for pad_pair in reversed(paddings):
        pad_list.extend(pad_pair)
    
    # Pad the tensor
    return F.pad(tensor_torch, pad_list, mode='constant', value=constant_values)