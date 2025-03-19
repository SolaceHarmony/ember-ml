"""PyTorch tensor manipulation operations."""

import torch
import torch.nn.functional as F
from typing import Union, Optional, Sequence, Any, List, Tuple

from ember_ml.backend.torch.tensor.ops.utility import convert_to_tensor

# Type aliases
Shape = Sequence[int]
TensorLike = Any

def reshape(data: TensorLike, shape: Shape) -> torch.Tensor:
    """
    Reshape a tensor.
    
    Args:
        data: The tensor to reshape
        shape: The new shape
        
    Returns:
        Reshaped tensor
    """
    tensor = convert_to_tensor(data)
    return tensor.reshape(shape)

def transpose(data: TensorLike, axes: Optional[List[int]] = None) -> torch.Tensor:
    """
    Transpose a tensor.
    
    Args:
        data: The tensor to transpose
        axes: Optional permutation of dimensions
        
    Returns:
        Transposed tensor
    """
    tensor = convert_to_tensor(data)
    
    if axes is None:
        return tensor.t()
    return tensor.permute(axes)

def concatenate(data: List[TensorLike], axis: int = 0) -> torch.Tensor:
    """
    Concatenate tensors along a specified axis.
    
    Args:
        data: The tensors to concatenate
        axis: The axis along which to concatenate
        
    Returns:
        Concatenated tensor
    """
    # Convert to PyTorch tensors
    torch_tensors = [convert_to_tensor(t) for t in data]
    return torch.cat(torch_tensors, dim=axis)

def stack(data: List[TensorLike], axis: int = 0) -> torch.Tensor:
    """
    Stack tensors along a new axis.
    
    Args:
        data: The tensors to stack
        axis: The axis along which to stack
        
    Returns:
        Stacked tensor
    """
    # Convert to PyTorch tensors
    torch_tensors = [convert_to_tensor(t) for t in data]
    return torch.stack(torch_tensors, dim=axis)

def split(data: TensorLike, num_or_size_splits: Union[int, List[int]], axis: int = 0) -> List[torch.Tensor]:
    """
    Split a tensor into sub-tensors.
    
    Args:
        data: The tensor to split
        num_or_size_splits: Number of splits or sizes of each split
        axis: The axis along which to split
        
    Returns:
        List of sub-tensors
    """
    tensor = convert_to_tensor(data)
    
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

def expand_dims(data: TensorLike, axis: Union[int, List[int]]) -> torch.Tensor:
    """
    Insert new axes into a tensor's shape.
    
    Args:
        data: The tensor to expand
        axis: The axis or axes at which to insert the new dimension(s)
        
    Returns:
        Expanded tensor
    """
    tensor = convert_to_tensor(data)
    
    if isinstance(axis, int):
        return tensor.unsqueeze(axis)
    
    # Handle multiple axes
    result = tensor
    for ax in sorted(axis):
        result = result.unsqueeze(ax)
    return result

def squeeze(data: TensorLike, axis: Optional[Union[int, List[int]]] = None) -> torch.Tensor:
    """
    Remove single-dimensional entries from a tensor's shape.
    
    Args:
        data: The tensor to squeeze
        axis: The axis or axes to remove
        
    Returns:
        Squeezed tensor
    """
    tensor = convert_to_tensor(data)
    
    if axis is None:
        return tensor.squeeze()
    
    if isinstance(axis, int):
        return tensor.squeeze(axis)
    
    # Handle multiple axes
    result = tensor
    for ax in sorted(axis, reverse=True):
        result = result.squeeze(ax)
    return result

def tile(data: TensorLike, reps: List[int]) -> torch.Tensor:
    """
    Construct a tensor by tiling a given tensor.
    
    Args:
        data: Input tensor
        reps: Number of repetitions along each dimension
        
    Returns:
        Tiled tensor
    """
    tensor = convert_to_tensor(data)
    return tensor.repeat(tuple(reps))

def pad(data: TensorLike, paddings: List[List[int]], constant_values: int = 0) -> torch.Tensor:
    """
    Pad a tensor with a constant value.
    
    Args:
        data: Input tensor
        paddings: List of lists of integers specifying the padding for each dimension
                Each inner list should contain two integers: [pad_before, pad_after]
        constant_values: Value to pad with
        
    Returns:
        Padded tensor
    """
    tensor = convert_to_tensor(data)
    
    # Convert paddings to the format expected by torch.nn.functional.pad
    # PyTorch expects (pad_left, pad_right, pad_top, pad_bottom, ...)
    # We need to reverse the order and flatten
    pad_list = []
    for pad_pair in reversed(paddings):
        pad_list.extend(pad_pair)
    
    # Pad the tensor
    return F.pad(tensor, pad_list, mode='constant', value=constant_values)