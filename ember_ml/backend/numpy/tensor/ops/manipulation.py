"""NumPy tensor manipulation operations."""

import numpy as np
from typing import Union, Optional, Sequence, Any, List, Tuple

# Type aliases
Shape = Union[int, Sequence[int]]

def reshape(tensor_obj, tensor, shape):
    """
    Reshape a tensor.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: The tensor to reshape
        shape: The new shape
        
    Returns:
        Reshaped tensor
    """
    if not isinstance(tensor, np.ndarray):
        tensor = tensor_obj.convert_to_tensor(tensor)
    
    return tensor.reshape(shape)

def transpose(tensor_obj, tensor, axes=None):
    """
    Transpose a tensor.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: The tensor to transpose
        axes: Optional permutation of dimensions
        
    Returns:
        Transposed tensor
    """
    if not isinstance(tensor, np.ndarray):
        tensor = tensor_obj.convert_to_tensor(tensor)
    
    return np.transpose(tensor, axes)

def concatenate(tensor_obj, tensors, axis=0):
    """
    Concatenate tensors along a specified axis.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensors: The tensors to concatenate
        axis: The axis along which to concatenate
        
    Returns:
        Concatenated tensor
    """
    # Convert to NumPy arrays
    numpy_tensors = [tensor_obj.convert_to_tensor(t) for t in tensors]
    return np.concatenate(numpy_tensors, axis=axis)

def stack(tensor_obj, tensors, axis=0):
    """
    Stack tensors along a new axis.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensors: The tensors to stack
        axis: The axis along which to stack
        
    Returns:
        Stacked tensor
    """
    # Convert to NumPy arrays
    numpy_tensors = [tensor_obj.convert_to_tensor(t) for t in tensors]
    return np.stack(numpy_tensors, axis=axis)

def split(tensor_obj, tensor, num_or_size_splits, axis=0):
    """
    Split a tensor into sub-tensors.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: The tensor to split
        num_or_size_splits: Number of splits or sizes of each split
        axis: The axis along which to split
        
    Returns:
        List of sub-tensors
    """
    if not isinstance(tensor, np.ndarray):
        tensor = tensor_obj.convert_to_tensor(tensor)
    
    if isinstance(num_or_size_splits, int):
        # Avoid using int() and division operator
        # Just use array_split directly
        return np.array_split(tensor, num_or_size_splits, axis=axis)
    
    return np.split(tensor, num_or_size_splits, axis=axis)

def expand_dims(tensor_obj, tensor, axis):
    """
    Insert a new axis into a tensor's shape.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: The tensor to expand
        axis: The axis at which to insert the new dimension
        
    Returns:
        Expanded tensor
    """
    if not isinstance(tensor, np.ndarray):
        tensor = tensor_obj.convert_to_tensor(tensor)
    
    if isinstance(axis, int):
        return np.expand_dims(tensor, axis)
    
    # Handle multiple axes
    result = tensor
    for ax in sorted(axis):
        result = np.expand_dims(result, ax)
    return result

def squeeze(tensor_obj, tensor, axis=None):
    """
    Remove single-dimensional entries from a tensor's shape.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: The tensor to squeeze
        axis: The axis to remove
        
    Returns:
        Squeezed tensor
    """
    if not isinstance(tensor, np.ndarray):
        tensor = tensor_obj.convert_to_tensor(tensor)
    
    return np.squeeze(tensor, axis=axis)

def tile(tensor_obj, tensor, reps):
    """
    Construct a tensor by tiling a given tensor.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: Input tensor
        reps: Number of repetitions along each dimension
        
    Returns:
        Tiled tensor
    """
    tensor_np = tensor_obj.convert_to_tensor(tensor)
    return np.tile(tensor_np, reps)

def pad(tensor_obj, tensor, paddings, constant_values=0):
    """
    Pad a tensor with a constant value.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: Input tensor
        paddings: Sequence of sequences of integers specifying the padding for each dimension
                Each inner sequence should contain two integers: [pad_before, pad_after]
        constant_values: Value to pad with
        
    Returns:
        Padded tensor
    """
    tensor_np = tensor_obj.convert_to_tensor(tensor)
    
    # Convert paddings to the format expected by np.pad
    # NumPy expects ((pad_before_dim1, pad_after_dim1), (pad_before_dim2, pad_after_dim2), ...)
    pad_width = tuple(tuple(p) for p in paddings)
    
    # Pad the tensor
    return np.pad(tensor_np, pad_width, mode='constant', constant_values=constant_values)