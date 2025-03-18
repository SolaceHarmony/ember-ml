"""PyTorch tensor utility operations."""

import torch
from typing import Union, Sequence

from ember_ml.backend.torch.tensor.dtype import TorchDType

# Type aliases
Shape = Union[int, Sequence[int]]

def convert_to_tensor(tensor_obj, data, dtype=None, device=None):
    """
    Convert data to a PyTorch tensor.
    
    Args:
        tensor_obj: TorchTensor instance
        data: The data to convert
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        PyTorch tensor
    """
    # If it's already a PyTorch tensor, return it
    if isinstance(data, torch.Tensor):
        if dtype is not None:
            # Convert dtype if needed
            torch_dtype = TorchDType().from_dtype_str(dtype)
            return data.to(torch_dtype)
        return data
    
    # Convert to PyTorch tensor
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    # Handle EmberTensor objects
    if isinstance(data, object) and getattr(data.__class__, '__name__', '') == 'EmberTensor':
        # For EmberTensor, extract the underlying PyTorch tensor
        # We know from inspection that _tensor is a torch.Tensor
        if hasattr(data, '_tensor'):
            return getattr(data, '_tensor')
    
    # Handle NumPy arrays - the community loves them so much!
    if hasattr(data, '__class__') and data.__class__.__module__ == 'numpy' and data.__class__.__name__ == 'ndarray':
        return torch.from_numpy(data)
    
    # Handle array-like objects
    try:
        return torch.tensor(data, dtype=torch_dtype)
    except:
        raise ValueError(f"Cannot convert {type(data)} to PyTorch tensor")

def to_numpy(tensor_obj, tensor):
    """
    Convert a PyTorch tensor to a NumPy-compatible array.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: The tensor to convert
        
    Returns:
        NumPy-compatible array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    # For non-tensor types, let PyTorch handle the conversion
    return tensor_obj.convert_to_tensor(tensor).detach().cpu().numpy()

def item(tensor_obj, tensor):
    """
    Get the value of a scalar tensor.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: The tensor to get the value from
        
    Returns:
        The scalar value
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.item()
    return tensor

def shape(tensor_obj, tensor):
    """
    Get the shape of a tensor.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: The tensor to get the shape of
        
    Returns:
        The shape of the tensor
    """
    if isinstance(tensor, torch.Tensor):
        return tuple(tensor.shape)
    # Convert to tensor first
    return tuple(tensor_obj.convert_to_tensor(tensor).shape)

def dtype(tensor_obj, tensor):
    """
    Get the data type of a tensor.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: The tensor to get the data type of
        
    Returns:
        The data type of the tensor
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.dtype
    # Convert to tensor first
    return tensor_obj.convert_to_tensor(tensor).dtype

def copy(tensor_obj, tensor):
    """
    Create a copy of a tensor.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: The tensor to copy
        
    Returns:
        Copy of the tensor
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = tensor_obj.convert_to_tensor(tensor)
    
    return tensor.clone()

def var(tensor_obj, tensor, axis=None, keepdims=False):
    """
    Compute the variance of a tensor along specified axes.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: Input tensor
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Variance of the tensor
    """
    tensor_torch = tensor_obj.convert_to_tensor(tensor)
    
    if axis is None:
        return torch.var(tensor_torch, keepdim=keepdims)
    
    return torch.var(tensor_torch, dim=axis, keepdim=keepdims)

def sort(tensor_obj, tensor, axis=-1, descending=False):
    """
    Sort a tensor along a specified axis.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: Input tensor
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Sorted tensor
    """
    tensor_torch = tensor_obj.convert_to_tensor(tensor)
    
    # PyTorch sort returns a tuple of (values, indices)
    values, _ = torch.sort(tensor_torch, dim=axis, descending=descending)
    
    return values

def argsort(tensor_obj, tensor, axis=-1, descending=False):
    """
    Return the indices that would sort a tensor along a specified axis.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: Input tensor
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Indices that would sort the tensor
    """
    tensor_torch = tensor_obj.convert_to_tensor(tensor)
    
    # PyTorch sort returns a tuple of (values, indices)
    _, indices = torch.sort(tensor_torch, dim=axis, descending=descending)
    
    return indices

def maximum(tensor_obj, x, y):
    """
    Element-wise maximum of two tensors.
    
    Args:
        tensor_obj: TorchTensor instance
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Element-wise maximum
    """
    x_torch = tensor_obj.convert_to_tensor(x)
    y_torch = tensor_obj.convert_to_tensor(y)
    return torch.maximum(x_torch, y_torch)