"""PyTorch tensor utility operations."""

import torch
import numpy
from typing import Union, Optional, Sequence, Any, List, Tuple

from ember_ml.backend.torch.tensor.dtype import TorchDType

# Type aliases
Shape = Sequence[int]
TensorLike = Any

def convert_to_tensor(data: Any, dtype: Optional[Any] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Convert data to a PyTorch tensor.
    
    Args:
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

def to_numpy(data: TensorLike) -> Optional[numpy.ndarray]:
    """
    Convert a PyTorch tensor to a NumPy array.
    
    IMPORTANT: This function is provided ONLY for visualization/plotting libraries
    that specifically require NumPy arrays. It should NOT be used for general tensor
    conversions or operations. Ember ML has a zero backend design where EmberTensor
    relies entirely on the selected backend for representation.
    
    Args:
        data: The tensor to convert
        
    Returns:
        NumPy array
    """
    if data is None:
        return None
        
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    
    # For non-tensor types, let PyTorch handle the conversion
    return convert_to_tensor(data).detach().cpu().numpy()

def item(data: TensorLike) -> Union[int, float, bool]:
    """
    Get the value of a scalar tensor.
    
    Args:
        data: The tensor to get the value from
        
    Returns:
        The scalar value
    """
    if isinstance(data, torch.Tensor):
        return data.item()
    return data

def shape(data: TensorLike) -> Shape:
    """
    Get the shape of a tensor.
    
    Args:
        data: The tensor to get the shape of
        
    Returns:
        The shape of the tensor
    """
    if isinstance(data, torch.Tensor):
        return tuple(data.shape)
    # Convert to tensor first
    return tuple(convert_to_tensor(data).shape)

def dtype(data: TensorLike) -> Any:
    """
    Get the data type of a tensor.
    
    Args:
        data: The tensor to get the data type of
        
    Returns:
        The data type of the tensor
    """
    if isinstance(data, torch.Tensor):
        return data.dtype
    # Convert to tensor first
    return convert_to_tensor(data).dtype

def copy(data: TensorLike) -> torch.Tensor:
    """
    Create a copy of a tensor.
    
    Args:
        data: The tensor to copy
        
    Returns:
        Copy of the tensor
    """
    if not isinstance(data, torch.Tensor):
        data = convert_to_tensor(data)
    
    return data.clone()

def var(data: TensorLike, axis: Optional[Union[int, List[int]]] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute the variance of a tensor along specified axes.
    
    Args:
        data: Input tensor
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Variance of the tensor
    """
    tensor_torch = convert_to_tensor(data)
    
    if axis is None:
        return torch.var(tensor_torch, keepdim=keepdims)
    
    return torch.var(tensor_torch, dim=axis, keepdim=keepdims)

def sort(data: TensorLike, axis: int = -1, descending: bool = False) -> torch.Tensor:
    """
    Sort a tensor along a specified axis.
    
    Args:
        data: Input tensor
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Sorted tensor
    """
    tensor_torch = convert_to_tensor(data)
    
    # PyTorch sort returns a tuple of (values, indices)
    values, _ = torch.sort(tensor_torch, dim=axis, descending=descending)
    
    return values

def argsort(data: TensorLike, axis: int = -1, descending: bool = False) -> torch.Tensor:
    """
    Return the indices that would sort a tensor along a specified axis.
    
    Args:
        data: Input tensor
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Indices that would sort the tensor
    """
    tensor_torch = convert_to_tensor(data)
    
    # PyTorch sort returns a tuple of (values, indices)
    _, indices = torch.sort(tensor_torch, dim=axis, descending=descending)
    
    return indices

def maximum(data1: TensorLike, data2: TensorLike) -> torch.Tensor:
    """
    Element-wise maximum of two tensors.
    
    Args:
        data1: First input tensor
        data2: Second input tensor
        
    Returns:
        Element-wise maximum
    """
    x_torch = convert_to_tensor(data1)
    y_torch = convert_to_tensor(data2)
    return torch.maximum(x_torch, y_torch)