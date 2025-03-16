"""PyTorch tensor creation operations."""

import torch
from typing import Any, Optional, Union, Sequence, List, Tuple

from ember_ml.backend.torch.tensor.dtype import TorchDType

# Type aliases
Shape = Union[int, Sequence[int]]

def zeros(tensor_obj, shape: Shape, dtype: Optional[Any] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of zeros.
    
    Args:
        tensor_obj: TorchTensor instance
        shape: The shape of the tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of zeros
    """
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    if isinstance(shape, int):
        shape = (shape,)
    
    return torch.zeros(shape, dtype=torch_dtype, device=device)

def ones(tensor_obj, shape: Shape, dtype: Optional[Any] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of ones.
    
    Args:
        tensor_obj: TorchTensor instance
        shape: The shape of the tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of ones
    """
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    if isinstance(shape, int):
        shape = (shape,)
    
    return torch.ones(shape, dtype=torch_dtype, device=device)

def zeros_like(tensor_obj, tensor, dtype: Optional[Any] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of zeros with the same shape as the input.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: The input tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of zeros with the same shape as the input
    """
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    if isinstance(tensor, torch.Tensor):
        return torch.zeros_like(tensor, dtype=torch_dtype, device=device)
    
    # Convert to PyTorch tensor first
    tensor = tensor_obj.convert_to_tensor(tensor)
    return torch.zeros_like(tensor, dtype=torch_dtype, device=device)

def ones_like(tensor_obj, tensor, dtype: Optional[Any] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of ones with the same shape as the input.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: The input tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of ones with the same shape as the input
    """
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    if isinstance(tensor, torch.Tensor):
        return torch.ones_like(tensor, dtype=torch_dtype, device=device)
    
    # Convert to PyTorch tensor first
    tensor = tensor_obj.convert_to_tensor(tensor)
    return torch.ones_like(tensor, dtype=torch_dtype, device=device)

def eye(tensor_obj, n: int, m: Optional[int] = None, dtype: Optional[Any] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create an identity matrix.
    
    Args:
        tensor_obj: TorchTensor instance
        n: Number of rows
        m: Number of columns (default: n)
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Identity matrix
    """
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    if m is None:
        m = n
    
    return torch.eye(n, m, dtype=torch_dtype, device=device)

def full(tensor_obj, shape: Shape, fill_value, dtype: Optional[Any] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor filled with a scalar value.
    
    Args:
        tensor_obj: TorchTensor instance
        shape: Shape of the tensor
        fill_value: Value to fill the tensor with
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor filled with the specified value
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        dtype = TorchDType().from_dtype_str(dtype)
    
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
    
    return torch.full(shape, fill_value, dtype=dtype, device=device)

def full_like(tensor_obj, tensor, fill_value, dtype: Optional[Any] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor filled with a scalar value with the same shape as the input.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: Input tensor
        fill_value: Value to fill the tensor with
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor filled with the specified value with the same shape as tensor
    """
    tensor_torch = tensor_obj.convert_to_tensor(tensor)
    
    # Handle string dtype values
    if isinstance(dtype, str):
        dtype = TorchDType().from_dtype_str(dtype)
    
    # If dtype is None, use the dtype of the input tensor
    if dtype is None:
        dtype = tensor_torch.dtype
    
    # Create a full tensor with the same shape as the input
    return torch.full_like(tensor_torch, fill_value, dtype=dtype, device=device)

def arange(tensor_obj, start, stop=None, step=1, dtype: Optional[Any] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor with evenly spaced values within a given interval.
    
    Args:
        tensor_obj: TorchTensor instance
        start: Start of interval (inclusive)
        stop: End of interval (exclusive)
        step: Spacing between values
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor with evenly spaced values
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        dtype = TorchDType().from_dtype_str(dtype)
    # Handle EmberDtype objects
    elif dtype is not None and hasattr(dtype, 'name'):
        dtype = TorchDType().from_dtype_str(dtype.name)
    
    if stop is None:
        # If only one argument is provided, it's the stop value
        return torch.arange(start=0, end=start, step=step, dtype=dtype, device=device)
    else:
        return torch.arange(start=start, end=stop, step=step, dtype=dtype, device=device)

def linspace(tensor_obj, start, stop, num, dtype: Optional[Any] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor with evenly spaced values within a given interval.
    
    Args:
        tensor_obj: TorchTensor instance
        start: Start of interval (inclusive)
        stop: End of interval (inclusive)
        num: Number of values to generate
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor with evenly spaced values
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        dtype = TorchDType().from_dtype_str(dtype)
    # Handle EmberDtype objects
    elif dtype is not None and hasattr(dtype, 'name'):
        dtype = TorchDType().from_dtype_str(dtype.name)
    
    return torch.linspace(start=start, end=stop, steps=num, dtype=dtype, device=device)