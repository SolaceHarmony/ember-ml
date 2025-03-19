"""PyTorch tensor creation operations."""

import torch
from typing import Any, Optional, Union, Sequence, List, Tuple

from ember_ml.backend.torch.tensor.dtype import TorchDType
from ember_ml.backend.torch.tensor.ops.utility import convert_to_tensor

# Type aliases
Shape = Sequence[int]
TensorLike = Any
DType = Any

def zeros(shape: Shape, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of zeros.
    
    Args:
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

def ones(shape: Shape, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of ones.
    
    Args:
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

def zeros_like(data: TensorLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of zeros with the same shape as the input.
    
    Args:
        data: The input tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of zeros with the same shape as the input
    """
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    # Convert to PyTorch tensor first
    tensor = convert_to_tensor(data)
    return torch.zeros_like(tensor, dtype=torch_dtype, device=device)

def ones_like(data: TensorLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of ones with the same shape as the input.
    
    Args:
        data: The input tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of ones with the same shape as the input
    """
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    # Convert to PyTorch tensor first
    tensor = convert_to_tensor(data)
    return torch.ones_like(tensor, dtype=torch_dtype, device=device)

def eye(n: int, m: Optional[int] = None, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create an identity matrix.
    
    Args:
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

def full(shape: Shape, fill_value: Union[int, float, bool], dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor filled with a scalar value.
    
    Args:
        shape: Shape of the tensor
        fill_value: Value to fill the tensor with
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor filled with the specified value
    """
    # Handle string dtype values
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
    
    return torch.full(shape, fill_value, dtype=torch_dtype, device=device)

def full_like(data: TensorLike, fill_value: Union[int, float, bool], dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor filled with a scalar value with the same shape as the input.
    
    Args:
        data: Input tensor
        fill_value: Value to fill the tensor with
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor filled with the specified value with the same shape as data
    """
    tensor_torch = convert_to_tensor(data)
    
    # Handle string dtype values
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    # Create a full tensor with the same shape as the input
    return torch.full_like(tensor_torch, fill_value, dtype=torch_dtype, device=device)

def arange(start: Union[int, float], stop: Optional[Union[int, float]] = None, step: int = 1, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor with evenly spaced values within a given interval.
    
    Args:
        start: Start of interval (inclusive)
        stop: End of interval (exclusive)
        step: Spacing between values
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor with evenly spaced values
    """
    # Handle string dtype values
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    if stop is None:
        # If only one argument is provided, it's the stop value
        return torch.arange(start=0, end=start, step=step, dtype=torch_dtype, device=device)
    else:
        return torch.arange(start=start, end=stop, step=step, dtype=torch_dtype, device=device)

def linspace(start: Union[int, float], stop: Union[int, float], num: int, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor with evenly spaced values within a given interval.
    
    Args:
        start: Start of interval (inclusive)
        stop: End of interval (inclusive)
        num: Number of values to generate
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor with evenly spaced values
    """
    # Handle string dtype values
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    return torch.linspace(start=start, end=stop, steps=num, dtype=torch_dtype, device=device)