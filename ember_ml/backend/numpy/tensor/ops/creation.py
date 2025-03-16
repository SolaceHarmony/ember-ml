"""NumPy tensor creation operations."""

import numpy as np
from typing import Union, Optional, Sequence, Any

from ember_ml.backend.numpy.tensor.dtype import NumpyDType

# Type aliases
Shape = Union[int, Sequence[int]]

def zeros(tensor_obj, shape, dtype=None, device=None):
    """
    Create a tensor of zeros.
    
    Args:
        tensor_obj: NumpyTensor instance
        shape: The shape of the tensor
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        Tensor of zeros
    """
    numpy_dtype = None
    if dtype is not None:
        numpy_dtype = NumpyDType().from_dtype_str(dtype)
    
    return np.zeros(shape, dtype=numpy_dtype)

def ones(tensor_obj, shape, dtype=None, device=None):
    """
    Create a tensor of ones.
    
    Args:
        tensor_obj: NumpyTensor instance
        shape: The shape of the tensor
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        Tensor of ones
    """
    numpy_dtype = None
    if dtype is not None:
        numpy_dtype = NumpyDType().from_dtype_str(dtype)
    
    return np.ones(shape, dtype=numpy_dtype)

def zeros_like(tensor_obj, tensor, dtype=None, device=None):
    """
    Create a tensor of zeros with the same shape as the input.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: The input tensor
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        Tensor of zeros with the same shape as the input
    """
    numpy_dtype = None
    if dtype is not None:
        numpy_dtype = NumpyDType().from_dtype_str(dtype)
    
    if isinstance(tensor, np.ndarray):
        return np.zeros_like(tensor, dtype=numpy_dtype)
    
    # Convert to NumPy array first
    tensor = tensor_obj.convert_to_tensor(tensor)
    return np.zeros_like(tensor, dtype=numpy_dtype)

def ones_like(tensor_obj, tensor, dtype=None, device=None):
    """
    Create a tensor of ones with the same shape as the input.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: The input tensor
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        Tensor of ones with the same shape as the input
    """
    numpy_dtype = None
    if dtype is not None:
        numpy_dtype = NumpyDType().from_dtype_str(dtype)
    
    if isinstance(tensor, np.ndarray):
        return np.ones_like(tensor, dtype=numpy_dtype)
    
    # Convert to NumPy array first
    tensor = tensor_obj.convert_to_tensor(tensor)
    return np.ones_like(tensor, dtype=numpy_dtype)

def eye(tensor_obj, n, m=None, dtype=None, device=None):
    """
    Create an identity matrix.
    
    Args:
        tensor_obj: NumpyTensor instance
        n: Number of rows
        m: Number of columns (default: n)
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        Identity matrix
    """
    numpy_dtype = None
    if dtype is not None:
        numpy_dtype = NumpyDType().from_dtype_str(dtype)
    
    return np.eye(n, m, dtype=numpy_dtype)

def full(tensor_obj, shape, fill_value, dtype=None, device=None):
    """
    Create a tensor filled with a scalar value.
    
    Args:
        tensor_obj: NumpyTensor instance
        shape: Shape of the tensor
        fill_value: Value to fill the tensor with
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        Tensor filled with the specified value
    """
    numpy_dtype = None
    if dtype is not None:
        numpy_dtype = NumpyDType().from_dtype_str(dtype)
    
    return np.full(shape, fill_value, dtype=numpy_dtype)

def full_like(tensor_obj, tensor, fill_value, dtype=None, device=None):
    """
    Create a tensor filled with a scalar value with the same shape as the input.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: Input tensor
        fill_value: Value to fill the tensor with
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        Tensor filled with the specified value with the same shape as tensor
    """
    tensor_np = tensor_obj.convert_to_tensor(tensor)
    
    numpy_dtype = None
    if dtype is not None:
        numpy_dtype = NumpyDType().from_dtype_str(dtype)
    elif numpy_dtype is None:
        numpy_dtype = tensor_np.dtype
    
    return np.full_like(tensor_np, fill_value, dtype=numpy_dtype)

def arange(tensor_obj, start, stop=None, step=1, dtype=None, device=None):
    """
    Create a tensor with evenly spaced values within a given interval.
    
    Args:
        tensor_obj: NumpyTensor instance
        start: Start of interval (inclusive)
        stop: End of interval (exclusive)
        step: Spacing between values
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        Tensor with evenly spaced values
    """
    numpy_dtype = None
    if dtype is not None:
        numpy_dtype = NumpyDType().from_dtype_str(dtype)
    
    if stop is None:
        # If only one argument is provided, it's the stop value
        return np.arange(start=0, stop=start, step=step, dtype=numpy_dtype)
    return np.arange(start=start, stop=stop, step=step, dtype=numpy_dtype)

def linspace(tensor_obj, start, stop, num, dtype=None, device=None):
    """
    Create a tensor with evenly spaced values within a given interval.
    
    Args:
        tensor_obj: NumpyTensor instance
        start: Start of interval (inclusive)
        stop: End of interval (inclusive)
        num: Number of values to generate
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        Tensor with evenly spaced values
    """
    numpy_dtype = None
    if dtype is not None:
        numpy_dtype = NumpyDType().from_dtype_str(dtype)
    
    return np.linspace(start=start, stop=stop, num=num, dtype=numpy_dtype)