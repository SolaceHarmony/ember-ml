"""
PyTorch data type operations for ember_ml.

This module provides PyTorch implementations of data type operations.
"""

import torch
from typing import Union, Any, Optional


def ember_dtype_to_torch(dtype: Union[Any, str, None]) -> Optional[torch.dtype]:
    """
    Convert an EmberDtype to a PyTorch data type.
    
    Args:
        dtype: The EmberDtype to convert
        
    Returns:
        The corresponding PyTorch data type
    """
    if dtype is None:
        return None
    
    # If it's already a PyTorch dtype, return it
    if isinstance(dtype, torch.dtype):
        return dtype
    
    # If it's an EmberDtype, use its name
    if hasattr(dtype, 'name'):
        dtype_name = dtype.name
    elif isinstance(dtype, str):
        dtype_name = dtype
    else:
        raise ValueError(f"Cannot convert {dtype} to PyTorch data type")
    
    # Map dtype names to PyTorch dtypes
    if dtype_name == 'float32':
        return torch.float32
    elif dtype_name == 'float64':
        return torch.float64
    elif dtype_name == 'int32':
        return torch.int32
    elif dtype_name == 'int64':
        return torch.int64
    elif dtype_name == 'bool' or dtype_name == 'bool_':
        return torch.bool
    elif dtype_name == 'int8':
        return torch.int8
    elif dtype_name == 'int16':
        return torch.int16
    elif dtype_name == 'uint8':
        return torch.uint8
    elif dtype_name == 'float16':
        return torch.float16
    else:
        raise ValueError(f"Unknown data type: {dtype_name}")


def torch_to_ember_dtype(dtype: Union[torch.dtype, str, None]) -> Optional[str]:
    """
    Convert a PyTorch data type to an EmberDtype.
    
    Args:
        dtype: The PyTorch data type to convert
        
    Returns:
        The corresponding EmberDtype name
    """
    if dtype is None:
        return None
    
    # Map PyTorch dtypes to EmberDtype names
    if dtype == torch.float32:
        return 'float32'
    elif dtype == torch.float64:
        return 'float64'
    elif dtype == torch.int32:
        return 'int32'
    elif dtype == torch.int64:
        return 'int64'
    elif dtype == torch.bool:
        return 'bool'
    elif dtype == torch.int8:
        return 'int8'
    elif dtype == torch.int16:
        return 'int16'
    elif dtype == torch.uint8:
        return 'uint8'
    elif dtype == torch.float16:
        return 'float16'
    elif isinstance(dtype, str):
        # If it's already a string, return it
        return dtype
    else:
        raise ValueError(f"Cannot convert {dtype} to EmberDtype")


class TorchDTypeOps:
    """PyTorch implementation of data type operations."""
    
    def get_dtype(self, name):
        """Get a data type by name."""
        return ember_dtype_to_torch(name)
    
    def to_numpy_dtype(self, dtype):
        """Convert a PyTorch data type to a NumPy data type."""
        return torch_to_ember_dtype(dtype)
    
    def from_numpy_dtype(self, dtype):
        """Convert a NumPy data type to a PyTorch data type."""
        return ember_dtype_to_torch(dtype)