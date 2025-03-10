"""
NumPy data type operations for ember_ml.

This module provides NumPy implementations of data type operations.
"""

import numpy as np
from typing import Any, Type


def get_dtype(name: str) -> Type:
    """
    Get a data type by name.
    
    Args:
        name: The name of the data type
        
    Returns:
        The corresponding NumPy data type
    """
    if name == 'float32':
        return np.float32
    elif name == 'float64':
        return np.float64
    elif name == 'int32':
        return np.int32
    elif name == 'int64':
        return np.int64
    elif name == 'bool' or name == 'bool_':
        return np.bool_
    elif name == 'int8':
        return np.int8
    elif name == 'int16':
        return np.int16
    elif name == 'uint8':
        return np.uint8
    elif name == 'uint16':
        return np.uint16
    elif name == 'uint32':
        return np.uint32
    elif name == 'uint64':
        return np.uint64
    elif name == 'float16':
        return np.float16
    else:
        raise ValueError(f"Unknown data type: {name}")


def to_numpy_dtype(dtype: Any) -> Type:
    """
    Convert a data type to a NumPy data type.
    
    Args:
        dtype: The data type to convert
        
    Returns:
        The corresponding NumPy data type
    """
    if isinstance(dtype, type) and hasattr(np, dtype.__name__):
        return dtype
    elif isinstance(dtype, str):
        return get_dtype(dtype)
    elif isinstance(dtype, np.dtype):
        # Handle numpy.dtype objects
        return dtype.type
    else:
        raise ValueError(f"Cannot convert {dtype} to NumPy data type")


def from_numpy_dtype(dtype: Type) -> Type:
    """
    Convert a NumPy data type to a NumPy data type.
    
    Args:
        dtype: The NumPy data type to convert
        
    Returns:
        The corresponding NumPy data type
    """
    return dtype


def float32() -> Type:
    """Get the float32 data type."""
    return np.float32


def float64() -> Type:
    """Get the float64 data type."""
    return np.float64


def float16() -> Type:
    """Get the float16 data type."""
    return np.float16


def int32() -> Type:
    """Get the int32 data type."""
    return np.int32


def int64() -> Type:
    """Get the int64 data type."""
    return np.int64


def int16() -> Type:
    """Get the int16 data type."""
    return np.int16


def int8() -> Type:
    """Get the int8 data type."""
    return np.int8


def uint8() -> Type:
    """Get the uint8 data type."""
    return np.uint8


def uint16() -> Type:
    """Get the uint16 data type."""
    return np.uint16


def uint32() -> Type:
    """Get the uint32 data type."""
    return np.uint32


def uint64() -> Type:
    """Get the uint64 data type."""
    return np.uint64


def bool_() -> Type:
    """Get the boolean data type."""
    return np.bool_


class NumpyDTypeOps:
    """NumPy implementation of data type operations."""
    
    def get_dtype(self, name):
        """Get a data type by name."""
        return get_dtype(name)
    
    def to_numpy_dtype(self, dtype):
        """Convert a data type to a NumPy data type."""
        return to_numpy_dtype(dtype)
    
    def from_numpy_dtype(self, dtype):
        """Convert a NumPy data type to a NumPy data type."""
        return from_numpy_dtype(dtype)
    
    def float32(self):
        """Get the float32 data type."""
        return float32()
    
    def float64(self):
        """Get the float64 data type."""
        return float64()
    
    def float16(self):
        """Get the float16 data type."""
        return float16()
    
    def int32(self):
        """Get the int32 data type."""
        return int32()
    
    def int64(self):
        """Get the int64 data type."""
        return int64()
    
    def int16(self):
        """Get the int16 data type."""
        return int16()
    
    def int8(self):
        """Get the int8 data type."""
        return int8()
    
    def uint8(self):
        """Get the uint8 data type."""
        return uint8()
    
    def uint16(self):
        """Get the uint16 data type."""
        return uint16()
    
    def uint32(self):
        """Get the uint32 data type."""
        return uint32()
    
    def uint64(self):
        """Get the uint64 data type."""
        return uint64()
    
    def bool_(self):
        """Get the boolean data type."""
        return bool_()