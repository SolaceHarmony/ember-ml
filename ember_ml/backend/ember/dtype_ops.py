"""
Ember data type operations for ember_ml.

This module provides Ember implementations of data type operations.
"""

from typing import Any

# Define string representations of data types
_DTYPE_STRINGS = {
    'float32': 'float32',
    'float64': 'float64',
    'int32': 'int32',
    'int64': 'int64',
    'bool': 'bool',
    'int8': 'int8',
    'int16': 'int16',
    'uint8': 'uint8',
    'uint16': 'uint16',
    'uint32': 'uint32',
    'uint64': 'uint64',
    'float16': 'float16'
}

# Define EmberDType class to represent data types
class EmberDType:
    """
    Represents a data type in the Ember backend.
    
    This class is a simple wrapper around a string representation of a data type.
    """
    
    def __init__(self, name: str):
        """
        Initialize an EmberDType.
        
        Args:
            name: The name of the data type
        """
        if name not in _DTYPE_STRINGS:
            raise ValueError(f"Unknown data type: {name}")
        self.name = name
    
    def __str__(self) -> str:
        """Get the string representation of the data type."""
        return self.name
    
    def __repr__(self) -> str:
        """Get the string representation of the data type."""
        return f"EmberDType('{self.name}')"
    
    def __eq__(self, other: Any) -> bool:
        """Check if two data types are equal."""
        if isinstance(other, EmberDType):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return False


def get_dtype(name: str) -> EmberDType:
    """
    Get a data type by name.
    
    Args:
        name: The name of the data type
        
    Returns:
        The corresponding Ember data type
    """
    return EmberDType(name)


def to_dtype_str(dtype: Any) -> str:
    """
    Convert an Ember data type to a string representation.
    
    Args:
        dtype: The Ember data type to convert
        
    Returns:
        String representation of the data type
    """
    if isinstance(dtype, EmberDType):
        return dtype.name
    elif isinstance(dtype, str):
        if dtype in _DTYPE_STRINGS:
            return dtype
        else:
            raise ValueError(f"Unknown data type string: {dtype}")
    else:
        # Delegate to the underlying backend
        from ember_ml.backend import get_backend_module
        backend_module = get_backend_module()
        if hasattr(backend_module, 'dtype_ops') and hasattr(backend_module.dtype_ops, 'to_dtype_str'):
            return backend_module.dtype_ops.to_dtype_str(dtype)
        else:
            raise ValueError(f"Cannot convert {dtype} to dtype string")


def from_dtype_str(dtype_str: str) -> EmberDType:
    """
    Convert a string representation to an Ember data type.
    
    Args:
        dtype_str: String representation of the data type
        
    Returns:
        The corresponding Ember data type
    """
    if isinstance(dtype_str, EmberDType):
        return dtype_str
    elif isinstance(dtype_str, str):
        return get_dtype(dtype_str)
    else:
        raise ValueError(f"Cannot convert {dtype_str} to Ember data type")


def float32() -> EmberDType:
    """Get the float32 data type."""
    return EmberDType('float32')


def float64() -> EmberDType:
    """Get the float64 data type."""
    return EmberDType('float64')


def float16() -> EmberDType:
    """Get the float16 data type."""
    return EmberDType('float16')


def int32() -> EmberDType:
    """Get the int32 data type."""
    return EmberDType('int32')


def int64() -> EmberDType:
    """Get the int64 data type."""
    return EmberDType('int64')


def int16() -> EmberDType:
    """Get the int16 data type."""
    return EmberDType('int16')


def int8() -> EmberDType:
    """Get the int8 data type."""
    return EmberDType('int8')


def uint8() -> EmberDType:
    """Get the uint8 data type."""
    return EmberDType('uint8')


def uint16() -> EmberDType:
    """Get the uint16 data type."""
    return EmberDType('uint16')


def uint32() -> EmberDType:
    """Get the uint32 data type."""
    return EmberDType('uint32')


def uint64() -> EmberDType:
    """Get the uint64 data type."""
    return EmberDType('uint64')


def bool_() -> EmberDType:
    """Get the boolean data type."""
    return EmberDType('bool')


class EmberBackendDTypeOps:
    """Ember backend implementation of data type operations."""
    
    def get_dtype(self, name):
        """Get a data type by name."""
        return get_dtype(name)
    
    def to_dtype_str(self, dtype):
        """Convert an Ember data type to a string representation."""
        return to_dtype_str(dtype)
    
    def from_dtype_str(self, dtype_str):
        """Convert a string representation to an Ember data type."""
        return from_dtype_str(dtype_str)
    
    @property
    def float32(self):
        """Get the float32 data type."""
        return float32()
    
    @property
    def float64(self):
        """Get the float64 data type."""
        return float64()
    
    @property
    def float16(self):
        """Get the float16 data type."""
        return float16()
    
    @property
    def int32(self):
        """Get the int32 data type."""
        return int32()
    
    @property
    def int64(self):
        """Get the int64 data type."""
        return int64()
    
    @property
    def int16(self):
        """Get the int16 data type."""
        return int16()
    
    @property
    def int8(self):
        """Get the int8 data type."""
        return int8()
    
    @property
    def uint8(self):
        """Get the uint8 data type."""
        return uint8()
    
    @property
    def uint16(self):
        """Get the uint16 data type."""
        return uint16()
    
    @property
    def uint32(self):
        """Get the uint32 data type."""
        return uint32()
    
    @property
    def uint64(self):
        """Get the uint64 data type."""
        return uint64()
    
    @property
    def bool_(self):
        """Get the boolean data type."""
        return bool_()