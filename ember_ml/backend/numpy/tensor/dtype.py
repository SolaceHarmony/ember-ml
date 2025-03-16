"""
NumPy data type implementation for ember_ml.

This module provides NumPy implementations of data type operations.
"""

import numpy as np
from typing import Union, Any, Optional


# Type aliases
DType = Any


class NumpyDType:
    """NumPy implementation of data type operations."""
    
    @property
    def float32(self):
        """Get the float32 data type."""
        return np.float32
    
    @property
    def float64(self):
        """Get the float64 data type."""
        return np.float64
    
    @property
    def int32(self):
        """Get the int32 data type."""
        return np.int32
    
    @property
    def int64(self):
        """Get the int64 data type."""
        return np.int64
    
    @property
    def bool_(self):
        """Get the boolean data type."""
        return np.bool_
    
    @property
    def int8(self):
        """Get the int8 data type."""
        return np.int8
    
    @property
    def int16(self):
        """Get the int16 data type."""
        return np.int16
    
    @property
    def uint8(self):
        """Get the uint8 data type."""
        return np.uint8
    
    @property
    def uint16(self):
        """Get the uint16 data type."""
        return np.uint16
    
    @property
    def uint32(self):
        """Get the uint32 data type."""
        return np.uint32
    
    @property
    def uint64(self):
        """Get the uint64 data type."""
        return np.uint64
    
    @property
    def float16(self):
        """Get the float16 data type."""
        return np.float16
    
    def get_dtype(self, name):
        """
        Get a data type by name.
        
        Args:
            name: The name of the data type
            
        Returns:
            NumPy data type
        """
        return self.from_dtype_str(name)
    
    def to_dtype_str(self, dtype: Union[DType, str, None]) -> Optional[str]:
        """
        Convert a NumPy data type to a string.
        
        Args:
            dtype: The NumPy data type
            
        Returns:
            String representation of the data type
        """
        if dtype is None:
            return None
            
        # If it's already a string, return it
        if isinstance(dtype, str):
            return dtype
            
        # Map NumPy dtypes to EmberDType names
        dtype_map = {
            np.float32: 'float32',
            np.float64: 'float64',
            np.int32: 'int32',
            np.int64: 'int64',
            np.bool_: 'bool',
            np.int8: 'int8',
            np.int16: 'int16',
            np.uint8: 'uint8',
            np.uint16: 'uint16',
            np.uint32: 'uint32',
            np.uint64: 'uint64',
            np.float16: 'float16'
        }
        
        if dtype in dtype_map:
            return dtype_map[dtype]
        else:
            return str(dtype)
    
    def from_dtype_str(self, dtype: Union[Any, str, None]) -> Optional[DType]:
        """
        Convert a string data type to a NumPy data type.
        
        Args:
            dtype: The string data type or EmberDType object
            
        Returns:
            NumPy data type
        """
        if dtype is None:
            return None
            
        # If it's already a NumPy dtype, return it
        if isinstance(dtype, np.dtype) or dtype in [np.float32, np.float64, np.int32, np.int64, 
                                                  np.bool_, np.int8, np.int16, np.uint8, 
                                                  np.uint16, np.uint32, np.uint64, np.float16]:
            return dtype
            
        # If it's a string, use it directly
        if isinstance(dtype, str):
            dtype_name = dtype
        # If it has a name attribute, use that
        elif hasattr(dtype, 'name'):
            dtype_name = dtype.name
        else:
            raise ValueError(f"Cannot convert {dtype} to NumPy data type")
            
        # Map dtype names to NumPy dtypes
        dtype_map = {
            'float32': np.float32,
            'float64': np.float64,
            'int32': np.int32,
            'int64': np.int64,
            'bool': np.bool_,
            'bool_': np.bool_,
            'int8': np.int8,
            'int16': np.int16,
            'uint8': np.uint8,
            'uint16': np.uint16,
            'uint32': np.uint32,
            'uint64': np.uint64,
            'float16': np.float16
        }
        
        if dtype_name in dtype_map:
            return dtype_map[dtype_name]
        
        # Try to convert directly
        try:
            return np.dtype(dtype_name)
        except TypeError:
            raise ValueError(f"Unknown dtype: {dtype_name}")