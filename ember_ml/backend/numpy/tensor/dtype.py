"""
NumPy data type implementation for ember_ml.

This module provides NumPy implementations of data type operations.
"""

import numpy as np
from typing import Union, Any, Optional

class NumpyDType:
    """NumPy implementation of data type operations."""

    @property
    def float16(self):
        """Get the float16 data type."""
        return np.float16
    
    @property
    def float32(self):
        """Get the float32 data type."""
        return np.float32
    
    @property
    def float64(self):
        """Get the float64 data type."""
        return np.float64
    
    @property
    def int8(self):
        """Get the int8 data type."""
        return np.int8
    
    @property
    def int16(self):
        """Get the int16 data type."""
        return np.int16
    
    @property
    def int32(self):
        """Get the int32 data type."""
        return np.int32
    
    @property
    def int64(self):
        """Get the int64 data type."""
        return np.int64
    
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
    def bool_(self):
        """Get the boolean data type."""
        return np.bool_
    
    def get_dtype(self, name):
        """Get a data type by name."""
        return self.from_dtype_str(name)

    def to_dtype_str(self, dtype: Union[Any, str, None]) -> Optional[str]:
        """
        Convert a NumPy data type to a dtype string.
        
        Args:
            dtype: The NumPy data type to convert
            
        Returns:
            The corresponding dtype string
        """
        if dtype is None:
            return None

        # If it's already a string, return it
        if isinstance(dtype, str):
            return dtype
            
        # Map NumPy dtypes to EmberDType names
        dtype_map = {
            np.float16: 'float16',
            np.float32: 'float32',
            np.float64: 'float64',
            np.int8: 'int8',
            np.int16: 'int16',
            np.int32: 'int32',
            np.int64: 'int64',
            np.uint8: 'uint8',
            np.uint16: 'uint16',
            np.uint32: 'uint32',
            np.uint64: 'uint64',
            np.bool_: 'bool'
        }

        if dtype in dtype_map:
            return dtype_map[dtype]
        else:
            raise ValueError(f"Cannot convert {dtype} to EmberDType")
    
    def validate_dtype(self, dtype: Optional[Any]) -> Optional[Any]:
        """
        Validate and convert dtype to NumPy format.
        
        Args:
            dtype: Input dtype to validate
            
        Returns:
            Validated NumPy dtype or None
        """
        if dtype is None:
            return None
        
        # Handle string dtypes
        if isinstance(dtype, str):
            return self.from_dtype_str(dtype)
            
        # Handle EmberDType objects
        if hasattr(dtype, 'name'):
            return self.from_dtype_str(str(dtype.name))
            
        # If it's already a NumPy dtype, return as is
        if isinstance(dtype, np.dtype) or dtype in [np.float32, np.float64, np.int32, np.int64,
                                                  np.bool_, np.int8, np.int16, np.uint8,
                                                  np.uint16, np.uint32, np.uint64, np.float16]:
            return dtype
            
        raise ValueError(f"Invalid dtype: {dtype}")
    
    def from_dtype_str(self, dtype: Union[Any, str, None]) -> Optional[Any]:
        """
        Convert a dtype string to a NumPy data type.
        
        Args:
            dtype: The dtype string to convert
            
        Returns:
            The corresponding NumPy data type
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
        if dtype_name == 'float32':
            return np.float32
        elif dtype_name == 'float64':
            return np.float64
        elif dtype_name == 'int32':
            return np.int32
        elif dtype_name == 'int64':
            return np.int64
        elif dtype_name == 'bool' or dtype_name == 'bool_':
            return np.bool_
        elif dtype_name == 'int8':
            return np.int8
        elif dtype_name == 'int16':
            return np.int16
        elif dtype_name == 'uint8':
            return np.uint8
        elif dtype_name == 'uint16':
            return np.uint16
        elif dtype_name == 'uint32':
            return np.uint32
        elif dtype_name == 'uint64':
            return np.uint64
        elif dtype_name == 'float16':
            return np.float16
        elif dtype_name == 'complex64':
            return np.complex64
        else:
            raise ValueError(f"Unknown data type: {dtype_name}")