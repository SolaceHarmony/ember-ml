"""
MLX data type implementation for ember_ml.

This module provides MLX implementations of data type operations.
"""

import mlx.core as mx
from typing import Union, Any, Optional



class MLXDType:
    """MLX implementation of data type operations."""


    @property
    def float16(self):
        """Get the float16 data type."""
        return mx.float16
    
    @property
    def float32(self):
        """Get the float32 data type."""
        return mx.float32
    
    @property
    def float64(self):
        """Get the float64 data type."""
        return mx.float64
    
    @property
    def int8(self):
        """Get the int8 data type."""
        return mx.int8
    
    @property
    def int16(self):
        """Get the int16 data type."""
        return mx.int16
    
    @property
    def int32(self):
        """Get the int32 data type."""
        return mx.int32
    
    @property
    def int64(self):
        """Get the int64 data type."""
        return mx.int64
    
    @property
    def uint8(self):
        """Get the uint8 data type."""
        return mx.uint8
    
    @property 
    def uint16(self):
        """Get the uint16 data type."""
        return mx.uint16
    
    @property
    def uint32(self):
        """Get the uint32 data type."""
        return mx.uint32
    
    @property
    def uint64(self):
        """Get the uint64 data type."""
        return mx.uint64
    
    @property
    def bool_(self):
        """Get the boolean data type."""
        return getattr(mx, 'bool_', mx.uint8)  # Fallback to uint8 if bool_ not available
    
    def get_dtype(self, name):
        """Get a data type by name."""
        return self.from_dtype_str(name)

    def to_dtype_str(self, dtype: Union[Any, str, None]) -> Optional[str]:
        """
        Convert an MLX data type to a dtype string.
        
        Args:
            dtype: The MLX data type to convert
            
        Returns:
            The corresponding dtype string
        """
        if dtype is None:
            return None

        # If it's already a string, return it
        if isinstance(dtype, str):
            return dtype
            
        # Map MLX dtypes to EmberDType names
        dtype_map = {
            mx.float16: 'float16',
            mx.float32: 'float32',
            mx.float64: 'float64',
            mx.int8: 'int8',
            mx.int16: 'int16',
            mx.int32: 'int32',
            mx.int64: 'int64',
            mx.uint8: 'uint8',
            mx.uint16: 'uint16',
            mx.uint32: 'uint32',
            mx.uint64: 'uint64'
        }

        # Add bool type if available
        bool_type = getattr(mx, 'bool_', None)
        if bool_type is not None:
            dtype_map[bool_type] = 'bool'

        if dtype in dtype_map:
            return dtype_map[dtype]
        else:
            raise ValueError(f"Cannot convert {dtype} to EmberDType")
    def validate_dtype(self, dtype: Optional[Any]) -> Optional[Any]:
        """
        Validate and convert dtype to MLX format.
        
        Args:
            dtype_cls: MLXDType instance for conversions
            dtype: Input dtype to validate
            
        Returns:
            Validated MLX dtype or None
        """
        if dtype is None:
            return None
        
        # Handle string dtypes
        if isinstance(dtype, str):
            return self.from_dtype_str(dtype)
            
        # Handle EmberDType objects
        if hasattr(dtype, 'name'):
            return self.from_dtype_str(str(dtype.name))
            
        # If it's already an MLX dtype, return as is
        if isinstance(dtype, type(mx.float32)):
            return dtype
            
        raise ValueError(f"Invalid dtype: {dtype}")
    def from_dtype_str(self, dtype: Union[Any, str, None]) -> Optional[Any]:
        """
        Convert a dtype string to an MLX data type.
        
        Args:
            dtype: The dtype string to convert
            
        Returns:
            The corresponding MLX data type
        """
        if dtype is None:
            return None
            
        # If it's already an MLX dtype, return it
        if isinstance(dtype, type(mx.float32)):  # Using float32 as a reference type
            return dtype
            
        # If it's a string, use it directly
        if isinstance(dtype, str):
            dtype_name = dtype
        # If it has a name attribute, use that
        elif hasattr(dtype, 'name'):
            dtype_name = dtype.name
        else:
            raise ValueError(f"Cannot convert {dtype} to MLX data type")
            
        # Map dtype names to MLX dtypes
        if dtype_name == 'float32':
            return mx.float32
        elif dtype_name == 'float64':
            return mx.float64
        elif dtype_name == 'int32':
            return mx.int32
        elif dtype_name == 'int64':
            return mx.int64
        elif dtype_name == 'bool' or dtype_name == 'bool_':
            return getattr(mx, 'bool_', mx.uint8)  # Fallback to uint8 if bool_ not available
        elif dtype_name == 'int8':
            return mx.int8
        elif dtype_name == 'int16':
            return mx.int16
        elif dtype_name == 'uint8':
            return mx.uint8
        elif dtype_name == 'uint16':
            return mx.uint16
        elif dtype_name == 'uint32':
            return mx.uint32
        elif dtype_name == 'uint64':
            return mx.uint64
        elif dtype_name == 'float16':
            return mx.float16
        elif dtype_name == 'complex64':
            return mx.complex64
        else:
            raise ValueError(f"Unknown data type: {dtype_name}")