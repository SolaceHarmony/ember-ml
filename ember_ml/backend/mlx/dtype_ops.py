"""
MLX implementation of data type operations.

This module provides MLX implementations of data type operations.
"""

import mlx.core as mx
from typing import Any

# Type aliases
DType = Any

def get_dtype(dtype_name: str) -> DType:
    """
    Get the MLX data type corresponding to the given name.
    
    Args:
        dtype_name: Name of the data type
        
    Returns:
        MLX data type
    
    Raises:
        ValueError: If the data type name is not recognized
    """
    # Create a dictionary of MLX data types
    dtype_map = {
        'float16': mx.float16,
        'float32': mx.float32,
        'bfloat16': mx.bfloat16,
        'int8': mx.int8,
        'int16': mx.int16,
        'int32': mx.int32,
        'uint8': mx.uint8,
        'uint16': mx.uint16,
        'uint32': mx.uint32,
        'bool': getattr(mx, 'bool_', None)  # Use getattr to avoid AttributeError
    }
    
    # If bool_ is not available, remove it from the map
    if dtype_map['bool'] is None:
        del dtype_map['bool']
    
    if dtype_name in dtype_map:
        return dtype_map[dtype_name]
    else:
        raise ValueError(f"Unknown data type name: {dtype_name}")

def to_dtype_str(dtype: DType) -> str:
    """
    Convert an MLX data type to a string representation.
    
    Args:
        dtype: MLX data type
        
    Returns:
        String representation of the data type
    """
    # Create a dictionary of MLX to string representations
    dtype_map = {
        mx.float16: 'float16',
        mx.float32: 'float32',
        mx.float64: 'float64',
        mx.bfloat16: 'bfloat16',
        mx.int8: 'int8',
        mx.int16: 'int16',
        mx.int32: 'int32',
        mx.int64: 'int64',
        mx.uint8: 'uint8',
        mx.uint16: 'uint16',
        mx.uint32: 'uint32',
        mx.uint64: 'uint64',
        mx.complex64: 'complex64',
    }
    
    # Add boolean type if available
    bool_type = getattr(mx, 'bool_', None)
    if bool_type is not None:
        dtype_map[bool_type] = 'bool'
    
    if dtype in dtype_map:
        return dtype_map[dtype]
    else:
        # Default to float32
        return 'float32'

def from_dtype_str(dtype_str: str) -> DType:
    """
    Convert a string representation to an MLX data type.
    
    Args:
        dtype_str: String representation of the data type
        
    Returns:
        MLX data type
    """
    # Create a dictionary of string representations to MLX data types
    dtype_map = {
        'float16': mx.float16,
        'float32': mx.float32,
        'float64': mx.float64,
        'bfloat16': mx.bfloat16,
        'int8': mx.int8,
        'int16': mx.int16,
        'int32': mx.int32,
        'int64': mx.int64,
        'uint8': mx.uint8,
        'uint16': mx.uint16,
        'uint32': mx.uint32,
        'uint64': mx.uint64,
        'complex64': mx.complex64,
        'bool': getattr(mx, 'bool_', mx.uint8),  # Fallback to uint8 if bool_ not available
    }
    
    if dtype_str in dtype_map:
        return dtype_map[dtype_str]
    else:
        # Default to float32
        return mx.float32


class MLXDTypeOps:
    """MLX implementation of data type operations."""
    
    def get_dtype(self, dtype_name):
        """Get the MLX data type corresponding to the given name."""
        return get_dtype(dtype_name)
    
    def to_dtype_str(self, dtype):
        """Convert an MLX data type to a string representation."""
        return to_dtype_str(dtype)
    
    def from_dtype_str(self, dtype_str):
        """Convert a string representation to an MLX data type."""
        return from_dtype_str(dtype_str)
    
    # Data type properties
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
        bool_type = getattr(mx, 'bool_', None)
        if bool_type is not None:
            return bool_type
        else:
            # Fallback to uint8 if bool_ is not available
            return mx.uint8