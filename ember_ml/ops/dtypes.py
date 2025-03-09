"""
Data types for emberharmony.

This module provides a backend-agnostic data type system that can be used
across different backends (NumPy, PyTorch, MLX).
"""

from typing import Any, Dict, Optional, Union, Type, List
import importlib
import inspect
import sys

import numpy as np

from ember_ml.backend import get_backend

class EmberDtype:
    """
    A backend-agnostic data type.
    
    This class represents data types that can be used across different backends
    (NumPy, PyTorch, MLX).
    """
    
    def __init__(self, name: str, numpy_dtype: Type):
        """
        Initialize an EmberDtype.
        
        Args:
            name: The name of the data type
            numpy_dtype: The corresponding NumPy data type
        """
        self.name = name
        self.numpy_dtype = numpy_dtype
    
    def __repr__(self) -> str:
        """Return a string representation of the data type."""
        return f"EmberDtype({self.name})"
    
    def __str__(self) -> str:
        """Return a string representation of the data type."""
        return self.name
    
    def __eq__(self, other: Any) -> bool:
        """Check if two data types are equal."""
        if isinstance(other, EmberDtype):
            return self.name == other.name
        elif isinstance(other, type):
            return self.numpy_dtype == other
        elif isinstance(other, str):
            return self.name == other
        return False


# Define mapping of common data types to NumPy types
_DTYPE_MAPPING = {
    'float32': np.float32,
    'float64': np.float64,
    'int32': np.int32,
    'int64': np.int64,
    'bool': np.bool_,
    'int8': np.int8,
    'int16': np.int16,
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint32': np.uint32,
    'uint64': np.uint64,
    'float16': np.float16,
}

# Create EmberDtype instances for each data type
float32 = EmberDtype('float32', np.float32)
float64 = EmberDtype('float64', np.float64)
int32 = EmberDtype('int32', np.int32)
int64 = EmberDtype('int64', np.int64)
bool_ = EmberDtype('bool', np.bool_)
int8 = EmberDtype('int8', np.int8)
int16 = EmberDtype('int16', np.int16)
uint8 = EmberDtype('uint8', np.uint8)
uint16 = EmberDtype('uint16', np.uint16)
uint32 = EmberDtype('uint32', np.uint32)
uint64 = EmberDtype('uint64', np.uint64)
float16 = EmberDtype('float16', np.float16)

# Define a list of all available data types
__all__ = [
    'EmberDtype',
    'float32', 'float64', 'int32', 'int64', 'bool_',
    'int8', 'int16', 'uint8', 'uint16', 'uint32', 'uint64', 'float16'
]

# Dynamic attribute lookup to check if a data type is supported by the backend
import types

class DTypeRegistry(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self._registry = {name: dtype for name, dtype in globals().items()
                         if isinstance(dtype, EmberDtype)}
    
    def __getattr__(self, name):
        if name in self._registry:
            return self._registry[name]
        raise AttributeError(f"Data type '{name}' is not supported by the current backend")

# Create a new module instance with our custom class
new_module = DTypeRegistry(__name__)

# Copy all existing attributes to the new module
current_globals = dict(globals())
for name, value in current_globals.items():
    if not name.startswith('__'):
        setattr(new_module, name, value)

# Replace the module in sys.modules
sys.modules[__name__] = new_module