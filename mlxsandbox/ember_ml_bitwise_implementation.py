"""
Implementation plan for adding bitwise operations to Ember ML.

This file demonstrates how to implement bitwise operations in Ember ML,
following the pattern of the existing linear algebra operations.
"""

# ----------------------------------------------------------------
# 1. Interface Definition (ops/interfaces/bitwise_ops.py)
# ----------------------------------------------------------------

"""
Interface for bitwise operations in Ember ML.

This module defines the interface for bitwise operations that will be
implemented by each backend (NumPy, PyTorch, MLX).
"""

from abc import ABC, abstractmethod
from typing import Any, Union, List

class BitwiseOps(ABC):
    """Interface for bitwise operations."""
    
    @abstractmethod
    def bitwise_and(self, x: Any, y: Any) -> Any:
        """
        Compute the bitwise AND of x and y element-wise.
        
        Args:
            x: First input tensor
            y: Second input tensor
            
        Returns:
            Tensor with the bitwise AND of x and y
        """
        pass
    
    @abstractmethod
    def bitwise_or(self, x: Any, y: Any) -> Any:
        """
        Compute the bitwise OR of x and y element-wise.
        
        Args:
            x: First input tensor
            y: Second input tensor
            
        Returns:
            Tensor with the bitwise OR of x and y
        """
        pass
    
    @abstractmethod
    def bitwise_xor(self, x: Any, y: Any) -> Any:
        """
        Compute the bitwise XOR of x and y element-wise.
        
        Args:
            x: First input tensor
            y: Second input tensor
            
        Returns:
            Tensor with the bitwise XOR of x and y
        """
        pass
    
    @abstractmethod
    def bitwise_not(self, x: Any) -> Any:
        """
        Compute the bitwise NOT of x element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with the bitwise NOT of x
        """
        pass
    
    @abstractmethod
    def left_shift(self, x: Any, shifts: Any) -> Any:
        """
        Shift the bits of x to the left by shifts positions.
        
        Args:
            x: Input tensor
            shifts: Number of bits to shift
            
        Returns:
            Tensor with x shifted left by shifts bits
        """
        pass
    
    @abstractmethod
    def right_shift(self, x: Any, shifts: Any) -> Any:
        """
        Shift the bits of x to the right by shifts positions.
        
        Args:
            x: Input tensor
            shifts: Number of bits to shift
            
        Returns:
            Tensor with x shifted right by shifts bits
        """
        pass
    
    @abstractmethod
    def count_ones(self, x: Any) -> Any:
        """
        Count the number of 1 bits in each element of x.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with the count of 1 bits in each element of x
        """
        pass
    
    @abstractmethod
    def count_zeros(self, x: Any) -> Any:
        """
        Count the number of 0 bits in each element of x.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with the count of 0 bits in each element of x
        """
        pass
    
    @abstractmethod
    def rotate_left(self, x: Any, shifts: Any, bit_width: int = 32) -> Any:
        """
        Rotate the bits of x to the left by shifts positions.
        
        Args:
            x: Input tensor
            shifts: Number of bits to rotate
            bit_width: Width of the bit representation
            
        Returns:
            Tensor with x rotated left by shifts bits
        """
        pass
    
    @abstractmethod
    def rotate_right(self, x: Any, shifts: Any, bit_width: int = 32) -> Any:
        """
        Rotate the bits of x to the right by shifts positions.
        
        Args:
            x: Input tensor
            shifts: Number of bits to rotate
            bit_width: Width of the bit representation
            
        Returns:
            Tensor with x rotated right by shifts bits
        """
        pass
    
    @abstractmethod
    def get_bit(self, x: Any, position: Any) -> Any:
        """
        Get the bit at the specified position in x.
        
        Args:
            x: Input tensor
            position: Bit position (0-based, from least significant bit)
            
        Returns:
            Tensor with the bit at the specified position in x
        """
        pass
    
    @abstractmethod
    def set_bit(self, x: Any, position: Any, value: Any) -> Any:
        """
        Set the bit at the specified position in x to value.
        
        Args:
            x: Input tensor
            position: Bit position (0-based, from least significant bit)
            value: Bit value (0 or 1)
            
        Returns:
            Tensor with the bit at the specified position in x set to value
        """
        pass
    
    @abstractmethod
    def toggle_bit(self, x: Any, position: Any) -> Any:
        """
        Toggle the bit at the specified position in x.
        
        Args:
            x: Input tensor
            position: Bit position (0-based, from least significant bit)
            
        Returns:
            Tensor with the bit at the specified position in x toggled
        """
        pass
    
    @abstractmethod
    def binary_wave_interference(self, waves: List[Any], mode: str = 'xor') -> Any:
        """
        Apply wave interference between multiple binary patterns.
        
        Args:
            waves: List of binary wave patterns
            mode: Interference type ('xor', 'and', or 'or')
            
        Returns:
            Interference pattern
        """
        pass
    
    @abstractmethod
    def binary_wave_propagate(self, wave: Any, shift: Any) -> Any:
        """
        Propagate a binary wave by shifting it.
        
        Args:
            wave: Binary wave pattern
            shift: Number of positions to shift
            
        Returns:
            Propagated wave pattern
        """
        pass
    
    @abstractmethod
    def create_duty_cycle(self, length: Any, duty_cycle: Any) -> Any:
        """
        Create a binary pattern with the specified duty cycle.
        
        Args:
            length: Length of the pattern
            duty_cycle: Fraction of bits that should be 1
            
        Returns:
            Binary pattern with the specified duty cycle
        """
        pass
    
    @abstractmethod
    def generate_blocky_sin(self, length: Any, half_period: Any) -> Any:
        """
        Generate a blocky sine wave pattern.
        
        Args:
            length: Length of the pattern
            half_period: Half the period of the wave
            
        Returns:
            Blocky sine wave pattern
        """
        pass

# ----------------------------------------------------------------
# 2. Frontend Exposure (ops/bitwise/__init__.py)
# ----------------------------------------------------------------

"""
Bitwise operations module.

This module dynamically aliases functions from the active backend
(NumPy, PyTorch, MLX) upon import to provide a consistent `ops.bitwise.*` interface.
"""

import importlib
import sys
import os
from typing import List, Optional, Callable, Any

# Import backend control functions
from ember_ml.backend import get_backend, get_backend_module

# Master list of bitwise operations expected to be aliased
_BITWISE_OPS_LIST = [
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not',
    'left_shift', 'right_shift', 'count_ones', 'count_zeros',
    'rotate_left', 'rotate_right', 'get_bit', 'set_bit', 'toggle_bit',
    'binary_wave_interference', 'binary_wave_propagate',
    'create_duty_cycle', 'generate_blocky_sin'
]

def get_bitwise_module():
    """Imports the bitwise operations from the active backend module."""
    backend_name = get_backend()
    module_name = get_backend_module().__name__ + '.bitwise'
    module = importlib.import_module(module_name)
    return module

# Placeholder initialization
for _op_name in _BITWISE_OPS_LIST:
    if _op_name not in globals():
        globals()[_op_name] = None

# Keep track if aliases have been set for the current backend
_aliased_backend_bitwise: Optional[str] = None

def _update_bitwise_aliases():
    """Dynamically updates this module's namespace with backend bitwise functions."""
    global _aliased_backend_bitwise
    backend_name = get_backend()

    # Avoid re-aliasing if backend hasn't changed since last update for this module
    if backend_name == _aliased_backend_bitwise:
        return

    backend_module = get_bitwise_module()
    current_module = sys.modules[__name__]
    missing_ops = []

    for func_name in _BITWISE_OPS_LIST:
        try:
            backend_function = getattr(backend_module, func_name)
            setattr(current_module, func_name, backend_function)
            globals()[func_name] = backend_function
        except AttributeError:
            setattr(current_module, func_name, None)
            globals()[func_name] = None
            missing_ops.append(func_name)

    if missing_ops:
        # Suppress warning here as ops/__init__ might also warn
        pass
    _aliased_backend_bitwise = backend_name

# --- Initial alias setup ---
# Populate aliases when this module is first imported.
# Relies on the backend having been determined by prior imports.
_update_bitwise_aliases()

# --- Define __all__ ---
__all__ = _BITWISE_OPS_LIST

# ----------------------------------------------------------------
# 3. Type Definitions (ops/bitwise/__init__.pyi)
# ----------------------------------------------------------------

"""
Type stub file for ember_ml.ops.bitwise module.

This provides explicit type hints for bitwise operations,
allowing type checkers to recognize them properly.
"""

from typing import List, Optional, Any, Union, Tuple, Literal

from ember_ml.backend.mlx.types import TensorLike
type Tensor = Any

# Basic bitwise operations
def bitwise_and(x: TensorLike, y: TensorLike) -> Tensor: ...
def bitwise_or(x: TensorLike, y: TensorLike) -> Tensor: ...
def bitwise_xor(x: TensorLike, y: TensorLike) -> Tensor: ...
def bitwise_not(x: TensorLike) -> Tensor: ...

# Shift operations
def left_shift(x: TensorLike, shifts: TensorLike) -> Tensor: ...
def right_shift(x: TensorLike, shifts: TensorLike) -> Tensor: ...
def rotate_left(x: TensorLike, shifts: TensorLike, bit_width: int = 32) -> Tensor: ...
def rotate_right(x: TensorLike, shifts: TensorLike, bit_width: int = 32) -> Tensor: ...

# Bit counting operations
def count_ones(x: TensorLike) -> Tensor: ...
def count_zeros(x: TensorLike) -> Tensor: ...

# Bit manipulation operations
def get_bit(x: TensorLike, position: TensorLike) -> Tensor: ...
def set_bit(x: TensorLike, position: TensorLike, value: TensorLike) -> Tensor: ...
def toggle_bit(x: TensorLike, position: TensorLike) -> Tensor: ...

# Binary wave operations
def binary_wave_interference(waves: List[TensorLike], mode: str = 'xor') -> Tensor: ...
def binary_wave_propagate(wave: TensorLike, shift: TensorLike) -> Tensor: ...
def create_duty_cycle(length: TensorLike, duty_cycle: TensorLike) -> Tensor: ...
def generate_blocky_sin(length: TensorLike, half_period: TensorLike) -> Tensor: ...

# ----------------------------------------------------------------
# 4. Backend Implementation (backend/numpy/bitwise/__init__.py)
# ----------------------------------------------------------------

"""NumPy bitwise operations for ember_ml."""

from ember_ml.backend.numpy.bitwise.basic_ops import bitwise_and, bitwise_or, bitwise_xor, bitwise_not
from ember_ml.backend.numpy.bitwise.shift_ops import left_shift, right_shift, rotate_left, rotate_right
from ember_ml.backend.numpy.bitwise.bit_ops import count_ones, count_zeros, get_bit, set_bit, toggle_bit
from ember_ml.backend.numpy.bitwise.wave_ops import binary_wave_interference, binary_wave_propagate, create_duty_cycle, generate_blocky_sin

__all__ = [
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    "left_shift",
    "right_shift",
    "rotate_left",
    "rotate_right",
    "count_ones",
    "count_zeros",
    "get_bit",
    "set_bit",
    "toggle_bit",
    "binary_wave_interference",
    "binary_wave_propagate",
    "create_duty_cycle",
    "generate_blocky_sin"
]

# ----------------------------------------------------------------
# 5. Backend Implementation (backend/numpy/bitwise/basic_ops.py)
# ----------------------------------------------------------------

"""
NumPy basic bitwise operations for ember_ml.

This module provides NumPy implementations of basic bitwise operations.
"""

import numpy as np
from typing import Any, List

from ember_ml.backend.numpy.types import TensorLike
from ember_ml.backend.numpy.tensor import NumpyTensor

def bitwise_and(x: TensorLike, y: TensorLike) -> TensorLike:
    """
    Compute the bitwise AND of x and y element-wise.
    
    Args:
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Tensor with the bitwise AND of x and y
    """
    # Convert inputs to NumPy arrays
    Tensor = NumpyTensor()
    x_array = Tensor.convert_to_tensor(x)
    y_array = Tensor.convert_to_tensor(y)
    
    # Use NumPy's built-in bitwise_and function
    return np.bitwise_and(x_array, y_array)

def bitwise_or(x: TensorLike, y: TensorLike) -> TensorLike:
    """
    Compute the bitwise OR of x and y element-wise.
    
    Args:
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Tensor with the bitwise OR of x and y
    """
    # Convert inputs to NumPy arrays
    Tensor = NumpyTensor()
    x_array = Tensor.convert_to_tensor(x)
    y_array = Tensor.convert_to_tensor(y)
    
    # Use NumPy's built-in bitwise_or function
    return np.bitwise_or(x_array, y_array)

def bitwise_xor(x: TensorLike, y: TensorLike) -> TensorLike:
    """
    Compute the bitwise XOR of x and y element-wise.
    
    Args:
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Tensor with the bitwise XOR of x and y
    """
    # Convert inputs to NumPy arrays
    Tensor = NumpyTensor()
    x_array = Tensor.convert_to_tensor(x)
    y_array = Tensor.convert_to_tensor(y)
    
    # Use NumPy's built-in bitwise_xor function
    return np.bitwise_xor(x_array, y_array)

def bitwise_not(x: TensorLike) -> TensorLike:
    """
    Compute the bitwise NOT of x element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with the bitwise NOT of x
    """
    # Convert input to NumPy array
    Tensor = NumpyTensor()
    x_array = Tensor.convert_to_tensor(x)
    
    # Use NumPy's built-in invert function
    return np.invert(x_array)

# ----------------------------------------------------------------
# 6. Backend Implementation (backend/torch/bitwise/basic_ops.py)
# ----------------------------------------------------------------

"""
PyTorch basic bitwise operations for ember_ml.

This module provides PyTorch implementations of basic bitwise operations.
"""

import torch
from typing import Any, List

from ember_ml.backend.torch.types import TensorLike
from ember_ml.backend.tensor.convert_to_tensor import TorchTensor

def bitwise_and(x: TensorLike, y: TensorLike) -> tensor.convert_to_tensor:
    """
    Compute the bitwise AND of x and y element-wise.
    
    Args:
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Tensor with the bitwise AND of x and y
    """
    # Convert inputs to PyTorch tensors
    Tensor = TorchTensor()
    x_tensor = Tensor.convert_to_tensor(x)
    y_tensor = Tensor.convert_to_tensor(y)
    
    # Use PyTorch's built-in bitwise_and function
    return torch.bitwise_and(x_tensor, y_tensor)

def bitwise_or(x: TensorLike, y: TensorLike) -> tensor.convert_to_tensor:
    """
    Compute the bitwise OR of x and y element-wise.
    
    Args:
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Tensor with the bitwise OR of x and y
    """
    # Convert inputs to PyTorch tensors
    Tensor = TorchTensor()
    x_tensor = Tensor.convert_to_tensor(x)
    y_tensor = Tensor.convert_to_tensor(y)
    
    # Use PyTorch's built-in bitwise_or function
    return torch.bitwise_or(x_tensor, y_tensor)

def bitwise_xor(x: TensorLike, y: TensorLike) -> tensor.convert_to_tensor:
    """
    Compute the bitwise XOR of x and y element-wise.
    
    Args:
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Tensor with the bitwise XOR of x and y
    """
    # Convert inputs to PyTorch tensors
    Tensor = TorchTensor()
    x_tensor = Tensor.convert_to_tensor(x)
    y_tensor = Tensor.convert_to_tensor(y)
    
    # Use PyTorch's built-in bitwise_xor function
    return torch.bitwise_xor(x_tensor, y_tensor)

def bitwise_not(x: TensorLike) -> tensor.convert_to_tensor:
    """
    Compute the bitwise NOT of x element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with the bitwise NOT of x
    """
    # Convert input to PyTorch tensor
    Tensor = TorchTensor()
    x_tensor = Tensor.convert_to_tensor(x)
    
    # Use PyTorch's built-in bitwise_not function
    return torch.bitwise_not(x_tensor)

# ----------------------------------------------------------------
# 7. Backend Implementation (backend/mlx/bitwise/basic_ops.py)
# ----------------------------------------------------------------

"""
MLX basic bitwise operations for ember_ml.

This module provides MLX implementations of basic bitwise operations.
"""

import mlx.core as mx
from typing import Any, List

from ember_ml.backend.mlx.types import TensorLike
from ember_ml.backend.mlx.tensor import MLXTensor

def bitwise_and(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Compute the bitwise AND of x and y element-wise.
    
    Args:
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Tensor with the bitwise AND of x and y
    """
    # Convert inputs to MLX arrays
    Tensor = MLXTensor()
    x_array = Tensor.convert_to_tensor(x)
    y_array = Tensor.convert_to_tensor(y)
    
    # Use MLX's built-in bitwise_and function
    return mx.bitwise_and(x_array, y_array)

def bitwise_or(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Compute the bitwise OR of x and y element-wise.
    
    Args:
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Tensor with the bitwise OR of x and y
    """
    # Convert inputs to MLX arrays
    Tensor = MLXTensor()
    x_array = Tensor.convert_to_tensor(x)
    y_array = Tensor.convert_to_tensor(y)
    
    # Use MLX's built-in bitwise_or function
    return mx.bitwise_or(x_array, y_array)

def bitwise_xor(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Compute the bitwise XOR of x and y element-wise.
    
    Args:
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Tensor with the bitwise XOR of x and y
    """
    # Convert inputs to MLX arrays
    Tensor = MLXTensor()
    x_array = Tensor.convert_to_tensor(x)
    y_array = Tensor.convert_to_tensor(y)
    
    # Use MLX's built-in bitwise_xor function
    return mx.bitwise_xor(x_array, y_array)

def bitwise_not(x: TensorLike) -> mx.array:
    """
    Compute the bitwise NOT of x element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with the bitwise NOT of x
    """
    # Convert input to MLX array
    Tensor = MLXTensor()
    x_array = Tensor.convert_to_tensor(x)
    
    # Use MLX's built-in bitwise_not function
    return mx.bitwise_not(x_array)

# ----------------------------------------------------------------
# 8. Usage Example
# ----------------------------------------------------------------

"""
Example usage of bitwise operations in Ember ML.
"""

from ember_ml import ops
from ember_ml.nn import tensor

def example_usage():
    """Example usage of bitwise operations."""
    # Create tensors
    a = tensor.convert_to_tensor([0b1010, 0b1100, 0b1111])
    b = tensor.convert_to_tensor([0b0101, 0b1010, 0b0000])
    
    # Basic bitwise operations
    c = ops.bitwise.bitwise_and(a, b)
    d = ops.bitwise.bitwise_or(a, b)
    e = ops.bitwise.bitwise_xor(a, b)
    f = ops.bitwise.bitwise_not(a)
    
    # Shift operations
    g = ops.bitwise.left_shift(a, tensor.convert_to_tensor([1, 2, 3]))
    h = ops.bitwise.right_shift(a, tensor.convert_to_tensor([1, 1, 1]))
    
    # Bit manipulation operations
    i = ops.bitwise.get_bit(a, tensor.convert_to_tensor([1, 1, 1]))
    j = ops.bitwise.set_bit(a, tensor.convert_to_tensor([2, 2, 2]), tensor.convert_to_tensor([1, 1, 1]))
    k = ops.bitwise.toggle_bit(a, tensor.convert_to_tensor([0, 0, 0]))
    
    # Binary wave operations
    l = ops.bitwise.binary_wave_interference([a, b], mode='xor')
    m = ops.bitwise.binary_wave_propagate(a, tensor.convert_to_tensor([1, 1, 1]))
    n = ops.bitwise.create_duty_cycle(tensor.convert_to_tensor([8, 8, 8]), tensor.convert_to_tensor([0.5, 0.25, 0.75]))
    o = ops.bitwise.generate_blocky_sin(tensor.convert_to_tensor([8, 8, 8]), tensor.convert_to_tensor([2, 2, 2]))
    
    return c, d, e, f, g, h, i, j, k, l, m, n, o

if __name__ == "__main__":
    example_usage()