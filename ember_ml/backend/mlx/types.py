"""
Type definitions for MLX tensor operations.

This module provides standard type aliases for tensor operations in the Ember ML framework.
These type aliases ensure consistent type annotations across the codebase and
help with static type checking.
"""

from typing import (
    Any, List, Optional, Sequence, Tuple, Union,
    TYPE_CHECKING
)
import os

import mlx.core as mx

# Basic type aliases that don't require imports
type Numeric = Union[int, float]
type OrdLike = Optional[Union[int, str]]
type Device = Optional[str]
type PathLike = Union[str, os.PathLike[str]]
type Shape = Sequence[int]
type ShapeType = Union[int, Tuple[int, ...], List[int]]
type ShapeLike = Union[int, List[int], Tuple[int, ...], Shape]
type DimSize = Union[int, 'mx.array']
type Axis = Optional[Union[int, Sequence[int]]]
type IndexType = Union[int, Sequence[int], 'mx.array'] 
type Indices = Union[Sequence[int], 'mx.array']
type TensorLike = Optional[Union[
    Numeric,
    bool,
    List[Any],
    Tuple[Any, ...],
    'MLXArray'
]]
type ScalarLike = Optional[Union[
    Numeric,
    bool,
    'MLXArray',
    'TensorLike'
]]

# MLX specific
type MLXArray = mx.array
type DTypeClass = mx.Dtype

# Precision related
default_int = mx.int32
default_float = mx.float32
default_bool = mx.bool_ if hasattr(mx, 'bool_') else Any


# Default type for dtype
type DType = Any

# Conditional type definitions
if TYPE_CHECKING == True:
    # These imports are for type checking only
    # Must be done inside TYPE_CHECKING block to avoid circular imports
    from typing import TypeVar
    T = TypeVar('T')  # Used for generic type definitions
    
    # Define types that reference external modules
    type TensorTypes = Union[
        MLXArray,
        Any,  # MLXTensor
        Any,  # EmberTensor
        Any,  # numpy.ndarray
    ]
    
    type ArrayLike = Union[
        Any,  # MLXTensor
        MLXArray, 
        Numeric, 
        List[Any], 
        Tuple[Any, ...]
    ]
    
    type DTypes = Union[
        mx.Dtype,
        Any,  # numpy.dtype
    ]
    
    type TensorLike = Optional[Union[
        Numeric,
        bool,
        List[Any],
        Tuple[Any, ...],
        'TensorTypes'
    ]]
    
    type ScalarLike = Optional[Union[
        Numeric,
        bool,
        MLXArray,
        'TensorLike'
    ]]


__all__ = [
    'Numeric',
    'TensorLike',
    'Shape',
    'ShapeType', 
    'ShapeLike',
    'DTypeClass',
    'DTypes',
    'MLXArray',
    'ArrayLike', 
    'TensorTypes',
    'DimSize',
    'Axis',
    'ScalarLike',
    'OrdLike',
    'Device',
    'IndexType',
    'Indices',
    'PathLike'
]


