"""
Type definitions for NumPy tensor operations.

This module provides standard type aliases for tensor operations in the Ember ML framework.
These type aliases ensure consistent type annotations across the codebase and
help with static type checking.
"""

from typing import (
    Any, List, Optional, Sequence, Tuple, Union, Literal,
    TYPE_CHECKING
)
import os

import numpy as np


# Basic type aliases that don't require imports
type Numeric = Union[int, float]
type OrdLike = Optional[Union[int, str]]
type Device = Optional[str]
type PathLike = Union[str, os.PathLike[str]]
type Shape = Sequence[int]
type ShapeType = Union[int, Tuple[int, ...], List[int]]
type ShapeLike = Union[int, List[int], Tuple[int, ...], Shape]
type DimSize = Union[int, 'np.ndarray']
type Axis = Optional[Union[int, Sequence[int]]]
type IndexType = Union[int, Sequence[int], 'np.ndarray'] 
type Indices = Union[Sequence[int], 'np.ndarray']
# Numpy specific
type NumpyArray = np.ndarray
type DTypeClass = np.dtype

# Precision related
default_int = np.int32
default_float = np.float32
default_bool = np.bool_ if hasattr(np, 'bool_') else Any


# Runtime definitions (simplified)
type TensorTypes = Any
type ArrayLike = Any
type TensorLike = Any
type ScalarLike = Any
type DTypes = Any
type DType = Any

# Conditional type definitions
if TYPE_CHECKING == True:
    # These imports are for type checking only
    # Must be done inside TYPE_CHECKING block to avoid circular imports
    from typing import TypeVar
    T = TypeVar('T')  # Used for generic type definitions
    
    # Define types that reference external modules
    type TensorTypes = Union[
        NumpyArray,
        Any,  # NumpyTensor
        Any,  # EmberTensor
        Any,  # numpy.ndarray
    ]
    
    type ArrayLike = Union[
        Any,  # NumpyTensor
        NumpyArray, 
        Numeric, 
        List[Any], 
        Tuple[Any, ...]
    ]
    
    type DTypes = Union[
        np.dtype,
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
        NumpyArray,
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
    'NumpyArray',
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


