"""
Type definitions for MLX tensor operations.

This module provides standard type aliases for tensor operations in the Ember ML framework.
These type aliases ensure consistent type annotations across the codebase and
help with static type checking.
"""

from typing import (
    Any, List, Optional, Sequence, Tuple, Union, Literal,
    TYPE_CHECKING
)
import os

import mlx.core as mx

default_int = mx.int32
default_float = mx.float32
default_bool = mx.bool_

# Conditional imports
if TYPE_CHECKING == True:
    from ember_ml.nn.tensor.common.ember_tensor import EmberTensor
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    from ember_ml.backend.torch.tensor.tensor import TorchTensor
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    import numpy
    # Basic type aliases
    type Numeric = Union[int, float]

    # PyTorch tensor types
    type MLXArray = mx.core.array
    type DTypeClass = mx.core.Dtype
    type ArrayLike = Union[TorchTensor, Numeric, List[Any], Tuple[Any, ...]]
 
    type dtype_int32 = Union[DTypeClass.int32, Literal['int32'] ]
    type dtype_int64 = Union[DTypeClass.int64, Literal['int64'] ]
    type dtype_float32 = Union[DTypeClass.float32, Literal['float32'] ]
    type dtype_float64 = Union[DTypeClass.float64, Literal['float64'] ]
    type dtype_bool = Union[DTypeClass.bool, Literal['bool']]
    type TensorTypes = Union[
        MLXArray,
        MLXTensor,
        EmberTensor,
        NumpyTensor
    ]
    type DTypes = Union[
        DTypeClass,
        numpy.dtype,
        dtype_int32,
        dtype_int64,
        dtype_float32,
        dtype_float64,
        numpy.dtype,
        numpy.int32,
        numpy.int64,
        numpy.float32,
        numpy.float64,
        numpy.bool_,]
else:
    DType = Any  # Using Any for maximum compatibility
    DTypes = Any  # Using Any for maximum compatibility
    TensorLike = Any  # Using Any for maximum compatibility
    Shape = Any  # Using Any for maximum compatibility
    Axis = Any  # Using Any for maximum compatibility
    OrdLike = Any  # Using Any for maximum compatibility
    TensorTypes = Any  # Using Any for maximum compatibility

    # Main type definitions
    type TensorLike = Optional[Union[
        Numeric,
        bool,
        List[Any],
        Tuple[Any, ...],
        TensorTypes
    ]]

    # Shape types
    type Shape = Sequence[int]
    type ShapeType = Union[int, Tuple[int, ...], List[int]]
    type ShapeLike = Union[int, List[int], Tuple[int, ...], Shape]

    # Dimension types
    type DimSize = Union[int, mx.array]
    type Axis = Optional[Union[int, Sequence[int]]]

    # Scalar types
    type ScalarLike = Optional[Union[
        Numeric,
        bool,
        MLXArray,
        TensorTypes
    ]]

    # OrdLike
    type OrdLike = Optional[Union[int, str]]

    # Device type
    type Device = Optional[str]

    # Index types
    type IndexType = Union[int, Sequence[int], MLXArray]
    type Indices = Union[Sequence[int], MLXArray]

    # File paths
    type PathLike = Union[str, os.PathLike[str]]

__all__ = [
    'Numeric',
    'TensorLike',
    'Shape',
    'ShapeType', 
    'ShapeLike',
    'DTypeStr',
    'DTypeClass',
    'DType',
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
