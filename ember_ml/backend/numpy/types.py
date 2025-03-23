"""
Type definitions for NumPy tensor operations.

This module provides standard type aliases for tensor operations in the Ember ML framework.
These type aliases ensure consistent type annotations across the codebase and
help with static type checking.
"""

from typing import (
    Any, List, Optional, Sequence, Tuple, Union,
    TYPE_CHECKING
)
import os
from types import ModuleType
import numpy
import numpy as np

# Basic type aliases
type Numeric = Union[int, float]

# Type definitions for MLX dtypes
type DTypeStr = str
type DTypeClass = Union[numpy.dtype, str, None]

# Type alias for dtype arguments that maintains compatibility
# with both MLX's dtype system and tensor.py's DType
type DType = Any  # Using Any for maximum compatibility

# MLX array types
type NumpyArray = numpy.ndarray

type ArrayLike = Union[NumpyArray, Numeric, List[Any], Tuple[Any, ...]]

type TensorTypes = Any
# Conditional imports
if TYPE_CHECKING:
    from ember_ml.nn.tensor.common.ember_tensor import EmberTensor
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    from ember_ml.backend.torch.tensor.tensor import TorchTensor

    type TensorTypes = Union[
        np.ndarray,
        NumpyArray,
        NumpyTensor,
        EmberTensor,
        MLXTensor,
        TorchTensor
    ]

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
type DimSize = Union[int, NumpyArray]
type Axis = Optional[Union[int, Sequence[int]]]

# Scalar types
type ScalarLike = Optional[Union[
    Numeric,
    bool,
    NumpyArray,
    TensorTypes
]]

# OrdLike
type OrdLike = Optional[Union[int, str]]

# Device type
type Device = Optional[str]

# Index types
type IndexType = Union[int, Sequence[int], NumpyArray]
type Indices = Union[Sequence[int], NumpyArray]

# File paths
type PathLike = Union[str, os.PathLike[str]]

__all__ = [
    'Numeric',
    'TensorLike',
    'Shape',
    'ShapeType',
    'ShapeLike',
    'ScalarLike',
    'NumpyArray',
    'ArrayLike',
    'Device',
    'DType',
    'DTypeStr',
    'DTypeClass',
    'DimSize',
    'Axis',
    'IndexType',
    'Indices',
    'PathLike',
    'T',
    'SupportsDType',
    'SupportsItem',
    'SupportsAsType',
    'SupportsToList'
]