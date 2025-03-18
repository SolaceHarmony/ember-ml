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
from types import ModuleType
import mlx.core

# Basic type aliases
type Numeric = Union[int, float]



# Type definitions for MLX dtypes
type DTypeStr = str
type DTypeClass = Union[mlx.core.Dtype, str, None]

# Type alias for dtype arguments that maintains compatibility
# with both MLX's dtype system and tensor.py's DType
type DType = Any  # Using Any for maximum compatibility

# MLX array types
type MLXArray = mlx.core.array
type ArrayLike = Union[MLXArray, Numeric, List[Any], Tuple[Any, ...]]


# Conditional imports
if TYPE_CHECKING:
    import numpy as np
    from ember_ml.nn.tensor.common.ember_tensor import EmberTensor
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    
    type TensorTypes = Union[
        np.ndarray,
        MLXArray,
        MLXTensor,
        EmberTensor
    ]
else:
    type TensorTypes = Any

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
type DimSize = Union[int, MLXArray]
type Axis = Optional[Union[int, Sequence[int]]]

# Scalar types
type ScalarLike = Optional[Union[
    Numeric,
    bool,
    MLXArray,
    TensorTypes
]]

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
    'ScalarLike',
    'MLXArray',
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