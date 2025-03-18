"""
Type definitions for MLX tensor operations.

This module provides standard type aliases for tensor operations in the Ember ML framework.
These type aliases ensure consistent type annotations across the codebase and
help with static type checking.
"""

from typing import (
    Any, List, Optional, Protocol, Sequence, Tuple, TypeVar, Union,
    runtime_checkable, TYPE_CHECKING
)
import os
from types import ModuleType
import mlx.core

# Basic type aliases
Numeric = Union[int, float]

# Protocol for type checking
@runtime_checkable
class SupportsDType(Protocol):
    """Protocol for objects that can be used as dtypes."""
    def __str__(self) -> str: ...

@runtime_checkable
class SupportsItem(Protocol):
    """Protocol for objects that support item()."""
    def item(self) -> Any: ...

@runtime_checkable
class SupportsAsType(Protocol):
    """Protocol for objects that support astype()."""
    def astype(self, dtype: Any) -> Any: ...

@runtime_checkable
class SupportsToList(Protocol):
    """Protocol for objects that support tolist()."""
    def tolist(self) -> Any: ...

# Type definitions for MLX dtypes
DTypeStr = str
DTypeClass = Union[mlx.core.Dtype, str, None]

# Type alias for dtype arguments that maintains compatibility
# with both MLX's dtype system and tensor.py's DType
DType = Any  # Using Any for maximum compatibility

# MLX array types
MLXArray = mlx.core.array
ArrayLike = Union[MLXArray, Numeric, List[Any], Tuple[Any, ...]]

# Type variable for generic operations
T = TypeVar('T', bound=Union[Numeric, MLXArray])

# Conditional imports
if TYPE_CHECKING:
    import numpy as np
    from ember_ml.nn.tensor.common.ember_tensor import EmberTensor
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    
    TensorTypes = Union[
        np.ndarray,
        MLXArray,
        MLXTensor,
        EmberTensor
    ]
else:
    TensorTypes = Any

# Main type definitions
TensorLike = Optional[Union[
    Numeric,
    bool,
    List[Any],
    Tuple[Any, ...],
    TensorTypes
]]

# Shape types
Shape = Sequence[int]
ShapeType = Union[int, Tuple[int, ...], List[int]]
ShapeLike = Union[int, List[int], Tuple[int, ...], Shape]

# Dimension types
DimSize = Union[int, MLXArray]
Axis = Optional[Union[int, Sequence[int]]]

# Scalar types
ScalarLike = Optional[Union[
    Numeric,
    bool,
    MLXArray,
    TensorTypes
]]

# Device type
Device = Optional[str]

# Index types
IndexType = Union[int, Sequence[int], MLXArray]
Indices = Union[Sequence[int], MLXArray]

# File paths
PathLike = Union[str, os.PathLike[str]]

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