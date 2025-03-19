"""
Type definitions for PyTorch tensor operations.

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
import torch

# Basic type aliases
type Numeric = Union[int, float]

# Type definitions for PyTorch dtypes
type DTypeStr = str
type DTypeClass = Union[torch.dtype, str, None]

# Type alias for dtype arguments that maintains compatibility
# with both PyTorch's dtype system and tensor.py's DType
type DType = Any  # Using Any for maximum compatibility

# PyTorch tensor types
type TorchTensor = torch.Tensor
type ArrayLike = Union[torch.Tensor, Numeric, List[Any], Tuple[Any, ...]]

type TensorTypes = Any
# Conditional imports
if TYPE_CHECKING:
    import numpy as np
    from ember_ml.nn.tensor.common.ember_tensor import EmberTensor
    from ember_ml.backend.torch.tensor.tensor import TorchTensor
    
    type TensorTypes = Union[
        np.ndarray,
        torch.Tensor,
        TorchTensor,
        EmberTensor
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
type DimSize = Union[int, torch.Tensor]
type Axis = Optional[Union[int, Sequence[int]]]

# Scalar types
type ScalarLike = Optional[Union[
    Numeric,
    bool,
    torch.Tensor,
    TensorTypes
]]

# Device type
type Device = Optional[str]

# Index types
type IndexType = Union[int, Sequence[int], torch.Tensor]
type Indices = Union[Sequence[int], torch.Tensor]

# File paths
type PathLike = Union[str, os.PathLike[str]]

# Type variables for generic functions
from typing import TypeVar
T = TypeVar('T')

# Protocol classes for type checking
from typing import Protocol

class SupportsDType(Protocol):
    @property
    def dtype(self) -> Any: ...

class SupportsItem(Protocol):
    def item(self) -> Union[int, float, bool]: ...

class SupportsAsType(Protocol):
    def astype(self, dtype: Any) -> Any: ...

class SupportsToList(Protocol):
    def tolist(self) -> List[Any]: ...

__all__ = [
    'Numeric',
    'TensorLike',
    'Shape',
    'ShapeType',
    'ShapeLike',
    'ScalarLike',
    'TorchTensor',
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