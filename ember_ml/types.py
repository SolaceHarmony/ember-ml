"""Core public type aliases for Ember ML frontend.

The goal of this module is to expose stable, backend-agnostic typing helpers
used throughout the high-level API while avoiding hard runtime dependencies
on any specific backend library (NumPy / Torch / MLX).  All heavy backend
imports remain optional and occur only behind ``TYPE_CHECKING`` guards so
that importing ``ember_ml`` does not force-install every backend.

Design principles:
* No runtime imports of backend modules purely for typing.
* Light Protocol-based structural typing where it provides value.
* Broad ``Any`` fallbacks at runtime to keep import cost minimal.
* Forward-compatible: new backends can integrate by matching Protocols.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, Tuple, Union, List, TYPE_CHECKING, runtime_checkable

# Re-exported numeric scalar convenience type
Numeric = Union[int, float]

# Minimal shape alias (list/tuple of ints) kept permissive intentionally
ShapeLike = Union[int, Sequence[int]]

@runtime_checkable
class SupportsDType(Protocol):
    """Protocol for objects exposing a ``dtype`` attribute.

    Backends commonly expose ``.dtype``; using a Protocol allows structural
    typing without importing concrete tensor classes.
    """

    @property
    def dtype(self) -> Any:  # pragma: no cover - structural typing only
        ...


@runtime_checkable
class SupportsDevice(Protocol):
    """Protocol for objects with an optional ``device`` attribute."""

    @property
    def device(self) -> Any:  # pragma: no cover - structural typing only
        ...


@runtime_checkable
class EmberTensorLike(Protocol):
    """Protocol capturing the minimal surface used by high-level ops.

    This intentionally omits numerical operations; those are mediated through
    ``ember_ml.ops`` which handles backend dispatch.  Only metadata queried
    by generic helpers (shape / dtype / device / size) is modeled here.
    """

    @property
    def shape(self) -> Any:  # Typically a tuple[int, ...]
        ...

    @property
    def dtype(self) -> Any:
        ...

    # Optional; some backends expose ``device``
    @property
    def device(self) -> Any:  # pragma: no cover
        ...

    def __array__(self, *args: Any, **kwargs: Any) -> Any:  # NumPy compatibility hook
        ...


# Conditional precise backend-only typing (ignored at runtime)
if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import torch
    import mlx.core as mx
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    from ember_ml.backend.torch.tensor import TorchTensor
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor

    BackendArray = Union[
        np.ndarray,
        torch.Tensor,
        mx.array,
        NumpyTensor,
        TorchTensor,
        MLXTensor,
    ]
else:
    BackendArray = Any  # Fallback placeholder when type checking is off


TensorLike = Optional[Union[
    Numeric,
    bool,
    List[Any],
    Tuple[Any, ...],
    BackendArray,  # Resolved to precise union only during type checking
    EmberTensorLike,
]]

# Public export list for star-import hygiene (optional, kept concise)
__all__ = [
    'Numeric',
    'ShapeLike',
    'SupportsDType',
    'SupportsDevice',
    'EmberTensorLike',
    'TensorLike',
]


#! Legacy compatibility aliases (lightweight)
Scalar = Union[int, float, bool]
Vector = Sequence[Scalar]
Matrix = Sequence[Sequence[Scalar]]
Shape = Union[int, Sequence[int]]
DType = Optional[Union[str, Any]]
Device = Optional[str]
Axis = Optional[Union[int, Sequence[int]]]
ScalarLike = Optional[Union[Numeric, bool]]

__all__ += [
    'Scalar', 'Vector', 'Matrix', 'Shape', 'DType', 'Device', 'Axis', 'ScalarLike'
]
