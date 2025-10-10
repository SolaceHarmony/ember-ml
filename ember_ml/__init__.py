"""Public package interface for Ember ML."""

from __future__ import annotations

from ember_ml.backend import (
    auto_select_backend,
    get_available_backends,
    get_backend,
    set_backend,
    using_backend,
)
from ember_ml import ops
from ember_ml.tensor import (
    arange,
    cast,
    convert_to_tensor,
    copy,
    dtype,
    eye,
    full,
    maximum,
    ones,
    reshape,
    shape,
    to_numpy,
    transpose,
    zeros,
)

# Lazily select a backend on import if none has been chosen yet.
if get_backend() is None:
    backend_name, _ = auto_select_backend()
    if backend_name is not None:
        set_backend(backend_name)


def set_seed(seed: int) -> None:
    """Seed random number generators for the active backend."""

    from ember_ml import tensor as tensor_module

    tensor_module.set_seed(seed)


tensor = convert_to_tensor

__all__ = [
    "auto_select_backend",
    "get_available_backends",
    "get_backend",
    "ops",
    "set_backend",
    "set_seed",
    "tensor",
    "using_backend",
    "arange",
    "cast",
    "convert_to_tensor",
    "copy",
    "dtype",
    "eye",
    "full",
    "maximum",
    "ones",
    "reshape",
    "shape",
    "to_numpy",
    "transpose",
    "zeros",
]

__version__ = "0.2.0"
