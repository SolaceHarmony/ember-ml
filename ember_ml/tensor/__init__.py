"""Tensor utilities for :mod:`ember_ml`.

This module provides backend-agnostic tensor operations that dispatch to the
currently selected backend (NumPy, PyTorch, MLX).
"""

from typing import Any, Optional
import importlib

from ember_ml.backend import get_backend
from ..dtypes import (
    EmberDType,
    DType,
    get_dtype,
    to_dtype_str,
    from_dtype_str,
    float32,
    float64,
    int32,
    int64,
    bool_,
    int8,
    int16,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
)


# Cache for imported backend tensor ops modules
_BACKEND_TENSOR_OPS_MODULES = {}


def _get_backend_tensor_ops_module():
    """Dynamically import and return the tensor ops module for the current backend."""

    backend_name = get_backend()
    if backend_name not in _BACKEND_TENSOR_OPS_MODULES:
        try:
            module_path = f"ember_ml.backend.{backend_name}.tensor.ops"
            _BACKEND_TENSOR_OPS_MODULES[backend_name] = importlib.import_module(
                module_path
            )
        except ImportError as e:  # pragma: no cover - import error path
            raise ImportError(
                f"Could not import tensor ops module for backend '{backend_name}': {e}"
            )
    return _BACKEND_TENSOR_OPS_MODULES[backend_name]


zeros = lambda *args, **kwargs: _get_backend_tensor_ops_module().zeros(
    *args, **kwargs
)
ones = lambda *args, **kwargs: _get_backend_tensor_ops_module().ones(
    *args, **kwargs
)
zeros_like = lambda *args, **kwargs: _get_backend_tensor_ops_module().zeros_like(
    *args, **kwargs
)
ones_like = lambda *args, **kwargs: _get_backend_tensor_ops_module().ones_like(
    *args, **kwargs
)
eye = lambda *args, **kwargs: _get_backend_tensor_ops_module().eye(
    *args, **kwargs
)
arange = lambda *args, **kwargs: _get_backend_tensor_ops_module().arange(
    *args, **kwargs
)
linspace = lambda *args, **kwargs: _get_backend_tensor_ops_module().linspace(
    *args, **kwargs
)
nonzero = lambda *args, **kwargs: _get_backend_tensor_ops_module().nonzero(
    *args, **kwargs
)
full = lambda *args, **kwargs: _get_backend_tensor_ops_module().full(
    *args, **kwargs
)
full_like = lambda *args, **kwargs: _get_backend_tensor_ops_module().full_like(
    *args, **kwargs
)
reshape = lambda *args, **kwargs: _get_backend_tensor_ops_module().reshape(
    *args, **kwargs
)
transpose = lambda *args, **kwargs: _get_backend_tensor_ops_module().transpose(
    *args, **kwargs
)
concatenate = lambda *args, **kwargs: _get_backend_tensor_ops_module().concatenate(
    *args, **kwargs
)
stack = lambda *args, **kwargs: _get_backend_tensor_ops_module().stack(
    *args, **kwargs
)
split = lambda *args, **kwargs: _get_backend_tensor_ops_module().split(
    *args, **kwargs
)
split_tensor = lambda *args, **kwargs: _get_backend_tensor_ops_module().split_tensor(
    *args, **kwargs
)
expand_dims = lambda *args, **kwargs: _get_backend_tensor_ops_module().expand_dims(
    *args, **kwargs
)
squeeze = lambda *args, **kwargs: _get_backend_tensor_ops_module().squeeze(
    *args, **kwargs
)
tile = lambda *args, **kwargs: _get_backend_tensor_ops_module().tile(
    *args, **kwargs
)
gather = lambda *args, **kwargs: _get_backend_tensor_ops_module().gather(
    *args, **kwargs
)
scatter = lambda *args, **kwargs: _get_backend_tensor_ops_module().scatter(
    *args, **kwargs
)
tensor_scatter_nd_update = (
    lambda *args, **kwargs: _get_backend_tensor_ops_module().tensor_scatter_nd_update(
        *args, **kwargs
    )
)
index_update = lambda *args, **kwargs: _get_backend_tensor_ops_module().index_update(
    *args, **kwargs
)
slice_tensor = lambda *args, **kwargs: _get_backend_tensor_ops_module().slice_tensor(
    *args, **kwargs
)
slice_update = lambda *args, **kwargs: _get_backend_tensor_ops_module().slice_update(
    *args, **kwargs
)
shape = lambda *args, **kwargs: _get_backend_tensor_ops_module().shape(
    *args, **kwargs
)
cast = lambda *args, **kwargs: _get_backend_tensor_ops_module().cast(
    *args, **kwargs
)
copy = lambda *args, **kwargs: _get_backend_tensor_ops_module().copy(
    *args, **kwargs
)
pad = lambda *args, **kwargs: _get_backend_tensor_ops_module().pad(
    *args, **kwargs
)
item = lambda *args, **kwargs: _get_backend_tensor_ops_module().item(
    *args, **kwargs
)
to_numpy = lambda *args, **kwargs: _get_backend_tensor_ops_module().to_numpy(
    *args, **kwargs
)
tolist = lambda *args, **kwargs: getattr(
    _get_backend_tensor_ops_module(), "tolist", lambda x: x.tolist()
)(*args, **kwargs)
random_uniform = (
    lambda *args, **kwargs: _get_backend_tensor_ops_module().random_uniform(
        *args, **kwargs
    )
)
random_normal = (
    lambda *args, **kwargs: _get_backend_tensor_ops_module().random_normal(
        *args, **kwargs
    )
)
maximum = lambda *args, **kwargs: _get_backend_tensor_ops_module().maximum(
    *args, **kwargs
)


def random_bernoulli(*args, **kwargs):
    """Generates Bernoulli random values."""

    seed = kwargs.pop("seed", None)
    if seed is not None:
        set_seed(seed)
    ops_module = _get_backend_tensor_ops_module()
    func = getattr(
        ops_module,
        "random_binomial",
        getattr(ops_module, "random_bernoulli", None),
    )
    if func:
        return func(*args, **kwargs)
    raise AttributeError(
        f"Backend '{get_backend()}' tensor ops module does not have a 'random_binomial' or 'random_bernoulli' function."
    )


random_gamma = lambda *args, **kwargs: _get_backend_tensor_ops_module().random_gamma(
    *args, **kwargs
)
random_exponential = (
    lambda *args, **kwargs: _get_backend_tensor_ops_module().random_exponential(
        *args, **kwargs
    )
)
random_poisson = lambda *args, **kwargs: _get_backend_tensor_ops_module().random_poisson(
    *args, **kwargs
)
random_categorical = (
    lambda *args, **kwargs: _get_backend_tensor_ops_module().random_categorical(
        *args, **kwargs
    )
)
random_permutation = (
    lambda *args, **kwargs: _get_backend_tensor_ops_module().random_permutation(
        *args, **kwargs
    )
)
shuffle = lambda *args, **kwargs: _get_backend_tensor_ops_module().shuffle(
    *args, **kwargs
)
random_shuffle = (
    lambda *args, **kwargs: _get_backend_tensor_ops_module().random_shuffle(
        *args, **kwargs
    )
)
set_seed = lambda *args, **kwargs: _get_backend_tensor_ops_module().set_seed(
    *args, **kwargs
)
get_seed = lambda *args, **kwargs: _get_backend_tensor_ops_module().get_seed(
    *args, **kwargs
)
meshgrid = lambda *args, **kwargs: _get_backend_tensor_ops_module().meshgrid(
    *args, **kwargs
)


class Index:
    """Simple index helper returning the key when indexed."""

    def __getitem__(self, key):  # pragma: no cover - trivial
        return key


index = Index()


class EmberTensor:
    """Backend-agnostic tensor wrapper."""

    def __init__(self, data, dtype=None, device=None):
        self._tensor = _convert_to_backend_tensor(data, dtype=dtype, device=device)
        self._dtype = dtype

    def to_backend_tensor(self):
        """Return the underlying backend tensor."""

        return self._tensor

    @property
    def backend(self):  # pragma: no cover - simple property
        return get_backend()

    @property
    def dtype(self):  # pragma: no cover - simple property
        return dtype(self._tensor)


_convert_to_backend_tensor = lambda *args, **kwargs: getattr(
    importlib.import_module(
        f"ember_ml.backend.{get_backend()}.tensor.ops.utility"
    ),
    "_convert_to_tensor",
)(*args, **kwargs)


def dtype(data):
    """Get the data type of a tensor."""

    return _get_backend_tensor_ops_module().dtype(data)


def array(data: Any, dtype: Any = None, device: Optional[str] = None) -> Any:
    """Create a raw backend tensor from data. Alias for :func:`convert_to_tensor`."""

    return convert_to_tensor(data, dtype=dtype, device=device)


def convert_to_tensor(
    data: Any, dtype: Any = None, device: Optional[str] = None
) -> Any:
    """Convert data to a raw backend tensor of the currently active backend."""

    if isinstance(data, EmberTensor):
        return _convert_to_backend_tensor(
            data.to_backend_tensor(), dtype=dtype, device=device
        )
    return _convert_to_backend_tensor(data, dtype=dtype, device=device)


__all__ = [
    "EmberTensor",
    "EmberDType",
    "DType",
    "dtype",
    "array",
    "convert_to_tensor",
    "zeros",
    "ones",
    "eye",
    "arange",
    "linspace",
    "zeros_like",
    "ones_like",
    "full",
    "full_like",
    "reshape",
    "transpose",
    "concatenate",
    "stack",
    "split",
    "split_tensor",
    "expand_dims",
    "squeeze",
    "tile",
    "gather",
    "scatter",
    "tensor_scatter_nd_update",
    "index_update",
    "slice_tensor",
    "slice_update",
    "shape",
    "cast",
    "copy",
    "pad",
    "item",
    "to_numpy",
    "tolist",
    "random_uniform",
    "random_normal",
    "maximum",
    "random_bernoulli",
    "random_gamma",
    "random_exponential",
    "random_poisson",
    "random_categorical",
    "random_permutation",
    "shuffle",
    "random_shuffle",
    "set_seed",
    "get_seed",
    "meshgrid",
    "nonzero",
    "index",
    "float32",
    "float64",
    "int32",
    "int64",
    "bool_",
    "int8",
    "int16",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "get_dtype",
    "to_dtype_str",
    "from_dtype_str",
]

