"""Neural network parameter initialization module.

This module provides backend-agnostic weight initialization schemes for 
neural network parameters.

Components:
    Standard Initializations:
        - glorot_uniform: Glorot/Xavier uniform initialization
        - glorot_normal: Glorot/Xavier normal initialization
        - orthogonal: Orthogonal matrix initialization
        
    Specialized Initializations:
        - BinomialInitializer: Discrete binary initialization
        - binomial: Helper function for binomial initialization

    Helper Functions:
        - get_initializer: Get an initializer function by name

All initializers maintain numerical stability and proper scaling
while preserving backend independence.
"""

from typing import Callable, Any, Optional

from ember_ml import tensor
from .binomial import BinomialInitializer, binomial
# Use relative imports for files within the same package
from .glorot import glorot_uniform, glorot_normal, orthogonal


def _resolve_dtype(
    explicit_dtype: Optional[Any], override_dtype: Optional[Any]
) -> Optional[Any]:
    """Pick the dtype to use for an initializer call."""

    return override_dtype if override_dtype is not None else explicit_dtype


def _resolve_device(
    explicit_device: Optional[str], override_device: Optional[str]
) -> Optional[str]:
    """Pick the target device for an initializer call."""

    return override_device if override_device is not None else explicit_device


def zeros(
    shape: Optional[Any] = None,
    *,
    dtype: Optional[Any] = None,
    device: Optional[str] = None,
) -> Any | Callable[[Any, Optional[Any], Optional[str]], Any]:
    """Return a zeros initializer that produces backend tensors.

    When ``shape`` is provided the tensor is created immediately. Otherwise a
    callable is returned so the initializer can be reused.
    """

    def _initializer(
        init_shape: Any,
        override_dtype: Optional[Any] = None,
        override_device: Optional[str] = None,
    ) -> Any:
        resolved_dtype = _resolve_dtype(dtype, override_dtype)
        resolved_device = _resolve_device(device, override_device)
        return tensor.zeros(init_shape, dtype=resolved_dtype, device=resolved_device)

    return _initializer(shape) if shape is not None else _initializer


def ones(
    shape: Optional[Any] = None,
    *,
    dtype: Optional[Any] = None,
    device: Optional[str] = None,
) -> Any | Callable[[Any, Optional[Any], Optional[str]], Any]:
    """Return an ones initializer matching the backend tensor output."""

    def _initializer(
        init_shape: Any,
        override_dtype: Optional[Any] = None,
        override_device: Optional[str] = None,
    ) -> Any:
        resolved_dtype = _resolve_dtype(dtype, override_dtype)
        resolved_device = _resolve_device(device, override_device)
        return tensor.ones(init_shape, dtype=resolved_dtype, device=resolved_device)

    return _initializer(shape) if shape is not None else _initializer


def constant(
    value: Any,
    shape: Optional[Any] = None,
    *,
    dtype: Optional[Any] = None,
    device: Optional[str] = None,
) -> Any | Callable[[Any, Optional[Any], Optional[str]], Any]:
    """Return an initializer that fills tensors with ``value``."""

    def _initializer(
        init_shape: Any,
        override_dtype: Optional[Any] = None,
        override_device: Optional[str] = None,
    ) -> Any:
        resolved_dtype = _resolve_dtype(dtype, override_dtype)
        resolved_device = _resolve_device(device, override_device)
        return tensor.full(
            init_shape,
            value,
            dtype=resolved_dtype,
            device=resolved_device,
        )

    return _initializer(shape) if shape is not None else _initializer


def random_uniform(
    shape: Optional[Any] = None,
    *,
    minval: float = 0.0,
    maxval: float = 1.0,
    dtype: Optional[Any] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> Any | Callable[[Any, Optional[Any], Optional[str]], Any]:
    """Return a uniform random initializer that respects backend types."""

    def _initializer(
        init_shape: Any,
        override_dtype: Optional[Any] = None,
        override_device: Optional[str] = None,
    ) -> Any:
        resolved_dtype = _resolve_dtype(dtype, override_dtype)
        resolved_device = _resolve_device(device, override_device)
        return tensor.random_uniform(
            init_shape,
            minval=minval,
            maxval=maxval,
            dtype=resolved_dtype,
            device=resolved_device,
        )
    if seed is not None:
        tensor.set_seed(seed)

    return _initializer(shape) if shape is not None else _initializer


def random_normal(
    shape: Optional[Any] = None,
    *,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: Optional[Any] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> Any | Callable[[Any, Optional[Any], Optional[str]], Any]:
    """Return a normal random initializer that outputs backend tensors."""

    def _initializer(
        init_shape: Any,
        override_dtype: Optional[Any] = None,
        override_device: Optional[str] = None,
    ) -> Any:
        resolved_dtype = _resolve_dtype(dtype, override_dtype)
        resolved_device = _resolve_device(device, override_device)
        return tensor.random_normal(
            init_shape,
            mean=mean,
            stddev=stddev,
            dtype=resolved_dtype,
            device=resolved_device,
        )
    if seed is not None:
        tensor.set_seed(seed)

    return _initializer(shape) if shape is not None else _initializer

# Dictionary mapping initializer names to functions
_INITIALIZERS = {
    'glorot_uniform': glorot_uniform,
    'glorot_normal': glorot_normal,
    'orthogonal': orthogonal,
    'zeros': zeros,
    'ones': ones,
    'constant': constant,
    'random_uniform': random_uniform,
    'random_normal': random_normal,
}

def get_initializer(name: str) -> Callable:
    """
    Get an initializer function by name.
    
    Args:
        name: Name of the initializer
        
    Returns:
        Initializer function
        
    Raises:
        ValueError: If the initializer name is not recognized
    """
    if name not in _INITIALIZERS:
        raise ValueError(f"Unknown initializer: {name}. Available initializers: {', '.join(_INITIALIZERS.keys())}")
    
    return _INITIALIZERS[name]

__all__ = [
    'glorot_uniform',
    'glorot_normal',
    'orthogonal',
    'zeros',
    'ones',
    'constant',
    'random_uniform',
    'random_normal',
    'BinomialInitializer',
    'binomial',
    'get_initializer',
]
