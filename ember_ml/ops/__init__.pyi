"""
Type stub file for ember_ml.ops module.

This provides explicit type hints for all dynamically aliased operations,
allowing type checkers to recognize the proper signatures of ops functions.
"""

from typing import List, Optional, Any, Union, Tuple, Dict, Set, TypeVar, overload

from ember_ml.nn.tensor.types import TensorLike
type Tensor = Any
# Constants

# Backend control
def set_backend(backend: str) -> None: ...
def get_backend() -> str: ...
def auto_select_backend() -> str: ...

# Mathematical operations
def add(x: TensorLike, y: TensorLike) -> Tensor: ...
def subtract(x: TensorLike, y: TensorLike) -> Tensor: ...
def multiply(x: TensorLike, y: TensorLike) -> Tensor: ...
def divide(x: TensorLike, y: TensorLike) -> Tensor: ...
def matmul(x: TensorLike, y: TensorLike) -> Tensor: ...
def dot(x: TensorLike, y: TensorLike) -> Tensor: ...
def exp(x: TensorLike) -> Tensor: ...
def log(x: TensorLike) -> Tensor: ...
def log10(x: TensorLike) -> Tensor: ...
def log2(x: TensorLike) -> Tensor: ...
def pow(x: TensorLike, y: TensorLike) -> Tensor: ...
def sqrt(x: TensorLike) -> Tensor: ...
def square(x: TensorLike) -> Tensor: ...
def abs(x: TensorLike) -> Tensor: ...
def sign(x: TensorLike) -> Tensor: ...
def sin(x: TensorLike) -> Tensor: ...
def cos(x: TensorLike) -> Tensor: ...
def tan(x: TensorLike) -> Tensor: ...
def sinh(x: TensorLike) -> Tensor: ...
def cosh(x: TensorLike) -> Tensor: ...
def clip(x: TensorLike, min_val: TensorLike, max_val: TensorLike) -> Tensor: ...
def negative(x: TensorLike) -> Tensor: ...
def mod(x: TensorLike, y: TensorLike) -> Tensor: ...
def floor_divide(x: TensorLike, y: TensorLike) -> Tensor: ...
def floor(x: TensorLike) -> Tensor: ...
def ceil(x: TensorLike) -> Tensor: ...
def gradient(x: TensorLike, y: TensorLike) -> Tensor: ...
def power(x: TensorLike, y: TensorLike) -> Tensor: ...

# Comparison operations
def equal(x: TensorLike, y: TensorLike) -> Tensor: ...
def not_equal(x: TensorLike, y: TensorLike) -> Tensor: ...
def less(x: TensorLike, y: TensorLike) -> Tensor: ...
def less_equal(x: TensorLike, y: TensorLike) -> Tensor: ...
def greater(x: TensorLike, y: TensorLike) -> Tensor: ...
def greater_equal(x: TensorLike, y: TensorLike) -> Tensor: ...
def logical_and(x: TensorLike, y: TensorLike) -> Tensor: ...
def logical_or(x: TensorLike, y: TensorLike) -> Tensor: ...
def logical_not(x: TensorLike) -> Tensor: ...
def logical_xor(x: TensorLike, y: TensorLike) -> Tensor: ...
def allclose(x: TensorLike, y: TensorLike, rtol: float = ..., atol: float = ...) -> bool: ...
def isclose(x: TensorLike, y: TensorLike, rtol: float = ..., atol: float = ...) -> Tensor: ...
def all(x: TensorLike, axis: Optional[Union[int, Tuple[int, ...]]] = ..., keepdims: bool = ...) -> Tensor: ...
def any(x: TensorLike, axis: Optional[Union[int, Tuple[int, ...]]] = ..., keepdims: bool = ...) -> Tensor: ...
def where(condition: TensorLike, x: TensorLike, y: TensorLike) -> Tensor: ...
def isnan(x: TensorLike) -> Tensor: ...

# Device operations
def to_device(x: TensorLike, device: str) -> Tensor: ...
def get_device(x: TensorLike) -> str: ...
def get_available_devices() -> List[str]: ...
def memory_usage(device: Optional[str] = ...) -> Dict[str, Any]: ...
def memory_info(device: Optional[str] = ...) -> Dict[str, Any]: ...
def synchronize(device: Optional[str] = ...) -> None: ...
def set_default_device(device: str) -> None: ...
def get_default_device() -> str: ...
def is_available(device: str) -> bool: ...

# IO operations
def save(obj: Any, path: str) -> None: ...
def load(path: str) -> Any: ...

# Loss functions
def mse(y_true: TensorLike, y_pred: TensorLike) -> Tensor: ...
def mean_absolute_error(y_true: TensorLike, y_pred: TensorLike) -> Tensor: ...
def binary_crossentropy(y_true: TensorLike, y_pred: TensorLike) -> Tensor: ...
def categorical_crossentropy(y_true: TensorLike, y_pred: TensorLike) -> Tensor: ...
def sparse_categorical_crossentropy(y_true: TensorLike, y_pred: TensorLike) -> Tensor: ...
def huber_loss(y_true: TensorLike, y_pred: TensorLike, delta: float = ...) -> Tensor: ...
def log_cosh_loss(y_true: TensorLike, y_pred: TensorLike) -> Tensor: ...

# Vector operations
def normalize_vector(x: TensorLike, axis: int = ...) -> Tensor: ...
def compute_energy_stability(x: TensorLike, window_size: int) -> Tensor: ...
def compute_interference_strength(x: TensorLike, y: TensorLike) -> Tensor: ...
def compute_phase_coherence(x: TensorLike, y: TensorLike) -> Tensor: ...
def partial_interference(x: TensorLike, y: TensorLike, mask: TensorLike) -> Tensor: ...
def euclidean_distance(x: TensorLike, y: TensorLike) -> Tensor: ...
def cosine_similarity(x: TensorLike, y: TensorLike) -> Tensor: ...
def exponential_decay(initial_value: float, decay_rate: float, decay_steps: int, step: int) -> float: ...

# FFT operations
def fft(x: TensorLike) -> Tensor: ...
def ifft(x: TensorLike) -> Tensor: ...
def fft2(x: TensorLike) -> Tensor: ...
def ifft2(x: TensorLike) -> Tensor: ...
def fftn(x: TensorLike) -> Tensor: ...
def ifftn(x: TensorLike) -> Tensor: ...
def rfft(x: TensorLike) -> Tensor: ...
def irfft(x: TensorLike) -> Tensor: ...
def rfft2(x: TensorLike) -> Tensor: ...
def irfft2(x: TensorLike) -> Tensor: ...
def rfftn(x: TensorLike) -> Tensor: ...
def irfftn(x: TensorLike) -> Tensor: ...

# Folders
# Import submodules for type checking
from ember_ml.ops import linearalg
from ember_ml.ops import stats
from ember_ml.ops import bitwise