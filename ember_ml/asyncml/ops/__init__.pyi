"""
Type stub file for ember_ml.async.ops module.

This provides explicit type hints for all dynamically aliased asynchronous operations,
allowing type checkers to recognize the proper signatures of async ops functions.
"""

from typing import List, Optional, Any, Union, Tuple, Dict, Set, TypeVar, overload, Awaitable

from ember_ml.nn.tensor.types import TensorLike
type Tensor = Any
# Constants

# Backend control (These might not be async, keep as is)
def set_backend(backend: str) -> None: ...
def get_backend() -> str: ...
def auto_select_backend() -> str: ...

# Mathematical operations
async def add(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def subtract(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def multiply(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def divide(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def matmul(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def dot(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def exp(x: TensorLike) -> Awaitable[Tensor]: ...
async def log(x: TensorLike) -> Awaitable[Tensor]: ...
async def log10(x: TensorLike) -> Awaitable[Tensor]: ...
async def log2(x: TensorLike) -> Awaitable[Tensor]: ...
async def pow(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def sqrt(x: TensorLike) -> Awaitable[Tensor]: ...
async def square(x: TensorLike) -> Awaitable[Tensor]: ...
async def abs(x: TensorLike) -> Awaitable[Tensor]: ...
async def sign(x: TensorLike) -> Awaitable[Tensor]: ...
async def sin(x: TensorLike) -> Awaitable[Tensor]: ...
async def cos(x: TensorLike) -> Awaitable[Tensor]: ...
async def tan(x: TensorLike) -> Awaitable[Tensor]: ...
async def sinh(x: TensorLike) -> Awaitable[Tensor]: ...
async def cosh(x: TensorLike) -> Awaitable[Tensor]: ...
async def clip(x: TensorLike, min_val: TensorLike, max_val: TensorLike) -> Awaitable[Tensor]: ...
async def negative(x: TensorLike) -> Awaitable[Tensor]: ...
async def mod(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def floor_divide(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def floor(x: TensorLike) -> Awaitable[Tensor]: ...
async def ceil(x: TensorLike) -> Awaitable[Tensor]: ...
async def gradient(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def power(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...

# Comparison operations
async def equal(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def not_equal(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def less(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def less_equal(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def greater(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def greater_equal(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def logical_and(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def logical_or(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def logical_not(x: TensorLike) -> Awaitable[Tensor]: ...
async def logical_xor(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def allclose(x: TensorLike, y: TensorLike, rtol: float = ..., atol: float = ...) -> Awaitable[bool]: ...
async def isclose(x: TensorLike, y: TensorLike, rtol: float = ..., atol: float = ...) -> Awaitable[Tensor]: ...
async def all(x: TensorLike, axis: Optional[Union[int, Tuple[int, ...]]] = ..., keepdims: bool = ...) -> Awaitable[Tensor]: ...
async def where(condition: TensorLike, x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def isnan(x: TensorLike) -> Awaitable[Tensor]: ...

# Device operations
async def to_device(x: TensorLike, device: str) -> Awaitable[Tensor]: ...
async def get_device(x: TensorLike) -> Awaitable[str]: ...
async def get_available_devices() -> Awaitable[List[str]]: ...
async def memory_usage(device: Optional[str] = ...) -> Awaitable[Dict[str, Any]]: ...
async def memory_info(device: Optional[str] = ...) -> Awaitable[Dict[str, Any]]: ...
async def synchronize(device: Optional[str] = ...) -> Awaitable[None]: ...
async def set_default_device(device: str) -> Awaitable[None]: ...
async def get_default_device() -> Awaitable[str]: ...
async def is_available(device: str) -> Awaitable[bool]: ...

# IO operations
async def save(obj: Any, path: str) -> Awaitable[None]: ...
async def load(path: str) -> Awaitable[Any]: ...

# Loss functions
async def mse(y_true: TensorLike, y_pred: TensorLike) -> Awaitable[Tensor]: ...
async def mean_absolute_error(y_true: TensorLike, y_pred: TensorLike) -> Awaitable[Tensor]: ...
async def binary_crossentropy(y_true: TensorLike, y_pred: TensorLike) -> Awaitable[Tensor]: ...
async def categorical_crossentropy(y_true: TensorLike, y_pred: TensorLike) -> Awaitable[Tensor]: ...
async def sparse_categorical_crossentropy(y_true: TensorLike, y_pred: TensorLike) -> Awaitable[Tensor]: ...
async def huber_loss(y_true: TensorLike, y_pred: TensorLike, delta: float = ...) -> Awaitable[Tensor]: ...
async def log_cosh_loss(y_true: TensorLike, y_pred: TensorLike) -> Awaitable[Tensor]: ...

# Vector operations
async def normalize_vector(x: TensorLike, axis: int = ...) -> Awaitable[Tensor]: ...
async def compute_energy_stability(x: TensorLike, window_size: int) -> Awaitable[Tensor]: ...
async def compute_interference_strength(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def compute_phase_coherence(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def partial_interference(x: TensorLike, y: TensorLike, mask: TensorLike) -> Awaitable[Tensor]: ...
async def euclidean_distance(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def cosine_similarity(x: TensorLike, y: TensorLike) -> Awaitable[Tensor]: ...
async def exponential_decay(initial_value: float, decay_rate: float, decay_steps: int, step: int) -> Awaitable[float]: ...

# FFT operations
async def fft(x: TensorLike) -> Awaitable[Tensor]: ...
async def ifft(x: TensorLike) -> Awaitable[Tensor]: ...
async def fft2(x: TensorLike) -> Awaitable[Tensor]: ...
async def ifft2(x: TensorLike) -> Awaitable[Tensor]: ...
async def fftn(x: TensorLike) -> Awaitable[Tensor]: ...
async def ifftn(x: TensorLike) -> Awaitable[Tensor]: ...
async def rfft(x: TensorLike) -> Awaitable[Tensor]: ...
async def irfft(x: TensorLike) -> Awaitable[Tensor]: ...
async def rfft2(x: TensorLike) -> Awaitable[Tensor]: ...
async def irfft2(x: TensorLike) -> Awaitable[Tensor]: ...
async def rfftn(x: TensorLike) -> Awaitable[Tensor]: ...
async def irfftn(x: TensorLike) -> Awaitable[Tensor]: ...