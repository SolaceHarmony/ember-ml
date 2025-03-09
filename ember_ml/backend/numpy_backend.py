"""
NumPy backend for emberharmony.

This module provides NumPy implementations of the emberharmony backend interface.
"""

import numpy as np
from typing import Union, Sequence, Optional, Tuple, Any, List

# Type aliases
ArrayLike = Union[np.ndarray, float, int, list, tuple]
Shape = Union[int, Sequence[int]]
DType = Union[np.dtype, str, None]

# Array Creation Operations
def convert_to_tensor(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
    """
    Convert input to a NumPy array.
    
    Args:
        x: Input data (array, tensor, scalar)
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array representation of the input
    
    Raises:
        TypeError: If x is a tensor from another backend
    """
    # Check if x is a tensor from another backend
    if hasattr(x, '__class__') and 'Tensor' in x.__class__.__name__ and not isinstance(x, np.ndarray):
        raise TypeError(f"Cannot convert tensor of type {type(x)} to NumPy array. "
                        f"Use the appropriate backend for this tensor type.")
    
    return np.asarray(x, dtype=dtype)

def zeros(shape: Shape, dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array of zeros.
    
    Args:
        shape: Shape of the array
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array of zeros with the specified shape
    """
    return np.zeros(shape, dtype=dtype)

def ones(shape: Shape, dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array of ones.
    
    Args:
        shape: Shape of the array
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array of ones with the specified shape
    """
    return np.ones(shape, dtype=dtype)

def zeros_like(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array of zeros with the same shape as the input.
    
    Args:
        x: Input array
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array of zeros with the same shape as x
    """
    return np.zeros_like(x, dtype=dtype)

def ones_like(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array of ones with the same shape as the input.
    
    Args:
        x: Input array
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array of ones with the same shape as x
    """
    return np.ones_like(x, dtype=dtype)

def eye(n: int, m: Optional[int] = None, dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy identity matrix.
    
    Args:
        n: Number of rows
        m: Number of columns (default: n)
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        NumPy identity matrix of shape (n, m)
    """
    return np.eye(n, m, dtype=dtype)

# Array Manipulation Operations
def reshape(x: ArrayLike, shape: Shape) -> np.ndarray:
    """
    Reshape a NumPy array to a new shape.
    
    Args:
        x: Input array
        shape: New shape
        
    Returns:
        Reshaped NumPy array
    """
    return np.reshape(x, shape)

def transpose(x: ArrayLike, axes: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    Permute the dimensions of a NumPy array.
    
    Args:
        x: Input array
        axes: Optional permutation of dimensions
        
    Returns:
        Transposed NumPy array
    """
    return np.transpose(x, axes)

def concatenate(arrays: Sequence[ArrayLike], axis: int = 0) -> np.ndarray:
    """
    Concatenate NumPy arrays along a specified axis.
    
    Args:
        arrays: Sequence of arrays
        axis: Axis along which to concatenate
        
    Returns:
        Concatenated NumPy array
    """
    return np.concatenate([convert_to_tensor(arr) for arr in arrays], axis=axis)

def stack(arrays: Sequence[ArrayLike], axis: int = 0) -> np.ndarray:
    """
    Stack NumPy arrays along a new axis.
    
    Args:
        arrays: Sequence of arrays
        axis: Axis along which to stack
        
    Returns:
        Stacked NumPy array
    """
    return np.stack([convert_to_tensor(arr) for arr in arrays], axis=axis)

def split(x: ArrayLike, num_or_size_splits: Union[int, Sequence[int]], axis: int = 0) -> List[np.ndarray]:
    """
    Split a NumPy array into sub-arrays.
    
    Args:
        x: Input array
        num_or_size_splits: Number of splits or sizes of each split
        axis: Axis along which to split
        
    Returns:
        List of sub-arrays
    """
    return np.split(x, num_or_size_splits, axis=axis)

def expand_dims(x: ArrayLike, axis: Union[int, Sequence[int]]) -> np.ndarray:
    """
    Insert new axes into a NumPy array's shape.
    
    Args:
        x: Input array
        axis: Position(s) where new axes should be inserted
        
    Returns:
        NumPy array with expanded dimensions
    """
    if isinstance(axis, (list, tuple)):
        result = convert_to_tensor(x)
        for ax in sorted(axis):
            result = np.expand_dims(result, ax)
        return result
    return np.expand_dims(x, axis)

def squeeze(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None) -> np.ndarray:
    """
    Remove single-dimensional entries from a NumPy array's shape.
    
    Args:
        x: Input array
        axis: Position(s) where dimensions should be removed
        
    Returns:
        NumPy array with squeezed dimensions
    """
    return np.squeeze(x, axis=axis)

# Mathematical Operations
def add(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Add two NumPy arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Element-wise sum
    """
    return np.add(x, y)

def subtract(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Subtract two NumPy arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Element-wise difference
    """
    return np.subtract(x, y)

def multiply(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Multiply two NumPy arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Element-wise product
    """
    return np.multiply(x, y)

def divide(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Divide two NumPy arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Element-wise quotient
    """
    return np.divide(x, y)

def dot(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Compute the dot product of two NumPy arrays.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Dot product
    """
    return np.dot(x, y)

def matmul(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Compute the matrix product of two NumPy arrays.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Matrix product
    """
    return np.matmul(x, y)

def mean(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> np.ndarray:
    """
    Compute the mean of a NumPy array along specified axes.
    
    Args:
        x: Input array
        axis: Axis or axes along which to compute the mean
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Mean of the array
    """
    return np.mean(x, axis=axis, keepdims=keepdims)

def sum(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> np.ndarray:
    """
    Compute the sum of a NumPy array along specified axes.
    
    Args:
        x: Input array
        axis: Axis or axes along which to compute the sum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Sum of the array
    """
    return np.sum(x, axis=axis, keepdims=keepdims)

def var(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> np.ndarray:
    """
    Compute the variance of a NumPy array along specified axes.
    
    Args:
        x: Input array
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Variance of the array
    """
    return np.var(x, axis=axis, keepdims=keepdims)

def exp(x: ArrayLike) -> np.ndarray:
    """
    Compute the exponential of a NumPy array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise exponential
    """
    return np.exp(x)

def log(x: ArrayLike) -> np.ndarray:
    """
    Compute the natural logarithm of a NumPy array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise logarithm
    """
    return np.log(x)

def pow(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Compute x raised to the power of y element-wise.
    
    Args:
        x: Base array
        y: Exponent array
        
    Returns:
        Element-wise power
    """
    return np.power(x, y)

def sqrt(x: ArrayLike) -> np.ndarray:
    """
    Compute the square root of a NumPy array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise square root
    """
    return np.sqrt(x)

def clip(x: ArrayLike, min_val: Union[float, ArrayLike], max_val: Union[float, ArrayLike]) -> np.ndarray:
    """
    Clip the values of a NumPy array to a specified range.
    
    Args:
        x: Input array
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clipped array
    """
    return np.clip(x, min_val, max_val)

# Activation Functions
def sigmoid(x: ArrayLike) -> np.ndarray:
    """
    Compute the sigmoid of a NumPy array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise sigmoid
    """
    x_safe = clip(x, -88.0, 88.0)  # Prevent overflow
    return 1.0 / (1.0 + np.exp(-x_safe))

def tanh(x: ArrayLike) -> np.ndarray:
    """
    Compute the hyperbolic tangent of a NumPy array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise tanh
    """
    return np.tanh(x)

def relu(x: ArrayLike) -> np.ndarray:
    """
    Compute the rectified linear unit of a NumPy array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise ReLU
    """
    return np.maximum(0, x)

def softmax(x: ArrayLike, axis: int = -1) -> np.ndarray:
    """
    Compute the softmax of a NumPy array along a specified axis.
    
    Args:
        x: Input array
        axis: Axis along which to compute the softmax
        
    Returns:
        Softmax of the array
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Random Operations
def random_normal(shape: Shape, mean: float = 0.0, stddev: float = 1.0,
                 dtype: DType = np.float32, device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array with random values from a normal distribution.
    
    Args:
        shape: Shape of the array
        mean: Mean of the normal distribution
        stddev: Standard deviation of the normal distribution
        dtype: Optional data type (default: float32)
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array with random normal values
    """
    return np.random.normal(mean, stddev, size=shape).astype(dtype)

def random_uniform(shape: Shape, minval: float = 0.0, maxval: float = 1.0,
                  dtype: DType = np.float32, device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array with random values from a uniform distribution.
    
    Args:
        shape: Shape of the array
        minval: Minimum value
        maxval: Maximum value
        dtype: Optional data type (default: float32)
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array with random uniform values
    """
    return np.random.uniform(minval, maxval, size=shape).astype(dtype)

def random_binomial(shape: Shape, p: float = 0.5,
                   dtype: DType = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array with random values from a binomial distribution.
    
    Args:
        shape: Shape of the array
        p: Probability of success
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array with random binomial values
    """
    return np.random.binomial(1, p, size=shape).astype(dtype)

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    np.random.seed(seed)

# Device Operations
def to_device(x: ArrayLike, device: str) -> np.ndarray:
    """
    Move a NumPy array to the specified device.
    
    Args:
        x: Input array
        device: Target device (ignored for NumPy backend)
        
    Returns:
        NumPy array (unchanged)
    """
    # NumPy doesn't have device concept, so just return the array
    return convert_to_tensor(x)

def get_device(x: ArrayLike) -> str:
    """
    Get the device of a NumPy array.
    
    Args:
        x: Input array
        
    Returns:
        Device of the array (always 'cpu' for NumPy backend)
    """
    return 'cpu'

# Utility Operations
def to_numpy(x: ArrayLike) -> np.ndarray:
    """
    Convert a tensor to a NumPy array.
    
    Args:
        x: Input array
        
    Returns:
        NumPy array
    """
    return convert_to_tensor(x)

def shape(x: ArrayLike) -> Tuple[int, ...]:
    """
    Get the shape of a NumPy array.
    
    Args:
        x: Input array
        
    Returns:
        Shape of the array
    """
    return convert_to_tensor(x).shape

def dtype(x: ArrayLike) -> np.dtype:
    """
    Get the data type of a NumPy array.
    
    Args:
        x: Input array
        
    Returns:
        Data type of the array
    """
    return convert_to_tensor(x).dtype

def cast(x: ArrayLike, dtype: DType) -> np.ndarray:
    """
    Cast a NumPy array to a different data type.
    
    Args:
        x: Input array
        dtype: Target data type
        
    Returns:
        NumPy array with the target data type
    """
    return convert_to_tensor(x).astype(dtype)

def copy(x: ArrayLike) -> np.ndarray:
    """
    Create a copy of a NumPy array.
    
    Args:
        x: Input array
        
    Returns:
        Copy of the array
    """
    return convert_to_tensor(x).copy()

def solve(a, b):
    """
    Solve a linear system of equations Ax = b for x using NumPy backend.
    
    Parameters
    ----------
    a : tensor
        Coefficient matrix A.
    b : tensor
        Right-hand side vector or matrix b.
    
    Returns
    -------
    tensor
        Solution to the system of equations.
    
    Notes
    -----
    Uses numpy.linalg.solve which requires a to be square and of full-rank.
    """
    import numpy as np
    # Use convert_to_tensor to ensure proper conversion
    a_array = convert_to_tensor(a)
    b_array = convert_to_tensor(b)
    return np.linalg.solve(a_array, b_array)


def power(x, y):
    """
    Element-wise power operation for NumPy backend.
    
    Parameters
    ----------
    x : tensor
        Base tensor.
    y : tensor or scalar
        Exponent tensor or scalar.
    
    Returns
    -------
    tensor
        Element-wise power of x raised to y.
    """
    import numpy as np
    # Use convert_to_tensor to ensure proper conversion
    x_array = convert_to_tensor(x)
    
    # Handle different types of y
    if isinstance(y, np.ndarray):
        y_array = y
    elif isinstance(y, (int, float)):
        y_array = y
    else:
        y_array = convert_to_tensor(y)
    
    return np.power(x_array, y_array)


    def linspace(self, start: float, stop: float, num: int, dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create a NumPy array with evenly spaced values within a given interval.
        
        Args:
            start: Start of interval (inclusive)
            stop: End of interval (inclusive)
            num: Number of values to generate
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            NumPy array with evenly spaced values
        """
        return np.linspace(start, stop, num, dtype=dtype)


    def int64(self) -> Type:
        """Get the int64 data type."""
        return np.int64


    def int8(self) -> Type:
        """Get the int8 data type."""
        return np.int8


    def random_categorical(self, logits: Any, num_samples: int,
                          dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Draw samples from a categorical distribution.
        
        Args:
            logits: 2D tensor with unnormalized log probabilities
            num_samples: Number of samples to draw
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            NumPy array with random categorical values
        """
        # Convert logits to probabilities
        logits = np.asarray(logits)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Draw samples
        samples = np.zeros((probs.shape[0], num_samples), dtype=np.int64)
        for i in range(probs.shape[0]):
            samples[i] = np.random.choice(probs.shape[1], size=num_samples, p=probs[i])
        
        if dtype is not None:
            samples = samples.astype(dtype)
        
        return samples


    def max(self, x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> np.ndarray:
        """
        Compute the maximum of a NumPy array along specified axes.
        
        Args:
            x: Input array
            axis: Axis or axes along which to compute the maximum
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Maximum of the array
        """
        return np.max(x, axis=axis, keepdims=keepdims)


    def log10(self, x: ArrayLike) -> np.ndarray:
        """
        Compute the base-10 logarithm of a NumPy array element-wise.
        
        Args:
            x: Input array
            
        Returns:
            Element-wise base-10 logarithm
        """
        return np.log10(x)


    def full(self, shape: Shape, fill_value: Union[float, int], dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create a NumPy array filled with a scalar value.
        
        Args:
            shape: Shape of the array
            fill_value: Value to fill the array with
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            NumPy array filled with the specified value
        """
        return np.full(shape, fill_value, dtype=dtype)


    def int16(self) -> Type:
        """Get the int16 data type."""
        return np.int16


    def uint32(self) -> Type:
        """Get the uint32 data type."""
        return np.uint32


    def random_permutation(self, n: int, dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Randomly permute a sequence of integers from 0 to n-1.
        
        Args:
            n: Upper bound for the range of integers to permute
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            NumPy array with a random permutation of integers from 0 to n-1
        """
        perm = np.random.permutation(n)
        if dtype is not None:
            perm = perm.astype(dtype)
        return perm

    def bool_(self) -> Type:
        """Get the boolean data type."""
        return np.bool_


    def shuffle(self, x: Any) -> np.ndarray:
        """
        Randomly shuffle a NumPy array along its first dimension.
        
        Args:
            x: Input array
            
        Returns:
            Shuffled NumPy array
        """
        x = np.asarray(x)
        indices = np.random.permutation(len(x))
        return x[indices]


    def logical_xor(self, x: Any, y: Any) -> np.ndarray:
        """
        Compute the logical XOR of two tensors element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean NumPy array with True where x XOR y
        """
        return np.logical_xor(x, y)

    def full_like(self, x: ArrayLike, fill_value: Union[float, int], dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create a NumPy array filled with a scalar value with the same shape as the input.
        
        Args:
            x: Input array
            fill_value: Value to fill the array with
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            NumPy array filled with the specified value with the same shape as x
        """
        return np.full_like(x, fill_value, dtype=dtype)


    def is_available(self, device_type: str) -> bool:
        """
        Check if a device type is available.
        
        Args:
            device_type: Device type to check
            
        Returns:
            True if the device type is available, False otherwise
        """
        return device_type == 'cpu'


    def arange(self, start: int, stop: Optional[int] = None, step: int = 1, dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create a NumPy array with evenly spaced values within a given interval.
        
        Args:
            start: Start of interval (inclusive)
            stop: End of interval (exclusive)
            step: Spacing between values
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            NumPy array with evenly spaced values
        """
        return np.arange(start, stop, step, dtype=dtype)


    def pi(self) -> np.ndarray:
        """
        Return the mathematical constant pi.
        
        Returns:
            The value of pi as a NumPy array
        """
        return np.array(np.pi)


    def sign(self, x: ArrayLike) -> np.ndarray:
        """
        Compute the sign of a NumPy array element-wise.
        
        Args:
            x: Input array
            
        Returns:
            Element-wise sign
        """
        return np.sign(x)


    def from_numpy_dtype(self, dtype: Type) -> Type:
        """
        Convert a NumPy data type to a NumPy data type.
        
        Args:
            dtype: The NumPy data type to convert
            
        Returns:
            The corresponding NumPy data type
        """
        return dtype

    def equal(self, x: Any, y: Any) -> np.ndarray:
        """
        Check if two tensors are equal element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean NumPy array with True where x == y
        """
        return np.equal(x, y)


    def sinh(self, x: ArrayLike) -> np.ndarray:
        """
        Compute the hyperbolic sine of a NumPy array element-wise.
        
        Args:
            x: Input array
            
        Returns:
            Element-wise hyperbolic sine
        """
        return np.sinh(x)


    def __init__(self):
        """Initialize NumPy solver operations."""
        self.tensor_ops = NumPyTensorOps()


    def get_default_device(self) -> str:
        """
        Get the default device for tensor operations.
        
        Returns:
            Default device
        """
        return self._default_device


    def logical_not(self, x: Any) -> np.ndarray:
        """
        Compute the logical NOT of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Boolean NumPy array with True where NOT x
        """
        return np.logical_not(x)


    def not_equal(self, x: Any, y: Any) -> np.ndarray:
        """
        Check if two tensors are not equal element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean NumPy array with True where x != y
        """
        return np.not_equal(x, y)


    def get_seed(self) -> Optional[int]:
        """
        Get the current random seed.
        
        Returns:
            Current random seed or None if not set
        """
        return self._current_seed


    def logical_or(self, x: Any, y: Any) -> np.ndarray:
        """
        Compute the logical OR of two tensors element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean NumPy array with True where x OR y
        """
        return np.logical_or(x, y)


    def int32(self) -> Type:
        """Get the int32 data type."""
        return np.int32


    def less_equal(self, x: Any, y: Any) -> np.ndarray:
        """
        Check if one tensor is less than or equal to another element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean NumPy array with True where x <= y
        """
        return np.less_equal(x, y)


    def float16(self) -> Type:
        """Get the float16 data type."""
        return np.float16


    def min(self, x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> np.ndarray:
        """
        Compute the minimum of a NumPy array along specified axes.
        
        Args:
            x: Input array
            axis: Axis or axes along which to compute the minimum
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Minimum of the array
        """
        return np.min(x, axis=axis, keepdims=keepdims)


    def set_default_device(self, device: str) -> None:
        """
        Set the default device for tensor operations.
        
        Args:
            device: Default device
        """
        if device != 'cpu':
            raise ValueError(f"NumPy backend only supports 'cpu' device, got {device}")
        
        self._default_device = device


    def synchronize(self, device: Optional[str] = None) -> None:
        """
        Synchronize the specified device.
        
        Args:
            device: Device to synchronize (default: current device)
        """
        # NumPy is synchronous, so this is a no-op
        pass


    def random_exponential(self, shape: Shape, scale: float = 1.0,
                          dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create a NumPy array with random values from an exponential distribution.
        
        Args:
            shape: Shape of the array
            scale: Scale parameter
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            NumPy array with random exponential values
        """
        return np.random.exponential(scale, size=shape).astype(dtype)


    def uint64(self) -> Type:
        """Get the uint64 data type."""
        return np.uint64


    def square(self, x: ArrayLike) -> np.ndarray:
        """
        Compute the square of a NumPy array element-wise.
        
        Args:
            x: Input array
            
        Returns:
            Element-wise square
        """
        return np.square(x)


    def sin(self, x: ArrayLike) -> np.ndarray:
        """
        Compute the sine of a NumPy array element-wise.
        
        Args:
            x: Input array
            
        Returns:
            Element-wise sine
        """
        return np.sin(x)


    def greater_equal(self, x: Any, y: Any) -> np.ndarray:
        """
        Check if one tensor is greater than or equal to another element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean NumPy array with True where x >= y
        """
        return np.greater_equal(x, y)


    def abs(self, x: ArrayLike) -> np.ndarray:
        """
        Compute the absolute value of a NumPy array element-wise.
        
        Args:
            x: Input array
            
        Returns:
            Element-wise absolute value
        """
        return np.abs(x)


    def logical_and(self, x: Any, y: Any) -> np.ndarray:
        """
        Compute the logical AND of two tensors element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean NumPy array with True where x AND y
        """
        return np.logical_and(x, y)


    def get_dtype(self, name: str) -> Type:
        """
        Get a data type by name.
        
        Args:
            name: The name of the data type
            
        Returns:
            The corresponding NumPy data type
        """
        if name == 'float32':
            return np.float32
        elif name == 'float64':
            return np.float64
        elif name == 'int32':
            return np.int32
        elif name == 'int64':
            return np.int64
        elif name == 'bool' or name == 'bool_':
            return np.bool_
        elif name == 'int8':
            return np.int8
        elif name == 'int16':
            return np.int16
        elif name == 'uint8':
            return np.uint8
        elif name == 'uint16':
            return np.uint16
        elif name == 'uint32':
            return np.uint32
        elif name == 'uint64':
            return np.uint64
        elif name == 'float16':
            return np.float16
        else:
            raise ValueError(f"Unknown data type: {name}")


    def less(self, x: Any, y: Any) -> np.ndarray:
        """
        Check if one tensor is less than another element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean NumPy array with True where x < y
        """
        return np.less(x, y)


    def random_gamma(self, shape: Shape, alpha: float = 1.0, beta: float = 1.0,
                    dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create a NumPy array with random values from a gamma distribution.
        
        Args:
            shape: Shape of the array
            alpha: Shape parameter
            beta: Scale parameter
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            NumPy array with random gamma values
        """
        return np.random.gamma(alpha, beta, size=shape).astype(dtype)


    def random_poisson(self, shape: Shape, lam: float = 1.0,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create a NumPy array with random values from a Poisson distribution.
        
        Args:
            shape: Shape of the array
            lam: Rate parameter
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            NumPy array with random Poisson values
        """
        return np.random.poisson(lam, size=shape).astype(dtype)


    def float32(self) -> Type:
        """Get the float32 data type."""
        return np.float32


    def tan(self, x: ArrayLike) -> np.ndarray:
        """
        Compute the tangent of a NumPy array element-wise.
        
        Args:
            x: Input array
            
        Returns:
            Element-wise tangent
        """
        return np.tan(x)


    def log2(self, x: ArrayLike) -> np.ndarray:
        """
        Compute the base-2 logarithm of a NumPy array element-wise.
        
        Args:
            x: Input array
            
        Returns:
            Element-wise base-2 logarithm
        """
        return np.log2(x)


    def uint8(self) -> Type:
        """Get the uint8 data type."""
        return np.uint8


    def memory_info(self, device: Optional[str] = None) -> dict:
        """
        Get memory information for the specified device.
        
        Args:
            device: Device to get memory information for (default: current device)
            
        Returns:
            Dictionary containing memory information
        """
        if device is not None and device != 'cpu':
            raise ValueError(f"NumPy backend only supports 'cpu' device, got {device}")
        
        # Get system memory information
        mem = psutil.virtual_memory()
        
        return {
            'total': mem.total,
            'available': mem.available,
            'used': mem.used,
            'percent': mem.percent
        }

    def cos(self, x: ArrayLike) -> np.ndarray:
        """
        Compute the cosine of a NumPy array element-wise.
        
        Args:
            x: Input array
            
        Returns:
            Element-wise cosine
        """
        return np.cos(x)


    def float64(self) -> Type:
        """Get the float64 data type."""
        return np.float64


    def cosh(self, x: ArrayLike) -> np.ndarray:
        """
        Compute the hyperbolic cosine of a NumPy array element-wise.
        
        Args:
            x: Input array
            
        Returns:
            Element-wise hyperbolic cosine
        """
        return np.cosh(x)


    def to_numpy_dtype(self, dtype: Any) -> Type:
        """
        Convert a data type to a NumPy data type.
        
        Args:
            dtype: The data type to convert
            
        Returns:
            The corresponding NumPy data type
        """
        if isinstance(dtype, type) and hasattr(np, dtype.__name__):
            return dtype
        elif isinstance(dtype, str):
            return self.get_dtype(dtype)
        else:
            raise ValueError(f"Cannot convert {dtype} to NumPy data type")


    def gather(self, x: ArrayLike, indices: Any, axis: int = 0) -> np.ndarray:
        """
        Gather slices from a NumPy array along an axis.
        
        Args:
            x: Input array
            indices: Indices of slices to gather
            axis: Axis along which to gather
            
        Returns:
            Gathered NumPy array
        """
        x_tensor = self.convert_to_tensor(x)
        indices_tensor = self.convert_to_tensor(indices)
        
        # Create a list of slice objects for each dimension
        slices = [slice(None)] * x_tensor.ndim
        slices[axis] = indices_tensor
        
        return x_tensor[tuple(slices)]


    def uint16(self) -> Type:
        """Get the uint16 data type."""
        return np.uint16


    def tile(self, x: ArrayLike, reps: Sequence[int]) -> np.ndarray:
        """
        Construct a NumPy array by tiling a given array.
        
        Args:
            x: Input array
            reps: Number of repetitions along each dimension
            
        Returns:
            Tiled NumPy array
        """
        return np.tile(x, reps)


    def greater(self, x: Any, y: Any) -> np.ndarray:
        """
        Check if one tensor is greater than another element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Boolean NumPy array with True where x > y
        """
        return np.greater(x, y)
