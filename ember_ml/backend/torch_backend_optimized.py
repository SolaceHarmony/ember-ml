"""
Optimized PyTorch backend for emberharmony.

This module provides PyTorch implementations of the emberharmony backend interface
with automatic device and precision selection for optimal performance.
"""

import torch
import numpy as np
from typing import Union, Sequence, Optional, Tuple, Any, List

# Type aliases
ArrayLike = Union[torch.Tensor, np.ndarray, float, int, list, tuple]
Shape = Union[int, Sequence[int]]
DType = Union[torch.dtype, np.dtype, str, None]

# Check for available devices
has_cuda = torch.cuda.is_available()
has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

# Determine optimal device and precision configurations
if has_cuda:
    DEFAULT_DEVICE = 'cuda'
    DEFAULT_PRECISION = torch.float16  # Use half precision on CUDA for speed
elif has_mps:
    DEFAULT_DEVICE = 'mps'
    DEFAULT_PRECISION = torch.float16  # Use half precision on MPS for speed
else:
    DEFAULT_DEVICE = 'cpu'
    DEFAULT_PRECISION = torch.float32  # Use full precision on CPU (half is slow on CPU)

# Backend information
__version__ = torch.__version__
has_gpu = has_cuda
has_mps = has_mps
default_float_type = DEFAULT_PRECISION

# Array Creation Operations
def convert_to_tensor(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Convert input to a PyTorch tensor with optimal precision and device placement.
    
    Args:
        x: Input data (array, tensor, scalar)
        dtype: Optional data type (if None, uses optimal default)
        device: Optional device to place the tensor on (if None, uses optimal default)
        
    Returns:
        PyTorch tensor representation of the input
    """
    # Determine optimal dtype if not specified
    if dtype is None:
        if device == 'cpu' or (device is None and DEFAULT_DEVICE == 'cpu'):
            # Use float32 on CPU as float16 is slow
            dtype = torch.float32
        else:
            # Use default precision (likely float16) on GPU/MPS
            dtype = DEFAULT_PRECISION
    
    # Create tensor
    if isinstance(x, torch.Tensor):
        tensor = x
    elif isinstance(x, np.ndarray):
        tensor = torch.from_numpy(x)
    else:
        tensor = torch.tensor(x, dtype=dtype)
    
    # Cast to correct dtype if needed
    if dtype is not None and tensor.dtype != dtype and not isinstance(dtype, str):
        tensor = tensor.to(dtype)
    
    # Use the specified device or the default device
    target_device = device or DEFAULT_DEVICE
    
    # Only move to device if it's different from current device
    if tensor.device.type != target_device:
        try:
            tensor = tensor.to(target_device)
        except RuntimeError:
            # Fallback to CPU if device is not available
            if target_device != 'cpu':
                print(f"Warning: Failed to move tensor to {target_device}, falling back to CPU")
                tensor = tensor.to('cpu')
    
    return tensor

def zeros(shape: Shape, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a PyTorch tensor of zeros with optimal precision and device placement.
    
    Args:
        shape: Shape of the tensor
        dtype: Optional data type (if None, uses optimal default)
        device: Optional device to place the tensor on (if None, uses optimal default)
        
    Returns:
        PyTorch tensor of zeros with the specified shape
    """
    # Determine optimal dtype and device if not specified
    if dtype is None:
        if device == 'cpu' or (device is None and DEFAULT_DEVICE == 'cpu'):
            dtype = torch.float32
        else:
            dtype = DEFAULT_PRECISION
    
    target_device = device or DEFAULT_DEVICE
    
    return torch.zeros(shape, dtype=dtype, device=target_device)

def ones(shape: Shape, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a PyTorch tensor of ones with optimal precision and device placement.
    
    Args:
        shape: Shape of the tensor
        dtype: Optional data type (if None, uses optimal default)
        device: Optional device to place the tensor on (if None, uses optimal default)
        
    Returns:
        PyTorch tensor of ones with the specified shape
    """
    # Determine optimal dtype and device if not specified
    if dtype is None:
        if device == 'cpu' or (device is None and DEFAULT_DEVICE == 'cpu'):
            dtype = torch.float32
        else:
            dtype = DEFAULT_PRECISION
    
    target_device = device or DEFAULT_DEVICE
    
    return torch.ones(shape, dtype=dtype, device=target_device)

def zeros_like(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a PyTorch tensor of zeros with the same shape as the input.
    
    Args:
        x: Input tensor
        dtype: Optional data type (if None, uses optimal default)
        device: Optional device to place the tensor on (if None, uses same as input)
        
    Returns:
        PyTorch tensor of zeros with the same shape as x
    """
    x_tensor = convert_to_tensor(x)
    
    # Determine optimal dtype if not specified
    if dtype is None:
        target_device = device or x_tensor.device.type
        if target_device == 'cpu':
            dtype = torch.float32
        else:
            dtype = DEFAULT_PRECISION
    
    return torch.zeros_like(x_tensor, dtype=dtype, device=device or x_tensor.device)

def ones_like(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a PyTorch tensor of ones with the same shape as the input.
    
    Args:
        x: Input tensor
        dtype: Optional data type (if None, uses optimal default)
        device: Optional device to place the tensor on (if None, uses same as input)
        
    Returns:
        PyTorch tensor of ones with the same shape as x
    """
    x_tensor = convert_to_tensor(x)
    
    # Determine optimal dtype if not specified
    if dtype is None:
        target_device = device or x_tensor.device.type
        if target_device == 'cpu':
            dtype = torch.float32
        else:
            dtype = DEFAULT_PRECISION
    
    return torch.ones_like(x_tensor, dtype=dtype, device=device or x_tensor.device)

def eye(n: int, m: Optional[int] = None, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a PyTorch identity matrix with optimal precision and device placement.
    
    Args:
        n: Number of rows
        m: Number of columns (default: n)
        dtype: Optional data type (if None, uses optimal default)
        device: Optional device to place the tensor on (if None, uses optimal default)
        
    Returns:
        PyTorch identity matrix of shape (n, m)
    """
    # Determine optimal dtype and device if not specified
    if dtype is None:
        if device == 'cpu' or (device is None and DEFAULT_DEVICE == 'cpu'):
            dtype = torch.float32
        else:
            dtype = DEFAULT_PRECISION
    
    target_device = device or DEFAULT_DEVICE
    
    return torch.eye(n, m, dtype=dtype, device=target_device)

# Array Manipulation Operations
def reshape(x: ArrayLike, shape: Shape) -> torch.Tensor:
    """
    Reshape a PyTorch tensor to a new shape.
    
    Args:
        x: Input tensor
        shape: New shape
        
    Returns:
        Reshaped PyTorch tensor
    """
    return convert_to_tensor(x).reshape(shape)

def transpose(x: ArrayLike, axes: Optional[Sequence[int]] = None) -> torch.Tensor:
    """
    Permute the dimensions of a PyTorch tensor.
    
    Args:
        x: Input tensor
        axes: Optional permutation of dimensions
        
    Returns:
        Transposed PyTorch tensor
    """
    x_tensor = convert_to_tensor(x)
    
    if axes is None:
        # Default transpose behavior (swap last two dimensions)
        return x_tensor.transpose(-2, -1)
    
    # Convert to PyTorch's permute format
    return x_tensor.permute(*axes)

def concatenate(arrays: Sequence[ArrayLike], axis: int = 0) -> torch.Tensor:
    """
    Concatenate PyTorch tensors along a specified axis.
    
    Args:
        arrays: Sequence of tensors
        axis: Axis along which to concatenate
        
    Returns:
        Concatenated PyTorch tensor
    """
    return torch.cat([convert_to_tensor(arr) for arr in arrays], dim=axis)

def stack(arrays: Sequence[ArrayLike], axis: int = 0) -> torch.Tensor:
    """
    Stack PyTorch tensors along a new axis.
    
    Args:
        arrays: Sequence of tensors
        axis: Axis along which to stack
        
    Returns:
        Stacked PyTorch tensor
    """
    return torch.stack([convert_to_tensor(arr) for arr in arrays], dim=axis)

def split(x: ArrayLike, num_or_size_splits: Union[int, Sequence[int]], axis: int = 0) -> List[torch.Tensor]:
    """
    Split a PyTorch tensor into sub-tensors.
    
    Args:
        x: Input tensor
        num_or_size_splits: Number of splits or sizes of each split
        axis: Axis along which to split
        
    Returns:
        List of sub-tensors
    """
    x_tensor = convert_to_tensor(x)
    
    if isinstance(num_or_size_splits, int):
        return torch.chunk(x_tensor, num_or_size_splits, dim=axis)
    else:
        return torch.split(x_tensor, num_or_size_splits, dim=axis)

def expand_dims(x: ArrayLike, axis: Union[int, Sequence[int]]) -> torch.Tensor:
    """
    Insert new axes into a PyTorch tensor's shape.
    
    Args:
        x: Input tensor
        axis: Position(s) where new axes should be inserted
        
    Returns:
        PyTorch tensor with expanded dimensions
    """
    x_tensor = convert_to_tensor(x)
    
    if isinstance(axis, (list, tuple)):
        for ax in sorted(axis):
            x_tensor = torch.unsqueeze(x_tensor, ax)
        return x_tensor
    
    return torch.unsqueeze(x_tensor, axis)

def squeeze(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None) -> torch.Tensor:
    """
    Remove single-dimensional entries from a PyTorch tensor's shape.
    
    Args:
        x: Input tensor
        axis: Position(s) where dimensions should be removed
        
    Returns:
        PyTorch tensor with squeezed dimensions
    """
    x_tensor = convert_to_tensor(x)
    
    if axis is None:
        return torch.squeeze(x_tensor)
    
    if isinstance(axis, (list, tuple)):
        for ax in sorted(axis, reverse=True):  # Squeeze from highest dim to lowest
            x_tensor = torch.squeeze(x_tensor, ax)
        return x_tensor
    
    return torch.squeeze(x_tensor, axis)

# Mathematical Operations
def add(x: ArrayLike, y: ArrayLike) -> torch.Tensor:
    """
    Add two PyTorch tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise sum
    """
    return torch.add(convert_to_tensor(x), convert_to_tensor(y))

def subtract(x: ArrayLike, y: ArrayLike) -> torch.Tensor:
    """
    Subtract two PyTorch tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise difference
    """
    return torch.subtract(convert_to_tensor(x), convert_to_tensor(y))

def multiply(x: ArrayLike, y: ArrayLike) -> torch.Tensor:
    """
    Multiply two PyTorch tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise product
    """
    return torch.multiply(convert_to_tensor(x), convert_to_tensor(y))

def divide(x: ArrayLike, y: ArrayLike) -> torch.Tensor:
    """
    Divide two PyTorch tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise quotient
    """
    return torch.divide(convert_to_tensor(x), convert_to_tensor(y))

def dot(x: ArrayLike, y: ArrayLike) -> torch.Tensor:
    """
    Compute the dot product of two PyTorch tensors.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Dot product
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    
    # Handle different dimensions
    if x_tensor.dim() == 1 and y_tensor.dim() == 1:
        return torch.dot(x_tensor, y_tensor)
    else:
        return torch.matmul(x_tensor, y_tensor)

def matmul(x: ArrayLike, y: ArrayLike) -> torch.Tensor:
    """
    Compute the matrix product of two PyTorch tensors.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Matrix product
    """
    return torch.matmul(convert_to_tensor(x), convert_to_tensor(y))

def mean(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute the mean of a PyTorch tensor along specified axes.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the mean
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Mean of the tensor
    """
    x_tensor = convert_to_tensor(x)
    
    if axis is None:
        return torch.mean(x_tensor)
    
    if isinstance(axis, (list, tuple)):
        # PyTorch doesn't support multiple axes directly, so we need to do it sequentially
        result = x_tensor
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = torch.mean(result, dim=ax, keepdim=keepdims)
        return result
    
    return torch.mean(x_tensor, dim=axis, keepdim=keepdims)

def sum(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute the sum of a PyTorch tensor along specified axes.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the sum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Sum of the tensor
    """
    x_tensor = convert_to_tensor(x)
    
    if axis is None:
        return torch.sum(x_tensor)
    
    if isinstance(axis, (list, tuple)):
        # PyTorch doesn't support multiple axes directly, so we need to do it sequentially
        result = x_tensor
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = torch.sum(result, dim=ax, keepdim=keepdims)
        return result
    
    return torch.sum(x_tensor, dim=axis, keepdim=keepdims)

def var(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute the variance of a PyTorch tensor along specified axes.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Variance of the tensor
    """
    x_tensor = convert_to_tensor(x)
    
    if axis is None:
        return torch.var(x_tensor)
    
    if isinstance(axis, (list, tuple)):
        # PyTorch doesn't support multiple axes directly, so we need to do it sequentially
        result = x_tensor
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = torch.var(result, dim=ax, keepdim=keepdims)
        return result
    
    return torch.var(x_tensor, dim=axis, keepdim=keepdims)

def exp(x: ArrayLike) -> torch.Tensor:
    """
    Compute the exponential of a PyTorch tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise exponential
    """
    return torch.exp(convert_to_tensor(x))

def log(x: ArrayLike) -> torch.Tensor:
    """
    Compute the natural logarithm of a PyTorch tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise logarithm
    """
    return torch.log(convert_to_tensor(x))

def pow(x: ArrayLike, y: ArrayLike) -> torch.Tensor:
    """
    Compute x raised to the power of y element-wise.
    
    Args:
        x: Base tensor
        y: Exponent tensor
        
    Returns:
        Element-wise power
    """
    return torch.pow(convert_to_tensor(x), convert_to_tensor(y))

def sqrt(x: ArrayLike) -> torch.Tensor:
    """
    Compute the square root of a PyTorch tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise square root
    """
    return torch.sqrt(convert_to_tensor(x))

def clip(x: ArrayLike, min_val: Union[float, ArrayLike], max_val: Union[float, ArrayLike]) -> torch.Tensor:
    """
    Clip the values of a PyTorch tensor to a specified range.
    
    Args:
        x: Input tensor
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clipped tensor
    """
    return torch.clamp(convert_to_tensor(x), min=min_val, max=max_val)

# Activation Functions
def sigmoid(x: ArrayLike) -> torch.Tensor:
    """
    Compute the sigmoid of a PyTorch tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise sigmoid
    """
    return torch.sigmoid(convert_to_tensor(x))

def tanh(x: ArrayLike) -> torch.Tensor:
    """
    Compute the hyperbolic tangent of a PyTorch tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise tanh
    """
    return torch.tanh(convert_to_tensor(x))

def relu(x: ArrayLike) -> torch.Tensor:
    """
    Compute the rectified linear unit of a PyTorch tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise ReLU
    """
    return torch.relu(convert_to_tensor(x))

def softmax(x: ArrayLike, axis: int = -1) -> torch.Tensor:
    """
    Compute the softmax of a PyTorch tensor along a specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to compute the softmax
        
    Returns:
        Softmax of the tensor
    """
    return torch.softmax(convert_to_tensor(x), dim=axis)

# Random Operations
def random_normal(shape: Shape, mean: float = 0.0, stddev: float = 1.0, 
                 dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a PyTorch tensor with random values from a normal distribution.
    
    Args:
        shape: Shape of the tensor
        mean: Mean of the normal distribution
        stddev: Standard deviation of the normal distribution
        dtype: Optional data type (if None, uses optimal default)
        device: Optional device to place the tensor on (if None, uses optimal default)
        
    Returns:
        PyTorch tensor with random normal values
    """
    # Determine optimal dtype and device if not specified
    if dtype is None:
        if device == 'cpu' or (device is None and DEFAULT_DEVICE == 'cpu'):
            dtype = torch.float32
        else:
            dtype = DEFAULT_PRECISION
    
    target_device = device or DEFAULT_DEVICE
    
    return torch.normal(mean, stddev, size=shape, dtype=dtype, device=target_device)

def random_uniform(shape: Shape, minval: float = 0.0, maxval: float = 1.0,
                  dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a PyTorch tensor with random values from a uniform distribution.
    
    Args:
        shape: Shape of the tensor
        minval: Minimum value
        maxval: Maximum value
        dtype: Optional data type (if None, uses optimal default)
        device: Optional device to place the tensor on (if None, uses optimal default)
        
    Returns:
        PyTorch tensor with random uniform values
    """
    # Determine optimal dtype and device if not specified
    if dtype is None:
        if device == 'cpu' or (device is None and DEFAULT_DEVICE == 'cpu'):
            dtype = torch.float32
        else:
            dtype = DEFAULT_PRECISION
    
    target_device = device or DEFAULT_DEVICE
    
    return torch.rand(shape, dtype=dtype, device=target_device) * (maxval - minval) + minval

def random_binomial(shape: Shape, p: float = 0.5,
                   dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a PyTorch tensor with random values from a binomial distribution.
    
    Args:
        shape: Shape of the tensor
        p: Probability of success
        dtype: Optional data type (if None, uses optimal default)
        device: Optional device to place the tensor on (if None, uses optimal default)
        
    Returns:
        PyTorch tensor with random binomial values
    """
    # Determine optimal dtype and device if not specified
    if dtype is None:
        if device == 'cpu' or (device is None and DEFAULT_DEVICE == 'cpu'):
            dtype = torch.float32
        else:
            dtype = DEFAULT_PRECISION
    
    target_device = device or DEFAULT_DEVICE
    
    return torch.bernoulli(torch.full(shape, p, dtype=dtype, device=target_device))

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Device Operations
def to_device(x: ArrayLike, device: str) -> torch.Tensor:
    """
    Move a PyTorch tensor to the specified device.
    
    Args:
        x: Input tensor
        device: Target device
        
    Returns:
        PyTorch tensor on the target device
    """
    x_tensor = convert_to_tensor(x)
    
    try:
        return x_tensor.to(device)
    except RuntimeError:
        print(f"Warning: Failed to move tensor to {device}, falling back to CPU")
        return x_tensor.to('cpu')

def get_device(x: ArrayLike) -> str:
    """
    Get the device of a PyTorch tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Device of the tensor
    """
    x_tensor = convert_to_tensor(x)
    return str(x_tensor.device)

# Utility Operations
def to_numpy(x: ArrayLike) -> np.ndarray:
    """
    Convert a tensor to a NumPy array.
    
    Args:
        x: Input tensor
        
    Returns:
        NumPy array
    """
    x_tensor = convert_to_tensor(x)
    
    # Move to CPU if on another device
    if x_tensor.device.type != 'cpu':
        x_tensor = x_tensor.cpu()
    
    # Convert to NumPy
    if x_tensor.requires_grad:
        return x_tensor.detach().numpy()
    else:
        return x_tensor.numpy()

def shape(x: ArrayLike) -> Tuple[int, ...]:
    """
    Get the shape of a PyTorch tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Shape of the tensor
    """
    return tuple(convert_to_tensor(x).shape)

def dtype(x: ArrayLike) -> torch.dtype:
    """
    Get the data type of a PyTorch tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Data type of the tensor
    """
    return convert_to_tensor(x).dtype

def ember_dtype_to_torch(dtype: Union[Any, str, None]) -> torch.dtype:
    """
    Convert an EmberDtype to a PyTorch data type.
    
    Args:
        dtype: The EmberDtype to convert
        
    Returns:
        The corresponding PyTorch data type
    """
    if dtype is None:
        return None
    
    # If it's already a PyTorch dtype, return it
    if isinstance(dtype, torch.dtype):
        return dtype
    
    # If it's an EmberDtype, use its name
    if hasattr(dtype, 'name'):
        dtype_name = dtype.name
    elif isinstance(dtype, str):
        dtype_name = dtype
    else:
        raise ValueError(f"Cannot convert {dtype} to PyTorch data type")
    
    # Map dtype names to PyTorch dtypes
    if dtype_name == 'float32':
        return torch.float32
    elif dtype_name == 'float64':
        return torch.float64
    elif dtype_name == 'int32':
        return torch.int32
    elif dtype_name == 'int64':
        return torch.int64
    elif dtype_name == 'bool' or dtype_name == 'bool_':
        return torch.bool
    elif dtype_name == 'int8':
        return torch.int8
    elif dtype_name == 'int16':
        return torch.int16
    elif dtype_name == 'uint8':
        return torch.uint8
    elif dtype_name == 'float16':
        return torch.float16
    else:
        raise ValueError(f"Unknown data type: {dtype_name}")

def torch_to_ember_dtype(dtype: Union[torch.dtype, str, None]) -> Any:
    """
    Convert a PyTorch data type to an EmberDtype.
    
    Args:
        dtype: The PyTorch data type to convert
        
    Returns:
        The corresponding EmberDtype name
    """
    if dtype is None:
        return None
    
    # Map PyTorch dtypes to EmberDtype names
    if dtype == torch.float32:
        return 'float32'
    elif dtype == torch.float64:
        return 'float64'
    elif dtype == torch.int32:
        return 'int32'
    elif dtype == torch.int64:
        return 'int64'
    elif dtype == torch.bool:
        return 'bool'
    elif dtype == torch.int8:
        return 'int8'
    elif dtype == torch.int16:
        return 'int16'
    elif dtype == torch.uint8:
        return 'uint8'
    elif dtype == torch.float16:
        return 'float16'
    elif isinstance(dtype, str):
        # If it's already a string, return it
        return dtype
    else:
        raise ValueError(f"Cannot convert {dtype} to EmberDtype")

def cast(x: ArrayLike, dtype: DType) -> torch.Tensor:
    """
    Cast a PyTorch tensor to a different data type.
    
    Args:
        x: Input tensor
        dtype: Target data type
        
    Returns:
        PyTorch tensor with the target data type
    """
    return convert_to_tensor(x).to(dtype)

def copy(x: ArrayLike) -> torch.Tensor:
    """
    Create a copy of a PyTorch tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Copy of the tensor
    """
    return convert_to_tensor(x).clone()