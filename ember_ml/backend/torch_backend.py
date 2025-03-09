"""
PyTorch backend for EmberHarmony.

This module provides PyTorch implementations of the backend operations
required by EmberHarmony.
"""

import torch
from typing import Optional, Union, Tuple, List, Any, Sequence, Dict

# Type aliases
ArrayLike = Union[torch.Tensor, list, tuple, float, int]
Shape = Union[int, Sequence[int]]
DType = Union[torch.dtype, str, None]

# Backend information
__version__ = torch.__version__
has_gpu = torch.cuda.is_available()
has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
default_float_type = torch.float32

# Determine the best available device
if has_gpu:
    DEFAULT_DEVICE = 'cuda'
elif has_mps:
    DEFAULT_DEVICE = 'mps'
else:
    DEFAULT_DEVICE = 'cpu'

# Array Creation Operations
def convert_to_tensor(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Convert input to a PyTorch tensor.
    
    Args:
        x: Input data (array, tensor, scalar)
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        PyTorch tensor representation of the input
    
    Raises:
        TypeError: If x is a tensor from another backend
    """
    # Check if x is a tensor from another backend
    if hasattr(x, '__class__') and 'Tensor' in x.__class__.__name__ and not isinstance(x, torch.Tensor):
        raise TypeError(f"Cannot convert tensor of type {type(x)} to PyTorch tensor. "
                        f"Use the appropriate backend for this tensor type.")
    
    # Create tensor
    if isinstance(x, torch.Tensor):
        tensor = x
    else:
        tensor = torch.tensor(x, dtype=dtype)
    
    # Move to device if specified
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor

def zeros(shape: Shape, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of zeros.
    
    Args:
        shape: Shape of the tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of zeros with the specified shape
    """
    return torch.zeros(shape, dtype=dtype, device=device)

def ones(shape: Shape, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of ones.
    
    Args:
        shape: Shape of the tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of ones with the specified shape
    """
    return torch.ones(shape, dtype=dtype, device=device)

def zeros_like(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of zeros with the same shape as the input.
    
    Args:
        x: Input tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of zeros with the same shape as x
    """
    x_tensor = convert_to_tensor(x)
    return torch.zeros_like(x_tensor, dtype=dtype, device=device)

def ones_like(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of ones with the same shape as the input.
    
    Args:
        x: Input tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of ones with the same shape as x
    """
    x_tensor = convert_to_tensor(x)
    return torch.ones_like(x_tensor, dtype=dtype, device=device)

def eye(n: int, m: Optional[int] = None, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create an identity matrix.
    
    Args:
        n: Number of rows
        m: Number of columns (default: n)
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Identity matrix of shape (n, m)
    """
    # Handle the case where m is None
    if m is None:
        return torch.eye(n, dtype=dtype, device=device)
    else:
        return torch.eye(n, m=m, dtype=dtype, device=device)

# Array Manipulation Operations
def reshape(x: ArrayLike, shape: Shape) -> torch.Tensor:
    """
    Reshape a tensor to a new shape.
    
    Args:
        x: Input tensor
        shape: New shape
        
    Returns:
        Reshaped tensor
    """
    return convert_to_tensor(x).reshape(shape)

def transpose(x: ArrayLike, axes: Optional[Sequence[int]] = None) -> torch.Tensor:
    """
    Permute the dimensions of a tensor.
    
    Args:
        x: Input tensor
        axes: Optional permutation of dimensions
        
    Returns:
        Transposed tensor
    """
    x_tensor = convert_to_tensor(x)
    
    if axes is None:
        # Default transpose behavior (swap last two dimensions)
        return x_tensor.transpose(-2, -1)
    
    # Convert to PyTorch's permute format
    return x_tensor.permute(*axes)

def expand_dims(x: ArrayLike, axis: Union[int, Sequence[int]]) -> torch.Tensor:
    """
    Insert new axes into a tensor's shape.
    
    Args:
        x: Input tensor
        axis: Position(s) where new axes should be inserted
        
    Returns:
        Tensor with expanded dimensions
    """
    x_tensor = convert_to_tensor(x)
    
    if isinstance(axis, (list, tuple)):
        # Handle multiple axes
        result = x_tensor
        # Sort axes in ascending order to avoid dimension shifting
        for ax in sorted(axis):
            result = torch.unsqueeze(result, dim=ax)
        return result
    
    # Handle single axis
    return torch.unsqueeze(x_tensor, dim=axis)

def concatenate(arrays: Sequence[ArrayLike], axis: int = 0) -> torch.Tensor:
    """
    Concatenate tensors along a specified axis.
    
    Args:
        arrays: Sequence of tensors
        axis: Axis along which to concatenate
        
    Returns:
        Concatenated tensor
    """
    return torch.cat([convert_to_tensor(arr) for arr in arrays], dim=axis)

def stack(arrays: Sequence[ArrayLike], axis: int = 0) -> torch.Tensor:
    """
    Stack tensors along a new axis.
    
    Args:
        arrays: Sequence of tensors
        axis: Axis along which to stack
        
    Returns:
        Stacked tensor
    """
    return torch.stack([convert_to_tensor(arr) for arr in arrays], dim=axis)

# Mathematical Operations
def add(x: ArrayLike, y: ArrayLike) -> torch.Tensor:
    """
    Add two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise sum
    """
    return torch.add(convert_to_tensor(x), convert_to_tensor(y))

def subtract(x: ArrayLike, y: ArrayLike) -> torch.Tensor:
    """
    Subtract two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise difference
    """
    return torch.subtract(convert_to_tensor(x), convert_to_tensor(y))

def multiply(x: ArrayLike, y: ArrayLike) -> torch.Tensor:
    """
    Multiply two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise product
    """
    return torch.multiply(convert_to_tensor(x), convert_to_tensor(y))

def divide(x: ArrayLike, y: ArrayLike) -> torch.Tensor:
    """
    Divide two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise quotient
    """
    return torch.divide(convert_to_tensor(x), convert_to_tensor(y))

def matmul(x: ArrayLike, y: ArrayLike) -> torch.Tensor:
    """
    Compute the matrix product of two tensors.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Matrix product
    """
    return torch.matmul(convert_to_tensor(x), convert_to_tensor(y))

def mean(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute the mean of a tensor along specified axes.
    
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

def var(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute the variance of a tensor along specified axes.
    
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

def sum(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute the sum of a tensor along specified axes.
    
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

def exp(x: ArrayLike) -> torch.Tensor:
    """
    Compute the exponential of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise exponential
    """
    return torch.exp(convert_to_tensor(x))

def log(x: ArrayLike) -> torch.Tensor:
    """
    Compute the natural logarithm of a tensor element-wise.
    
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
    Compute the square root of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise square root
    """
    return torch.sqrt(convert_to_tensor(x))

# Activation Functions
def sigmoid(x: ArrayLike) -> torch.Tensor:
    """
    Compute the sigmoid of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise sigmoid
    """
    return torch.sigmoid(convert_to_tensor(x))

def tanh(x: ArrayLike) -> torch.Tensor:
    """
    Compute the hyperbolic tangent of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise tanh
    """
    return torch.tanh(convert_to_tensor(x))

def relu(x: ArrayLike) -> torch.Tensor:
    """
    Compute the rectified linear unit of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise ReLU
    """
    return torch.relu(convert_to_tensor(x))

def softmax(x: ArrayLike, axis: int = -1) -> torch.Tensor:
    """
    Compute the softmax of a tensor along a specified axis.
    
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
    Create a tensor with random values from a normal distribution.
    
    Args:
        shape: Shape of the tensor
        mean: Mean of the normal distribution
        stddev: Standard deviation of the normal distribution
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor with random normal values
    """
    return torch.normal(mean, stddev, size=shape, dtype=dtype, device=device)

def random_uniform(shape: Shape, minval: float = 0.0, maxval: float = 1.0,
                   dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor with random values from a uniform distribution.
    
    Args:
        shape: Shape of the tensor
        minval: Minimum value
        maxval: Maximum value
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor with random uniform values
    """
    # Generate random values between 0 and 1
    rand_tensor = torch.rand(shape, dtype=dtype, device=device)
    
    # Calculate the range using torch.subtract instead of direct subtraction
    maxval_tensor = convert_to_tensor(maxval)
    minval_tensor = convert_to_tensor(minval)
    range_tensor = torch.subtract(maxval_tensor, minval_tensor)
    
    # Scale the random values to the desired range
    scaled_tensor = torch.multiply(rand_tensor, range_tensor)
    
    # Shift the values to start at minval
    min_tensor = convert_to_tensor(minval)
    result_tensor = torch.add(scaled_tensor, min_tensor)
    
    return result_tensor

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
    Move a tensor to the specified device.
    
    Args:
        x: Input tensor
        device: Target device
        
    Returns:
        Tensor on the target device
    """
    x_tensor = convert_to_tensor(x)
    return x_tensor.to(device)

def get_device(x: ArrayLike) -> str:
    """
    Get the device of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Device of the tensor
    """
    x_tensor = convert_to_tensor(x)
    return str(x_tensor.device)

def get_available_devices() -> List[str]:
    """
    Get a list of available devices.
    
    Returns:
        List of available devices
    """
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append('mps')
    return devices

_default_device = DEFAULT_DEVICE

def set_default_device(device: str) -> None:
    """
    Set the default device for PyTorch operations.
    
    Args:
        device: Default device
    """
    global _default_device
    _default_device = device
    
    # Set the default device for PyTorch
    if device.startswith('cuda'):
        if torch.cuda.is_available():
            device_idx = 0
            if ':' in device:
                # Use convert_to_tensor and cast instead of int()
                device_idx_str = device.split(':')[1]
                device_idx_tensor = convert_to_tensor(device_idx_str)
                device_idx = cast(device_idx_tensor, torch.int32).item()
            torch.cuda.set_device(device_idx)

def get_default_device() -> str:
    """
    Get the default device for PyTorch operations.
    
    Returns:
        Default device
    """
    return _default_device

def is_available(device: str) -> bool:
    """
    Check if the specified device is available.
    
    Args:
        device: Device to check
        
    Returns:
        True if the device is available, False otherwise
    """
    if device == 'cpu':
        return True
    elif device.startswith('cuda'):
        return torch.cuda.is_available()
    elif device == 'mps':
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    return False

def memory_usage(device: Optional[str] = None) -> Dict[str, int]:
    """
    Get memory usage information for the specified device.
    
    Args:
        device: Target device
        
    Returns:
        Dictionary with memory usage information
    """
    if device is None:
        device = _default_device
        
    if device.startswith('cuda'):
        if torch.cuda.is_available():
            device_idx = 0
            if ':' in device:
                # Use convert_to_tensor and cast instead of int()
                device_idx_str = device.split(':')[1]
                device_idx_tensor = convert_to_tensor(device_idx_str)
                device_idx = cast(device_idx_tensor, torch.int32).item()
            
            # Get memory information
            allocated = torch.cuda.memory_allocated(device_idx)
            reserved = torch.cuda.memory_reserved(device_idx)
            
            # Get total memory
            total = torch.cuda.get_device_properties(device_idx).total_memory
            
            # Calculate free memory using torch.subtract instead of direct subtraction
            free = torch.subtract(total, reserved)
            
            return {
                'allocated': allocated,
                'reserved': reserved,
                'free': free,
                'total': total
            }
    
    # For CPU or other devices, return zeros
    return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0}

def memory_info(device: Optional[str] = None) -> Dict[str, int]:
    """
    Get memory information for the specified device.
    
    Args:
        device: Target device
        
    Returns:
        Dictionary with memory information
    """
    return memory_usage(device)

def synchronize(device: Optional[str] = None) -> None:
    """
    Synchronize the specified device.
    
    Args:
        device: Target device
    """
    if device is None:
        device = _default_device
        
    if device.startswith('cuda'):
        if torch.cuda.is_available():
            device_idx = 0
            if ':' in device:
                # Use convert_to_tensor and cast instead of int()
                device_idx_str = device.split(':')[1]
                device_idx_tensor = convert_to_tensor(device_idx_str)
                device_idx = cast(device_idx_tensor, torch.int32).item()
            torch.cuda.synchronize(device_idx)

# Utility Operations
def to_numpy(x: ArrayLike) -> Any:
    """
    Convert a tensor to a NumPy array.
    
    This function is an exception to the general backend purity rules as it's specifically
    designed to convert tensors to NumPy arrays for use with plotting libraries and other
    external tools that require NumPy arrays.
    
    Args:
        x: Input tensor
        
    Returns:
        NumPy array
    """
    x_tensor = convert_to_tensor(x)
    
    # Move to CPU if on another device
    if x_tensor.device.type != 'cpu':
        x_tensor = x_tensor.cpu()
    
    # EMBERLINT: IGNORE - Direct NumPy usage is allowed in this function as an exception
    # Convert to NumPy using PyTorch's native method
    if x_tensor.requires_grad:
        # First detach the tensor to remove the gradient tracking
        detached_tensor = x_tensor.detach()
        # EMBERLINT: IGNORE - Direct tensor.numpy() usage is allowed here
        import numpy as np
        return np.array(detached_tensor.cpu().detach())
    else:
        # EMBERLINT: IGNORE - Direct tensor.numpy() usage is allowed here
        import numpy as np
        return np.array(x_tensor.cpu().detach())

def shape(x: ArrayLike) -> Tuple[int, ...]:
    """
    Get the shape of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Shape of the tensor
    """
    return tuple(convert_to_tensor(x).shape)

def dtype(x: ArrayLike) -> torch.dtype:
    """
    Get the data type of a tensor.
    
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
    Cast a tensor to a different data type.
    
    Args:
        x: Input tensor
        dtype: Target data type
        
    Returns:
        Tensor with the target data type
    """
    return convert_to_tensor(x).to(dtype)

def copy(x: ArrayLike) -> torch.Tensor:
    """
    Create a copy of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Copy of the tensor
    """
    return convert_to_tensor(x).clone()

def placeholder(shape: Shape, name: Optional[str] = None, dtype: DType = None) -> torch.Tensor:
    """
    Create a placeholder tensor.
    
    Args:
        shape: Shape of the placeholder
        name: Optional name for the placeholder
        dtype: Optional data type for the placeholder
        
    Returns:
        Placeholder tensor
    """
    # In PyTorch, we can just create a tensor of zeros with requires_grad=True
    return torch.zeros(shape, dtype=dtype, requires_grad=True)

def linspace(start: float, stop: float, num: int, dtype: DType = None) -> torch.Tensor:
    """
    Create a tensor with evenly spaced values within a given interval.
    
    Args:
        start: Start of the interval
        stop: End of the interval
        num: Number of values to generate
        dtype: Optional data type
        
    Returns:
        Tensor with evenly spaced values
    """
    return torch.linspace(start, stop, num, dtype=dtype)

def index_update(x: ArrayLike, indices: Any, values: ArrayLike) -> torch.Tensor:
    """
    Update the values of a tensor at the specified indices.
    
    Args:
        x: Input tensor
        indices: Indices to update
        values: Values to update with
        
    Returns:
        Updated tensor
    """
    x_tensor = convert_to_tensor(x).clone()
    values_tensor = convert_to_tensor(values)
    
    if isinstance(indices, tuple) and len(indices) > 0 and isinstance(indices[0], slice):
        # Handle slice indices
        x_tensor[indices] = values_tensor
    else:
        # Handle regular indices
        x_tensor[indices] = values_tensor
    
    return x_tensor

def index(indices: Any) -> Any:
    """
    Create an index object for use with index_update.
    
    Args:
        indices: Indices to use
        
    Returns:
        Index object
    """
    # In PyTorch, we can just return the indices directly
    return indices

def solve(a, b):
    """
    Solve a linear system of equations Ax = b for x using PyTorch backend.
    
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
    Uses torch.linalg.solve which requires a to be square and of full-rank.
    """
    import torch
    
    # Convert inputs to PyTorch tensors with the correct dtype
    if not isinstance(a, torch.Tensor):
        a_tensor = torch.tensor(a, dtype=torch.float32)
    else:
        a_tensor = a.to(torch.float32)
        
    if not isinstance(b, torch.Tensor):
        b_tensor = torch.tensor(b, dtype=torch.float32)
    else:
        b_tensor = b.to(torch.float32)
    
    # Ensure a is a square matrix
    if len(a_tensor.shape) < 2:
        # If a is a vector or scalar, reshape it to a square matrix
        # Use convert_to_tensor and cast instead of int()
        # Replace ** with torch.pow
        numel = a_tensor.numel()
        numel_tensor = convert_to_tensor(numel)
        power_value = convert_to_tensor(0.5)
        numel_sqrt = torch.pow(numel_tensor, power_value)
        numel_sqrt_tensor = convert_to_tensor(numel_sqrt)
        n = cast(numel_sqrt_tensor, torch.int32).item()
        a_tensor = a_tensor.reshape(n, n)
    
    # Ensure b has the right shape for torch.linalg.solve
    if len(b_tensor.shape) < 2:
        # If b is a vector, reshape it to a column vector
        b_tensor = b_tensor.reshape(-1, 1)
    
    # Make sure a and b are compatible
    if a_tensor.shape[0] != b_tensor.shape[0]:
        raise ValueError(f"Incompatible shapes: a {a_tensor.shape}, b {b_tensor.shape}")
    
    # Solve the system
    return torch.linalg.solve(a_tensor, b_tensor)


def power(x, y):
    """
    Element-wise power operation for PyTorch backend.
    
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
    import torch
    # Use convert_to_tensor to ensure proper conversion
    x_tensor = convert_to_tensor(x)
    
    # Handle different types of y
    if isinstance(y, torch.Tensor):
        # Ensure y is not 0-dimensional
        if y.dim() == 0:
            y_tensor = y.reshape(1)
        else:
            y_tensor = y
    elif isinstance(y, (int, float)):
        y_tensor = torch.tensor(y, dtype=x_tensor.dtype)
    else:
        y_tensor = convert_to_tensor(y)
        # Ensure y is not 0-dimensional
        if y_tensor.dim() == 0:
            y_tensor = y_tensor.reshape(1)
    
    return torch.pow(x_tensor, y_tensor)

def tan(x: ArrayLike) -> torch.Tensor:
    """
    Compute the tangent of a PyTorch tensor.

    Args:
        x: Input tensor
        
    Returns:
        Tangent of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.tan(x_tensor)

def random_categorical(logits: Any, num_samples: int, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Draw samples from a categorical distribution.
    
    Args:
        logits: 2D tensor with unnormalized log probabilities
        num_samples: Number of samples to draw
        dtype: Optional data type
        device: Optional device
        
    Returns:
        Tensor with random categorical values
    """
    # Convert to PyTorch tensor if needed
    logits_tensor = convert_to_tensor(logits)
    
    # Convert to probabilities
    probs = torch.softmax(logits_tensor, dim=-1)
    
    # Move to the specified device
    if device is not None:
        probs = probs.to(device=device)
        
    # Sample from the categorical distribution
    samples = torch.multinomial(probs, num_samples, replacement=True)
    
    # Convert to the specified data type
    if dtype is not None:
        samples = samples.to(dtype=dtype)
        
    return samples


def full_like(x: ArrayLike, fill_value: Union[float, int], dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a PyTorch tensor filled with a scalar value with the same shape as the input.
    
    Args:
        x: Input tensor
        fill_value: Value to fill the tensor with
        dtype: Optional data type
        device: Optional device (defaults to DEFAULT_DEVICE if None)
        
    Returns:
        PyTorch tensor filled with the specified value with the same shape as x
    """
    if device is None:
        device = DEFAULT_DEVICE
    x_tensor = convert_to_tensor(x)
    # Since we don't have _convert_dtype, use ember_dtype_to_torch instead
    torch_dtype = ember_dtype_to_torch(dtype) if dtype is not None else None
    return torch.full_like(x_tensor, fill_value, dtype=torch_dtype, device=device)


def tile(x: ArrayLike, reps: Sequence[int]) -> torch.Tensor:
    """
    Construct a PyTorch tensor by tiling a given tensor.
    
    Args:
        x: Input tensor
        reps: Number of repetitions along each dimension
        
    Returns:
        Tiled PyTorch tensor
    """
    x_tensor = convert_to_tensor(x)
    return x_tensor.repeat(*reps)



def dot(x: ArrayLike, y: ArrayLike) -> torch.Tensor:
    """
    Compute the dot product of two PyTorch tensors.
    
    Args:
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Dot product of x and y
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return torch.matmul(x_tensor.flatten(), y_tensor.flatten())


def cosh(x: ArrayLike) -> torch.Tensor:
    """
    Compute the hyperbolic cosine of a PyTorch tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Hyperbolic cosine of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.cosh(x_tensor)




def random_binomial(shape: Shape, p: float = 0.5, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Generate random values from a binomial distribution.
    
    Args:
        shape: Shape of the output tensor
        p: Probability of success
        dtype: Optional data type
        device: Optional device
        
    Returns:
        PyTorch tensor with random values from a binomial distribution
    """
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
        
    # Generate random values
    result = torch.bernoulli(torch.full(shape, p))
    
    # Convert to the specified data type
    if dtype is not None:
        result = result.to(dtype=dtype)
        
    # Move to the specified device
    if device is not None:
        result = result.to(device=device)
        
    return result


def log2(x: ArrayLike) -> torch.Tensor:
    """
    Compute the base-2 logarithm of a PyTorch tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Base-2 logarithm of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.log2(x_tensor)




def random_exponential(shape: Shape, scale: float = 1.0, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Generate random values from an exponential distribution.
    
    Args:
        shape: Shape of the output tensor
        scale: Scale parameter
        dtype: Optional data type
        device: Optional device
        
    Returns:
        PyTorch tensor with random values from an exponential distribution
    """
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
        
    # Generate uniform random values
    u = torch.rand(shape, dtype=dtype, device=device)
    
    # Transform to exponential distribution
    # Exponential distribution: f(x) = (1/scale) * exp(-x/scale)
    # Can be sampled by taking -scale * ln(U) where U is uniform(0,1)
    # Avoid log(0) by using 1-u instead of u
    # Replace direct operators with torch functions
    one_tensor = convert_to_tensor(1.0)
    one_minus_u = torch.subtract(one_tensor, u)
    log_result = torch.log(one_minus_u)
    scale_tensor = convert_to_tensor(scale)
    scaled_result = torch.multiply(scale_tensor, log_result)
    result = torch.negative(scaled_result)
    
    return result




def set_default_device(device: str) -> None:
    """
    Set the default device for PyTorch operations.
    
    Args:
        device: Default device
    """
    global _default_device
    _default_device = device
    
    # Set the default device for PyTorch
    if device.startswith('cuda'):
        if torch.cuda.is_available():
            device_idx = 0
            if ':' in device:
                # Use convert_to_tensor and cast instead of int()
                device_idx_str = device.split(':')[1]
                device_idx_tensor = convert_to_tensor(device_idx_str)
                device_idx = cast(device_idx_tensor, torch.int32).item()
            torch.cuda.set_device(device_idx)


def pi() -> torch.Tensor:
    """
    Return the mathematical constant pi.
    
    Returns:
        The value of pi as a PyTorch tensor
    """
    return torch.tensor(torch.pi)


def sign(x: ArrayLike) -> torch.Tensor:
    """
    Compute the sign of a PyTorch tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Sign of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.sign(x_tensor)


def max(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute the maximum of a PyTorch tensor along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the maximum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Maximum of x along the specified axis
    """
    x_tensor = convert_to_tensor(x)
    if axis is None:
        return torch.max(x_tensor)
    return torch.max(x_tensor, dim=axis, keepdim=keepdims)[0]


def arange(start: int, stop: Optional[int] = None, step: int = 1, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a PyTorch tensor with evenly spaced values within a given interval.
    
    Args:
        start: Start of interval (inclusive)
        stop: End of interval (exclusive)
        step: Spacing between values
        dtype: Optional data type
        device: Optional device (defaults to DEFAULT_DEVICE if None)
        
    Returns:
        PyTorch tensor with evenly spaced values
    """
    if device is None:
        device = DEFAULT_DEVICE
    if stop is None:
        # If only one argument is provided, it's the stop value
        return torch.arange(0, start, step, dtype=ember_dtype_to_torch(dtype), device=device)
    return torch.arange(start, stop, step, dtype=ember_dtype_to_torch(dtype), device=device)


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
        # Calculate split size using torch operations
        # Use torch.div with rounding_mode='trunc' instead of // operator
        split_size = torch.div(torch.tensor(x_tensor.shape[axis]),
                              torch.tensor(num_or_size_splits),
                              rounding_mode='trunc').item()
        return torch.split(x_tensor, split_size, dim=axis)
    return torch.split(x_tensor, num_or_size_splits, dim=axis)



def set_random_seed(seed: int) -> None:
    """
    Alias for set_seed for backward compatibility.
    
    Args:
        seed: Random seed
    """
    set_seed(seed)


def sin(x: ArrayLike) -> torch.Tensor:
    """
    Compute the sine of a PyTorch tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Sine of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.sin(x_tensor)




def gather(x: ArrayLike, indices: Any, axis: int = 0) -> torch.Tensor:
    """
    Gather slices from a PyTorch tensor along an axis.
    
    Args:
        x: Input tensor
        indices: Indices of slices to gather
        axis: Axis along which to gather
        
    Returns:
        Gathered PyTorch tensor
    """
    x_tensor = convert_to_tensor(x)
    indices_tensor = convert_to_tensor(indices)
    return torch.gather(x_tensor, axis, indices_tensor)


def random_permutation(n: int, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Randomly permute a sequence of integers from 0 to n-1.
    
    Args:
        n: Upper bound for the range of integers to permute
        dtype: Optional data type
        device: Optional device
        
    Returns:
        PyTorch tensor with a random permutation of integers from 0 to n-1
    """
    # Generate random permutation using PyTorch's randperm function
    perm = torch.randperm(n, device=device)
    
    # Convert to the specified data type
    if dtype is not None:
        perm = perm.to(dtype=dtype)
        
    return perm

def abs(x: ArrayLike) -> torch.Tensor:
    """
    Compute the absolute value of a PyTorch tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Absolute value of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.abs(x_tensor)


def cos(x: ArrayLike) -> torch.Tensor:
    """
    Compute the cosine of a PyTorch tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Cosine of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.cos(x_tensor)


def random_poisson(shape: Shape, lam: float = 1.0, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Generate random values from a Poisson distribution.
    
    Args:
        shape: Shape of the output tensor
        lam: Rate parameter
        dtype: Optional data type
        device: Optional device
        
    Returns:
        PyTorch tensor with random values from a Poisson distribution
    """
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
        
    # Create a tensor filled with the rate parameter
    rate_tensor = torch.full(shape, lam, dtype=dtype, device=device)
    
    # Sample from the Poisson distribution
    result = torch.poisson(rate_tensor)
    
    return result


def square(x: ArrayLike) -> torch.Tensor:
    """
    Compute the square of a PyTorch tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Square of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.square(x_tensor)


def sinh(x: ArrayLike) -> torch.Tensor:
    """
    Compute the hyperbolic sine of a PyTorch tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Hyperbolic sine of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.sinh(x_tensor)


def log10(x: ArrayLike) -> torch.Tensor:
    """
    Compute the base-10 logarithm of a PyTorch tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Base-10 logarithm of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.log10(x_tensor)


def full(shape: Shape, fill_value: Union[float, int], dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a PyTorch tensor filled with a scalar value.
    
    Args:
        shape: Shape of the tensor
        fill_value: Value to fill the tensor with
        dtype: Optional data type
        device: Optional device (defaults to DEFAULT_DEVICE if None)
        
    Returns:
        PyTorch tensor filled with the specified value
    """
    if device is None:
        device = DEFAULT_DEVICE
    return torch.full(shape, fill_value, dtype=ember_dtype_to_torch(dtype), device=device)


def min(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute the minimum of a PyTorch tensor along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the minimum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Minimum of x along the specified axis
    """
    x_tensor = convert_to_tensor(x)
    if axis is None:
        return torch.min(x_tensor)
    return torch.min(x_tensor, dim=axis, keepdim=keepdims)[0]


def is_available(device: str) -> bool:
    """
    Check if the specified device is available.
    
    Args:
        device: Device to check
        
    Returns:
        True if the device is available, False otherwise
    """
    if device == 'cpu':
        return True
    elif device.startswith('cuda'):
        return torch.cuda.is_available()
    elif device == 'mps':
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    return False


def shuffle(x: Any) -> torch.Tensor:
    """
    Randomly shuffle a PyTorch tensor along the first dimension.
    
    Args:
        x: Input tensor
        
    Returns:
        Shuffled PyTorch tensor
    """
    x_tensor = convert_to_tensor(x)
    
    # Get the shape of the tensor
    shape = x_tensor.shape
    
    # If the tensor is empty or has only one element, return it as is
    if shape[0] <= 1:
        return x_tensor
    
    # Generate random indices
    indices = torch.randperm(shape[0], device=x_tensor.device)
    
    # Gather along the first dimension
    return x_tensor[indices]


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
        result = x_tensor
        for ax in sorted(axis, reverse=True):
            result = torch.squeeze(result, ax)
        return result
    return torch.squeeze(x_tensor, axis)


def synchronize(device: Optional[str] = None) -> None:
    """
    Synchronize the specified device.
    
    Args:
        device: Target device
    """
    if device is None:
        device = _default_device
        
    if device.startswith('cuda'):
        if torch.cuda.is_available():
            device_idx = 0
            if ':' in device:
                # Use convert_to_tensor and cast instead of int()
                device_idx_str = device.split(':')[1]
                device_idx_tensor = convert_to_tensor(device_idx_str)
                device_idx = cast(device_idx_tensor, torch.int32).item()
            torch.cuda.synchronize(device_idx)

def clip(x: ArrayLike, min_val: Optional[float] = None, max_val: Optional[float] = None) -> torch.Tensor:
    """
    Clip the values of a PyTorch tensor to the specified range.
    
    Args:
        x: Input tensor
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clipped tensor
    """
    x_tensor = convert_to_tensor(x)
    return torch.clamp(x_tensor, min=min_val, max=max_val)


def random_gamma(shape: Shape, alpha: float = 1.0, beta: float = 1.0, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Generate random values from a gamma distribution.
    
    Args:
        shape: Shape of the output tensor
        alpha: Shape parameter
        beta: Scale parameter
        dtype: Optional data type
        device: Optional device
        
    Returns:
        PyTorch tensor with random values from a gamma distribution
    """
    # Convert shape to tuple if it's an integer
    if isinstance(shape, int):
        shape = (shape,)
        
    # PyTorch's gamma distribution uses rate parameter (1/beta)
    # Replace direct division with torch.divide
    one_tensor = convert_to_tensor(1.0)
    beta_tensor = convert_to_tensor(beta)
    rate = torch.divide(one_tensor, beta_tensor)
    
    # Create a gamma distribution
    gamma_dist = torch.distributions.gamma.Gamma(alpha, rate)
    
    # Sample from the distribution
    result = gamma_dist.sample(shape)
    
    # Convert to the specified data type
    if dtype is not None:
        result = result.to(dtype=dtype)
        
    # Move to the specified device
    if device is not None:
        result = result.to(device=device)
        
    return result


# Class implementations for ops module
class TorchTensorOps:
    """PyTorch implementation of tensor operations."""
    
    def zeros(self, shape, dtype=None, device=None):
        """Create a tensor of zeros."""
        return zeros(shape, dtype=dtype, device=device)
    
    def ones(self, shape, dtype=None, device=None):
        """Create a tensor of ones."""
        return ones(shape, dtype=dtype, device=device)
    
    def zeros_like(self, x, dtype=None, device=None):
        """Create a tensor of zeros with the same shape as the input."""
        return zeros_like(x, dtype=dtype, device=device)
    
    def ones_like(self, x, dtype=None, device=None):
        """Create a tensor of ones with the same shape as the input."""
        return ones_like(x, dtype=dtype, device=device)
    
    def eye(self, n, m=None, dtype=None, device=None):
        """Create an identity matrix."""
        return eye(n, m=m, dtype=dtype, device=device)
    
    def arange(self, start, stop=None, step=1, dtype=None, device=None):
        """Create a tensor with evenly spaced values within a given interval."""
        return arange(start, stop=stop, step=step, dtype=dtype, device=device)
    
    def linspace(self, start, stop, num, dtype=None, device=None):
        """Create a tensor with evenly spaced values within a given interval."""
        return linspace(start, stop, num, dtype=dtype)
    
    def full(self, shape, fill_value, dtype=None, device=None):
        """Create a tensor filled with a scalar value."""
        return full(shape, fill_value, dtype=dtype, device=device)
    
    def full_like(self, x, fill_value, dtype=None, device=None):
        """Create a tensor filled with a scalar value with the same shape as the input."""
        return full_like(x, fill_value, dtype=dtype, device=device)
    
    def reshape(self, x, shape):
        """Reshape a tensor to a new shape."""
        return reshape(x, shape)
    
    def transpose(self, x, axes=None):
        """Permute the dimensions of a tensor."""
        return transpose(x, axes=axes)
    
    def concatenate(self, tensors, axis=0):
        """Concatenate tensors along a specified axis."""
        return concatenate(tensors, axis=axis)
    
    def stack(self, tensors, axis=0):
        """Stack tensors along a new axis."""
        return stack(tensors, axis=axis)
    
    def split(self, x, num_or_size_splits, axis=0):
        """Split a tensor into sub-tensors."""
        return split(x, num_or_size_splits, axis=axis)
    
    def expand_dims(self, x, axis):
        """Insert new axes into a tensor's shape."""
        return expand_dims(x, axis)
    
    def squeeze(self, x, axis=None):
        """Remove single-dimensional entries from a tensor's shape."""
        return squeeze(x, axis=axis)
    
    def tile(self, x, reps):
        """Construct a tensor by tiling a given tensor."""
        return tile(x, reps)
    
    def gather(self, x, indices, axis=0):
        """Gather slices from a tensor along an axis."""
        return gather(x, indices, axis=axis)
    
    def convert_to_tensor(self, x, dtype=None, device=None):
        """Convert input to a tensor."""
        return convert_to_tensor(x, dtype=dtype, device=device)
    
    def shape(self, x):
        """Get the shape of a tensor."""
        return shape(x)
    
    def dtype(self, x):
        """Get the data type of a tensor."""
        return dtype(x)
    
    def cast(self, x, dtype):
        """Cast a tensor to a different data type."""
        return cast(x, dtype)
    
    def copy(self, x):
        """Create a copy of a tensor."""
        return copy(x)
    
    def var(self, x, axis=None, keepdims=False):
        """Compute the variance of a tensor along specified axes."""
        return var(x, axis=axis, keepdims=keepdims)
    
    def to_numpy(self, x):
        """Convert a tensor to a NumPy array."""
        return to_numpy(x)


class TorchMathOps:
    """PyTorch implementation of math operations."""
    
    @property
    def pi(self):
        """Return the mathematical constant pi."""
        return pi()
    
    def add(self, x, y):
        """Add two tensors element-wise."""
        return add(x, y)
    
    def subtract(self, x, y):
        """Subtract two tensors element-wise."""
        return subtract(x, y)
    
    def multiply(self, x, y):
        """Multiply two tensors element-wise."""
        return multiply(x, y)
    
    def divide(self, x, y):
        """Divide two tensors element-wise."""
        return divide(x, y)
    
    def dot(self, x, y):
        """Compute the dot product of two tensors."""
        return dot(x, y)
    
    def matmul(self, x, y):
        """Compute the matrix product of two tensors."""
        return matmul(x, y)
    
    def mean(self, x, axis=None, keepdims=False):
        """Compute the mean of a tensor along specified axes."""
        return mean(x, axis=axis, keepdims=keepdims)
    
    def sum(self, x, axis=None, keepdims=False):
        """Compute the sum of a tensor along specified axes."""
        return sum(x, axis=axis, keepdims=keepdims)
    
    def max(self, x, axis=None, keepdims=False):
        """Compute the maximum of a tensor along specified axes."""
        return max(x, axis=axis, keepdims=keepdims)
    
    def min(self, x, axis=None, keepdims=False):
        """Compute the minimum of a tensor along specified axes."""
        return min(x, axis=axis, keepdims=keepdims)
    
    def exp(self, x):
        """Compute the exponential of a tensor."""
        return exp(x)
    
    def log(self, x):
        """Compute the natural logarithm of a tensor."""
        return log(x)
    
    def log10(self, x):
        """Compute the base-10 logarithm of a tensor."""
        return log10(x)
    
    def log2(self, x):
        """Compute the base-2 logarithm of a tensor."""
        return log2(x)
    
    def pow(self, x, y):
        """Compute x raised to the power of y."""
        return pow(x, y)
    
    def sqrt(self, x):
        """Compute the square root of a tensor."""
        return sqrt(x)
    
    def square(self, x):
        """Compute the square of a tensor."""
        return square(x)
    
    def abs(self, x):
        """Compute the absolute value of a tensor."""
        return abs(x)
    
    def sign(self, x):
        """Compute the sign of a tensor."""
        return sign(x)
    
    def sin(self, x):
        """Compute the sine of a tensor."""
        return sin(x)
    
    def cos(self, x):
        """Compute the cosine of a tensor."""
        return cos(x)
    
    def tan(self, x):
        """Compute the tangent of a tensor."""
        return tan(x)
    
    def sinh(self, x):
        """Compute the hyperbolic sine of a tensor."""
        return sinh(x)
    
    def cosh(self, x):
        """Compute the hyperbolic cosine of a tensor."""
        return cosh(x)
    
    def tanh(self, x):
        """Compute the hyperbolic tangent of a tensor."""
        return tanh(x)
    
    def sigmoid(self, x):
        """Compute the sigmoid of a tensor."""
        return sigmoid(x)
    
    def relu(self, x):
        """Compute the rectified linear unit of a tensor."""
        return relu(x)
    
    def softmax(self, x, axis=-1):
        """Compute the softmax of a tensor."""
        return softmax(x, axis=axis)
    
    def clip(self, x, min_val=None, max_val=None):
        """Clip the values of a tensor."""
        return clip(x, min_val=min_val, max_val=max_val)
    
    def var(self, x, axis=None, keepdims=False):
        """Compute the variance of a tensor."""
        return var(x, axis=axis, keepdims=keepdims)


class TorchRandomOps:
    """PyTorch implementation of random operations."""
    
    def random_normal(self, shape, mean=0.0, stddev=1.0, dtype=None, device=None):
        """Generate random values from a normal distribution."""
        return random_normal(shape, mean=mean, stddev=stddev, dtype=dtype, device=device)
    
    def random_uniform(self, shape, minval=0.0, maxval=1.0, dtype=None, device=None):
        """Generate random values from a uniform distribution."""
        return random_uniform(shape, minval=minval, maxval=maxval, dtype=dtype, device=device)
    
    def random_binomial(self, shape, p=0.5, dtype=None, device=None):
        """Generate random values from a binomial distribution."""
        return random_binomial(shape, p=p, dtype=dtype, device=device)
    
    def random_permutation(self, x):
        """Generate a random permutation."""
        return random_permutation(x, dtype=None, device=None)
    
    def set_random_seed(self, seed):
        """Set the random seed for reproducibility."""
        return set_random_seed(seed)


class TorchComparisonOps:
    """PyTorch implementation of comparison operations."""
    
    def equal(self, x, y):
        """Check if two tensors are equal element-wise."""
        return torch.eq(convert_to_tensor(x), convert_to_tensor(y))
    
    def not_equal(self, x, y):
        """Check if two tensors are not equal element-wise."""
        return torch.ne(convert_to_tensor(x), convert_to_tensor(y))
    
    def less(self, x, y):
        """Check if one tensor is less than another element-wise."""
        return torch.lt(convert_to_tensor(x), convert_to_tensor(y))
    
    def less_equal(self, x, y):
        """Check if one tensor is less than or equal to another element-wise."""
        return torch.le(convert_to_tensor(x), convert_to_tensor(y))
    
    def greater(self, x, y):
        """Check if one tensor is greater than another element-wise."""
        return torch.gt(convert_to_tensor(x), convert_to_tensor(y))
    
    def greater_equal(self, x, y):
        """Check if one tensor is greater than or equal to another element-wise."""
        return torch.ge(convert_to_tensor(x), convert_to_tensor(y))
    
    def logical_and(self, x, y):
        """Compute the logical AND of two tensors element-wise."""
        return torch.logical_and(convert_to_tensor(x), convert_to_tensor(y))
    
    def logical_or(self, x, y):
        """Compute the logical OR of two tensors element-wise."""
        return torch.logical_or(convert_to_tensor(x), convert_to_tensor(y))
    
    def logical_not(self, x):
        """Compute the logical NOT of a tensor element-wise."""
        return torch.logical_not(convert_to_tensor(x))
    
    def logical_xor(self, x, y):
        """Compute the logical XOR of two tensors element-wise."""
        return torch.logical_xor(convert_to_tensor(x), convert_to_tensor(y))


class TorchDeviceOps:
    """PyTorch implementation of device operations."""
    
    def to_device(self, x, device):
        """Move a tensor to the specified device."""
        return to_device(x, device)
    
    def get_device(self, x):
        """Get the device of a tensor."""
        return get_device(x)
    
    def get_available_devices(self):
        """Get a list of available devices."""
        return get_available_devices()
    
    def memory_usage(self, device=None):
        """Get memory usage information for the specified device."""
        return memory_usage(device)


class TorchDTypeOps:
    """PyTorch implementation of data type operations."""
    
    def get_dtype(self, name):
        """Get a data type by name."""
        return ember_dtype_to_torch(name)
    
    def to_numpy_dtype(self, dtype):
        """Convert a PyTorch data type to a NumPy data type."""
        return torch_to_ember_dtype(dtype)
    
    def from_numpy_dtype(self, dtype):
        """Convert a NumPy data type to a PyTorch data type."""
        return ember_dtype_to_torch(dtype)


class TorchSolverOps:
    """PyTorch implementation of solver operations."""
    
    def solve(self, a, b):
        """Solve a linear system of equations."""
        return solve(a, b)
