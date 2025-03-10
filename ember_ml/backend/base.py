"""
Base interface for ember_ml backends.

This module defines the common interface that all backends must implement.
Each backend (NumPy, PyTorch, MLX) will provide implementations of these
functions with the same signatures but using their respective tensor libraries.
"""

from typing import Union, Sequence, Optional, Tuple, Any, List

# Type aliases
Shape = Union[int, Sequence[int]]
DType = Any  # Each backend will define its own dtype

# Array Creation Operations
def convert_to_tensor(x: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
    """
    Convert input to a tensor.
    
    Args:
        x: Input data (array, tensor, scalar)
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor representation of the input
    """
    raise NotImplementedError()

def zeros(shape: Shape, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
    """
    Create a tensor of zeros.
    
    Args:
        shape: Shape of the tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of zeros with the specified shape
    """
    raise NotImplementedError()

def ones(shape: Shape, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
    """
    Create a tensor of ones.
    
    Args:
        shape: Shape of the tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of ones with the specified shape
    """
    raise NotImplementedError()

def zeros_like(x: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
    """
    Create a tensor of zeros with the same shape as the input.
    
    Args:
        x: Input tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of zeros with the same shape as x
    """
    raise NotImplementedError()

def ones_like(x: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
    """
    Create a tensor of ones with the same shape as the input.
    
    Args:
        x: Input tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of ones with the same shape as x
    """
    raise NotImplementedError()

def eye(n: int, m: Optional[int] = None, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
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
    raise NotImplementedError()

# Array Manipulation Operations
def reshape(x: Any, shape: Shape) -> Any:
    """
    Reshape a tensor to a new shape.
    
    Args:
        x: Input tensor
        shape: New shape
        
    Returns:
        Reshaped tensor
    """
    raise NotImplementedError()

def transpose(x: Any, axes: Optional[Sequence[int]] = None) -> Any:
    """
    Permute the dimensions of a tensor.
    
    Args:
        x: Input tensor
        axes: Optional permutation of dimensions
        
    Returns:
        Transposed tensor
    """
    raise NotImplementedError()

def concatenate(arrays: Sequence[Any], axis: int = 0) -> Any:
    """
    Concatenate tensors along a specified axis.
    
    Args:
        arrays: Sequence of tensors
        axis: Axis along which to concatenate
        
    Returns:
        Concatenated tensor
    """
    raise NotImplementedError()

def stack(arrays: Sequence[Any], axis: int = 0) -> Any:
    """
    Stack tensors along a new axis.
    
    Args:
        arrays: Sequence of tensors
        axis: Axis along which to stack
        
    Returns:
        Stacked tensor
    """
    raise NotImplementedError()

def split(x: Any, num_or_size_splits: Union[int, Sequence[int]], axis: int = 0) -> List[Any]:
    """
    Split a tensor into sub-tensors.
    
    Args:
        x: Input tensor
        num_or_size_splits: Number of splits or sizes of each split
        axis: Axis along which to split
        
    Returns:
        List of sub-tensors
    """
    raise NotImplementedError()

def expand_dims(x: Any, axis: Union[int, Sequence[int]]) -> Any:
    """
    Insert new axes into a tensor's shape.
    
    Args:
        x: Input tensor
        axis: Position(s) where new axes should be inserted
        
    Returns:
        Tensor with expanded dimensions
    """
    raise NotImplementedError()

def squeeze(x: Any, axis: Optional[Union[int, Sequence[int]]] = None) -> Any:
    """
    Remove single-dimensional entries from a tensor's shape.
    
    Args:
        x: Input tensor
        axis: Position(s) where dimensions should be removed
        
    Returns:
        Tensor with squeezed dimensions
    """
    raise NotImplementedError()

# Mathematical Operations
def add(x: Any, y: Any) -> Any:
    """
    Add two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise sum
    """
    raise NotImplementedError()

def subtract(x: Any, y: Any) -> Any:
    """
    Subtract two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise difference
    """
    raise NotImplementedError()

def multiply(x: Any, y: Any) -> Any:
    """
    Multiply two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise product
    """
    raise NotImplementedError()

def divide(x: Any, y: Any) -> Any:
    """
    Divide two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise quotient
    """
    raise NotImplementedError()

def dot(x: Any, y: Any) -> Any:
    """
    Compute the dot product of two tensors.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Dot product
    """
    raise NotImplementedError()

def matmul(x: Any, y: Any) -> Any:
    """
    Compute the matrix product of two tensors.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Matrix product
    """
    raise NotImplementedError()

def mean(x: Any, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Any:
    """
    Compute the mean of a tensor along specified axes.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the mean
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Mean of the tensor
    """
    raise NotImplementedError()

def sum(x: Any, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Any:
    """
    Compute the sum of a tensor along specified axes.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the sum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Sum of the tensor
    """
    raise NotImplementedError()

def exp(x: Any) -> Any:
    """
    Compute the exponential of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise exponential
    """
    raise NotImplementedError()

def log(x: Any) -> Any:
    """
    Compute the natural logarithm of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise logarithm
    """
    raise NotImplementedError()

def pow(x: Any, y: Any) -> Any:
    """
    Compute x raised to the power of y element-wise.
    
    Args:
        x: Base tensor
        y: Exponent tensor
        
    Returns:
        Element-wise power
    """
    raise NotImplementedError()

def sqrt(x: Any) -> Any:
    """
    Compute the square root of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise square root
    """
    raise NotImplementedError()

def clip(x: Any, min_val: Union[float, Any], max_val: Union[float, Any]) -> Any:
    """
    Clip the values of a tensor to a specified range.
    
    Args:
        x: Input tensor
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clipped tensor
    """
    raise NotImplementedError()

# Activation Functions
def sigmoid(x: Any) -> Any:
    """
    Compute the sigmoid of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise sigmoid
    """
    raise NotImplementedError()

def tanh(x: Any) -> Any:
    """
    Compute the hyperbolic tangent of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise tanh
    """
    raise NotImplementedError()

def relu(x: Any) -> Any:
    """
    Compute the rectified linear unit of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise ReLU
    """
    raise NotImplementedError()

def softmax(x: Any, axis: int = -1) -> Any:
    """
    Compute the softmax of a tensor along a specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to compute the softmax
        
    Returns:
        Softmax of the tensor
    """
    raise NotImplementedError()

# Random Operations
def random_normal(shape: Shape, mean: float = 0.0, stddev: float = 1.0, 
                 dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
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
    raise NotImplementedError()

def random_uniform(shape: Shape, minval: float = 0.0, maxval: float = 1.0,
                  dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
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
    raise NotImplementedError()

def random_binomial(shape: Shape, p: float = 0.5,
                   dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
    """
    Create a tensor with random values from a binomial distribution.
    
    Args:
        shape: Shape of the tensor
        p: Probability of success
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor with random binomial values
    """
    raise NotImplementedError()

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    raise NotImplementedError()

# Device Operations
def to_device(x: Any, device: str) -> Any:
    """
    Move a tensor to the specified device.
    
    Args:
        x: Input tensor
        device: Target device
        
    Returns:
        Tensor on the target device
    """
    raise NotImplementedError()

def get_device(x: Any) -> str:
    """
    Get the device of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Device of the tensor
    """
    raise NotImplementedError()

# Utility Operations
def to_numpy(x: Any) -> Any:
    """
    Convert a tensor to a NumPy array.
    
    Args:
        x: Input tensor
        
    Returns:
        NumPy array
    """
    raise NotImplementedError()

def shape(x: Any) -> Tuple[int, ...]:
    """
    Get the shape of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Shape of the tensor
    """
    raise NotImplementedError()

def dtype(x: Any) -> DType:
    """
    Get the data type of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Data type of the tensor
    """
    raise NotImplementedError()

def cast(x: Any, dtype: DType) -> Any:
    """
    Cast a tensor to a different data type.
    
    Args:
        x: Input tensor
        dtype: Target data type
        
    Returns:
        Tensor with the target data type
    """
    raise NotImplementedError()

def copy(x: Any) -> Any:
    """
    Create a copy of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Copy of the tensor
    """
    raise NotImplementedError()

def placeholder(shape: Shape, name: Optional[str] = None, dtype: Optional[DType] = None) -> Any:
    """
    Create a placeholder tensor.
    
    Args:
        shape: Shape of the placeholder
        name: Optional name for the placeholder
        dtype: Optional data type for the placeholder
        
    Returns:
        Placeholder tensor
    """
    raise NotImplementedError()

def linspace(start: float, stop: float, num: int, dtype: Optional[DType] = None) -> Any:
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
    raise NotImplementedError()

def index_update(x: Any, indices: Any, values: Any) -> Any:
    """
    Update the values of a tensor at the specified indices.
    
    Args:
        x: Input tensor
        indices: Indices to update
        values: Values to update with
        
    Returns:
        Updated tensor
    """
    raise NotImplementedError()

def index(indices: Any) -> Any:
    """
    Create an index object for use with index_update.
    
    Args:
        indices: Indices to use
        
    Returns:
        Index object
    """
    raise NotImplementedError()

def save(filepath: str, obj: Any, allow_pickle: bool = True) -> None:
    """
    Save a tensor or dictionary of tensors to a file.
    
    Args:
        filepath: Path to save the object to
        obj: Tensor or dictionary of tensors to save
        allow_pickle: Whether to allow saving objects that can't be saved directly
        
    Returns:
        None
    """
    raise NotImplementedError()

def load(filepath: str, allow_pickle: bool = True) -> Any:
    """
    Load a tensor or dictionary of tensors from a file.
    
    Args:
        filepath: Path to load the object from
        allow_pickle: Whether to allow loading objects that can't be loaded directly
        
    Returns:
        Loaded tensor or dictionary of tensors
    """
    raise NotImplementedError()