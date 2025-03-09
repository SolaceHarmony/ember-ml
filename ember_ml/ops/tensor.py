"""
Tensor wrapper for emberharmony.

This module provides a backend-agnostic tensor wrapper that can be used
across different backends (NumPy, PyTorch, MLX).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from ember_ml import ops
from ember_ml.backend import get_backend

class EmberTensor:
    """
    A backend-agnostic tensor wrapper.
    
    This class wraps tensors from different backends (NumPy, PyTorch, MLX)
    and provides a consistent interface for working with them.
    """
    
    def __init__(
        self,
        data: Any,
        shape: Optional[Sequence[int]] = None,
        dtype: Any = None,
        name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize an EmberTensor.
        
        Args:
            data: The tensor data or values to initialize the tensor with
            shape: Optional shape for the tensor
            dtype: Optional data type for the tensor
            name: Optional name for the tensor
            device: Optional device to place the tensor on
        """
        self._backend = get_backend()
        self._data = ops.convert_to_tensor(data, dtype=dtype, device=device)
        self._shape = None  # Will be computed on demand
        self._dtype = None  # Will be computed on demand
        self.name = name
        self._device = device
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the tensor."""
        # Get shape directly from the underlying tensor
        if hasattr(self._data, 'shape'):
            # For NumPy, PyTorch, and other tensors with shape attribute
            if isinstance(self._data.shape, tuple):
                return self._data.shape
            else:
                # Convert to tuple if it's not already
                return tuple(self._data.shape)
        elif hasattr(self._data, 'get_shape'):
            # For TensorFlow tensors
            return tuple(self._data.get_shape().as_list())
        else:
            # Fallback to ops.shape
            return ops.shape(self._data)
    
    def shape_as_list(self) -> List[int]:
        """Get the shape of the tensor as a list."""
        return list(self.shape)
    
    def shape_as_tuple(self) -> Tuple[int, ...]:
        """Get the shape of the tensor as a tuple."""
        return tuple(self.shape)
    
    def shape_at(self, index: int) -> int:
        """Get the size of the tensor at the specified dimension."""
        return self.shape[index]
    
    @property
    def device(self) -> Optional[str]:
        """Get the device of the tensor."""
        return self._device
    
    def to(self, dtype: Any = None, device: Optional[str] = None) -> 'EmberTensor':
        """
        Convert the tensor to a different dtype or device.
        
        Args:
            dtype: Target data type
            device: Target device
            
        Returns:
            Converted tensor
        """
        if dtype is None and device is None:
            return self
        
        new_data = self._data
        if dtype is not None:
            new_data = ops.cast(new_data, dtype)
        
        # Device conversion would be handled here if supported by the backend
        
        return EmberTensor(new_data, dtype=dtype, device=device)
    
    @property
    def dtype(self) -> Any:
        """Get the data type of the tensor."""
        # Get dtype directly from the underlying tensor
        if hasattr(self._data, 'dtype'):
            return self._data.dtype
        else:
            # Fallback to ops.dtype
            return ops.dtype(self._data)
    
    @property
    def data(self) -> Any:
        """Get the underlying tensor data."""
        return self._data
    
    @property
    def backend(self) -> str:
        """Get the backend used for this tensor."""
        return self._backend
    
    def numpy(self) -> Any:
        """Convert the tensor to a NumPy array."""
        return ops.to_numpy(self._data)
    
    def __repr__(self) -> str:
        """Return a string representation of the tensor."""
        return f"EmberTensor(shape={self.shape}, dtype={self.dtype}, backend={self.backend})"
    
    def __str__(self) -> str:
        """Return a string representation of the tensor."""
        return self.__repr__()
    
    # Arithmetic operations
    def __add__(self, other: Any) -> 'EmberTensor':
        """Add two tensors."""
        other_data = other._data if isinstance(other, EmberTensor) else other
        return EmberTensor(ops.add(self._data, other_data))
    
    def __sub__(self, other: Any) -> 'EmberTensor':
        """Subtract two tensors."""
        other_data = other._data if isinstance(other, EmberTensor) else other
        return EmberTensor(ops.subtract(self._data, other_data))
    
    def __mul__(self, other: Any) -> 'EmberTensor':
        """Multiply two tensors."""
        other_data = other._data if isinstance(other, EmberTensor) else other
        return EmberTensor(ops.multiply(self._data, other_data))
    
    def __truediv__(self, other: Any) -> 'EmberTensor':
        """Divide two tensors."""
        other_data = other._data if isinstance(other, EmberTensor) else other
        return EmberTensor(ops.divide(self._data, other_data))
    
    def __neg__(self) -> 'EmberTensor':
        """Negate the tensor."""
        return EmberTensor(ops.negative(self._data))
    
    def __abs__(self) -> 'EmberTensor':
        """Get the absolute value of the tensor."""
        return EmberTensor(ops.abs(self._data))
    
    # Comparison operations
    def __eq__(self, other: Any) -> 'EmberTensor':
        """Check if two tensors are equal."""
        other_data = other._data if isinstance(other, EmberTensor) else other
        return EmberTensor(ops.equal(self._data, other_data))
    
    def __ne__(self, other: Any) -> 'EmberTensor':
        """Check if two tensors are not equal."""
        other_data = other._data if isinstance(other, EmberTensor) else other
        return EmberTensor(ops.not_equal(self._data, other_data))
    
    def __lt__(self, other: Any) -> 'EmberTensor':
        """Check if the tensor is less than another tensor."""
        other_data = other._data if isinstance(other, EmberTensor) else other
        return EmberTensor(ops.less(self._data, other_data))
    
    def __le__(self, other: Any) -> 'EmberTensor':
        """Check if the tensor is less than or equal to another tensor."""
        other_data = other._data if isinstance(other, EmberTensor) else other
        return EmberTensor(ops.less_equal(self._data, other_data))
    
    def __gt__(self, other: Any) -> 'EmberTensor':
        """Check if the tensor is greater than another tensor."""
        other_data = other._data if isinstance(other, EmberTensor) else other
        return EmberTensor(ops.greater(self._data, other_data))
    
    def __ge__(self, other: Any) -> 'EmberTensor':
        """Check if the tensor is greater than or equal to another tensor."""
        other_data = other._data if isinstance(other, EmberTensor) else other
        return EmberTensor(ops.greater_equal(self._data, other_data))
    
    # Shape operations
    def reshape(self, shape: Sequence[int]) -> 'EmberTensor':
        """Reshape the tensor."""
        return EmberTensor(ops.reshape(self._data, shape))
    
    def transpose(self, axes: Optional[Sequence[int]] = None) -> 'EmberTensor':
        """Transpose the tensor."""
        return EmberTensor(ops.transpose(self._data, axes))
    
    def squeeze(self, axis: Optional[Union[int, Sequence[int]]] = None) -> 'EmberTensor':
        """Remove dimensions of size 1 from the tensor."""
        return EmberTensor(ops.squeeze(self._data, axis))
    
    def unsqueeze(self, axis: int) -> 'EmberTensor':
        """Add a dimension of size 1 to the tensor."""
        return EmberTensor(ops.expand_dims(self._data, axis))
    
    # Reduction operations
    def sum(self, axis: Optional[Union[int, Sequence[int]]] = None) -> 'EmberTensor':
        """Sum the tensor along the specified axis."""
        return EmberTensor(ops.sum(self._data, axis=axis))
    
    def mean(self, axis: Optional[Union[int, Sequence[int]]] = None) -> 'EmberTensor':
        """Compute the mean of the tensor along the specified axis."""
        return EmberTensor(ops.mean(self._data, axis=axis))
    
    def max(self, axis: Optional[Union[int, Sequence[int]]] = None) -> 'EmberTensor':
        """Compute the maximum of the tensor along the specified axis."""
        return EmberTensor(ops.max(self._data, axis=axis))
    
    def min(self, axis: Optional[Union[int, Sequence[int]]] = None) -> 'EmberTensor':
        """Compute the minimum of the tensor along the specified axis."""
        return EmberTensor(ops.min(self._data, axis=axis))
    
    # Static methods for tensor creation
    @staticmethod
    def zeros(shape: Sequence[int], dtype: Any = None, device: Optional[str] = None) -> 'EmberTensor':
        """Create a tensor of zeros."""
        return EmberTensor(ops.zeros(shape, dtype=dtype, device=device), dtype=dtype, device=device)
    
    @staticmethod
    def ones(shape: Sequence[int], dtype: Any = None, device: Optional[str] = None) -> 'EmberTensor':
        """Create a tensor of ones."""
        return EmberTensor(ops.ones(shape, dtype=dtype, device=device), dtype=dtype, device=device)
    
    @staticmethod
    def full(shape: Sequence[int], fill_value: Union[int, float], dtype: Any = None, device: Optional[str] = None) -> 'EmberTensor':
        """Create a tensor filled with a scalar value."""
        return EmberTensor(ops.full(shape, fill_value, dtype=dtype, device=device), dtype=dtype, device=device)
    
    @staticmethod
    def arange(start: int, stop: Optional[int] = None, step: int = 1, dtype: Any = None, device: Optional[str] = None) -> 'EmberTensor':
        """Create a tensor with evenly spaced values within a given interval."""
        return EmberTensor(ops.arange(start, stop, step, dtype=dtype, device=device), dtype=dtype, device=device)
    
    @staticmethod
    def linspace(start: float, stop: float, num: int, dtype: Any = None, device: Optional[str] = None) -> 'EmberTensor':
        """Create a tensor with evenly spaced values within a given interval."""
        return EmberTensor(ops.linspace(start, stop, num, dtype=dtype, device=device), dtype=dtype, device=device)
    
    @staticmethod
    def eye(n: int, m: Optional[int] = None, dtype: Any = None, device: Optional[str] = None) -> 'EmberTensor':
        """Create an identity matrix."""
        return EmberTensor(ops.eye(n, m, dtype=dtype, device=device), dtype=dtype, device=device)
    
    @staticmethod
    def random_normal(shape: Sequence[int], mean: float = 0.0, stddev: float = 1.0, dtype: Any = None, device: Optional[str] = None) -> 'EmberTensor':
        """Create a tensor with random values from a normal distribution."""
        return EmberTensor(ops.random_normal(shape, mean=mean, stddev=stddev, dtype=dtype, device=device), dtype=dtype, device=device)
    
    @staticmethod
    def random_uniform(shape: Sequence[int], minval: float = 0.0, maxval: float = 1.0, dtype: Any = None, device: Optional[str] = None) -> 'EmberTensor':
        """Create a tensor with random values from a uniform distribution."""
        return EmberTensor(ops.random_uniform(shape, minval=minval, maxval=maxval, dtype=dtype, device=device), dtype=dtype, device=device)
    
    @staticmethod
    def zeros_like(x: Any, dtype: Any = None, device: Optional[str] = None) -> 'EmberTensor':
        """Create a tensor of zeros with the same shape as the input."""
        if isinstance(x, EmberTensor):
            x = x._data
        return EmberTensor(ops.zeros_like(x, dtype=dtype, device=device), dtype=dtype, device=device)
    
    @staticmethod
    def ones_like(x: Any, dtype: Any = None, device: Optional[str] = None) -> 'EmberTensor':
        """Create a tensor of ones with the same shape as the input."""
        if isinstance(x, EmberTensor):
            x = x._data
        return EmberTensor(ops.ones_like(x, dtype=dtype, device=device), dtype=dtype, device=device)
    
    # Conversion methods
    @staticmethod
    def from_numpy(array: Any, dtype: Any = None, device: Optional[str] = None) -> 'EmberTensor':
        """Create a tensor from a NumPy array."""
        return EmberTensor(array, dtype=dtype, device=device)
    
    @staticmethod
    def from_tensor(tensor: Any, dtype: Any = None, device: Optional[str] = None) -> 'EmberTensor':
        """Create a tensor from another tensor."""
        if isinstance(tensor, EmberTensor):
            tensor = tensor._data
        return EmberTensor(tensor, dtype=dtype, device=device)