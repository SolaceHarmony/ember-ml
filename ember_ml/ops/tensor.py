"""
Tensor wrapper for ember_ml.

This module provides a backend-agnostic tensor wrapper that can be used
across different backends (NumPy, PyTorch, MLX).
"""

from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union
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
    
    def size(self) -> int:
        """
        Get the total number of elements in the tensor.
        
        This is equivalent to the product of all dimensions in the shape.
        
        Returns:
            int: Total number of elements in the tensor
        """
        # Calculate the product of all dimensions in the shape
        shape_dims = self.shape
        if not shape_dims:  # Handle scalar tensors (empty shape)
            return 1
        
        # Calculate product of all dimensions
        total_size = 1
        for dim in shape_dims:
            total_size *= dim
        
        return total_size
    
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
        if self._backend is None:
            return "unknown"
        return self._backend
    
    def numpy(self) -> Any:
        """Convert the tensor to a NumPy array."""
        # ops.to_numpy should return a NumPy array
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
    def __eq__(self, other: Any) -> bool:
        """Check if two tensors are equal."""
        if not isinstance(other, EmberTensor):
            return NotImplemented
        # Compare tensors by checking if they have the same shape, dtype, and data
        if self.shape != other.shape or self.dtype != other.dtype:
            return False
        # Use ops.equal and ops.all to check if all elements are equal
        equality_tensor = ops.equal(self._data, other._data)
        result = ops.item(ops.all(equality_tensor))
        # Explicitly convert to bool to satisfy the return type
        return bool(result)
    
    def __ne__(self, other: Any) -> bool:
        """Check if two tensors are not equal."""
        return not self.__eq__(other)
    
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
    
    def __getitem__(self, index: Any) -> 'EmberTensor':
        """
        Get an item or slice from the tensor.
        
        Args:
            index: Index, slice, or sequence of indices
            
        Returns:
            Tensor element(s) at the specified index/indices
        """
        # Handle different types of indexing
        if isinstance(index, tuple):
            # Multi-dimensional indexing
            result = self._data
            for idx in index:
                result = result[idx]
            return EmberTensor(result)
        else:
            # Single-dimensional indexing
            return EmberTensor(self._data[index])
    
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
    
    # Generator and iteration utilities
    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over the elements of the tensor.
        
        For multi-dimensional tensors, this flattens the tensor and iterates
        over all elements.
        
        Returns:
            Iterator over tensor elements
        """
        self.index = 0
        self.flat_size = self.size()
        return self
    
    def __next__(self) -> Any:
        """
        Get the next element in the iteration.
        
        Returns:
            Next element in the tensor
        
        Raises:
            StopIteration: When there are no more elements
        """
        if self.index < self.flat_size:
            # Reshape to a flat tensor for iteration
            flat_tensor = ops.reshape(self._data, (-1,))
            value = EmberTensor(flat_tensor[self.index])
            self.index += 1
            return value
        else:
            raise StopIteration
    
    @staticmethod
    def item(tensor: Any) -> Union[int, float]:
        """
        Get a scalar value from a tensor.
        
        This is a static method that can be used to extract a scalar value
        from a tensor, similar to tensor.item() in PyTorch.
        
        Args:
            tensor: Tensor to extract scalar from
            
        Returns:
            Scalar value
        """
        return ops.item(tensor)
    
    @staticmethod
    def range(start: Any, stop: Optional[Any] = None, step: Any = 1) -> range:
        """
        Create a range object based on tensor values.
        
        This is a convenience method that extracts scalar values from tensors
        and creates a Python range object.
        
        Args:
            start: Start value or tensor
            stop: Stop value or tensor (optional)
            step: Step value or tensor
            
        Returns:
            Python range object
        """
        # Extract scalar values and convert to integers
        start_val = int(ops.item(start)) if hasattr(start, 'shape') else int(start)
        
        if stop is None:
            return range(start_val)
        
        stop_val = int(ops.item(stop)) if hasattr(stop, 'shape') else int(stop)
        step_val = int(ops.item(step)) if hasattr(step, 'shape') else int(step)
        
        return range(start_val, stop_val, step_val)