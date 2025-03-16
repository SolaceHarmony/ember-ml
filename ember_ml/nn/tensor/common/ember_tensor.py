"""
Backend-agnostic EmberTensor implementation.

This module provides a common implementation of the tensor interface that works
with any backend (NumPy, PyTorch, MLX) using the backend abstraction layer.
"""

from typing import Any, Optional, List, Union, Tuple, Sequence, Callable

from ember_ml.nn.tensor.common.dtypes import EmberDType as DType
from ember_ml.nn.tensor.interfaces import TensorInterface
from ember_ml.nn.tensor.common import (
    _convert_to_backend_tensor, to_numpy, item, shape, dtype, zeros, ones, zeros_like, ones_like,
    eye, arange, linspace, full, full_like, reshape, transpose, concatenate, stack, split,
    expand_dims, squeeze, tile, gather, tensor_scatter_nd_update, slice, slice_update,
    cast, copy, var, pad, sort, argsort, maximum, random_normal, random_uniform,
    random_bernoulli, random_gamma, random_exponential, random_poisson,
    random_categorical, random_permutation, shuffle, set_seed, get_seed
)


class EmberTensor(TensorInterface):
    """
    A backend-agnostic tensor implementation using the backend abstraction layer.
    
    This implementation delegates all operations to the current backend through
    the backend abstraction layer, ensuring backend purity and compatibility.
    """
    
    def __repr__(self) -> str:
        """Return a string representation of the tensor."""
        return f"EmberTensor({to_numpy(self._tensor)})"
 
    def __init__(
        self,
        data: Optional[Any] = None,
        *,
        dtype: Optional[Union[DType, str, Callable[[], Any]]] = None,
        device: Optional[str] = None,
        requires_grad: bool = False
    ) -> None:
        """
        Initialize an EmberTensor.

        Args:
            data: Input data to create tensor from
            dtype: Optional dtype for the tensor (can be a DType, string, or callable)
            device: Optional device to place the tensor on
            requires_grad: Whether the tensor requires gradients
        """
        # Handle different dtype formats
        processed_dtype = None
        if dtype is not None:
            if callable(dtype):
                # If dtype is a callable (like the lambda functions in dtypes.py),
                # call it to get the actual dtype
                processed_dtype = dtype()
            elif isinstance(dtype, str):
                # If dtype is a string, use it directly
                processed_dtype = dtype
            else:
                # Otherwise, use it as is (assuming it's a DType or compatible)
                processed_dtype = dtype
        
        self._tensor = _convert_to_backend_tensor(data, dtype=processed_dtype)
        # Use the get_device function from the backend module to get the default device
        from ember_ml.backend import get_device, get_backend
        self._device = device if device is not None else get_device()
        self._requires_grad = requires_grad
        
        # Store the dtype explicitly
        from ember_ml.nn.tensor.common import dtype as get_dtype
        self._dtype = processed_dtype if processed_dtype is not None else get_dtype(self._tensor)
        
        # Store the backend information
        self._backend = get_backend()

    def to_backend_tensor(self) -> Any:
        """Get the underlying backend tensor."""
        return self._tensor

    def __array__(self) -> Any:
        """NumPy array interface."""
        return to_numpy(self._tensor)
    
    def __getitem__(self, key) -> 'EmberTensor':
        """
        Get values at specified indices.
        
        Args:
            key: Index or slice
            
        Returns:
            Tensor with values at specified indices
        """
        # Use slice_update to implement indexing
        result = slice_update(self._tensor, key, None)
        return EmberTensor(result, device=self.device, requires_grad=self._requires_grad)

    def item(self) -> Union[int, float, bool]:
        """Get the value of a scalar tensor."""
        return item(self._tensor)

    def tolist(self) -> List[Any]:
        """Convert tensor to a (nested) list."""
        return to_numpy(self._tensor).tolist()

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the tensor."""
        return shape(self._tensor)

    @property
    def dtype(self) -> DType:
        """Get the dtype of the tensor."""
        # Use the stored dtype if available
        if hasattr(self, '_dtype') and self._dtype is not None:
            # If it's already a DType, return it directly
            if isinstance(self._dtype, DType):
                return self._dtype
            # If it's a string or other format, convert to DType
            return DType(str(self._dtype).split('.')[-1])
            
        # Otherwise, get it from the backend tensor
        backend_dtype = dtype(self._tensor)
        # Convert backend-specific dtype to EmberDType
        if isinstance(backend_dtype, DType):
            return backend_dtype
        # If it's a backend-specific dtype, extract the name and create an EmberDType
        dtype_name = str(backend_dtype).split('.')[-1]
        return DType(dtype_name)

    @property
    def device(self) -> str:
        """Get the device the tensor is on."""
        device_str = str(self._device)
        # Handle MLX DeviceType.gpu format
        if device_str.startswith("DeviceType."):
            return device_str.split(".")[-1]
        return device_str
        
    @property
    def backend(self) -> str:
        """Get the backend the tensor is using."""
        # Use the stored backend if available
        if hasattr(self, '_backend') and self._backend is not None:
            return str(self._backend)
            
        # Otherwise, infer from device
        device_str = str(self._device)
        # Handle MLX DeviceType.gpu format
        if device_str.startswith("DeviceType."):
            return device_str.split(".")[-1]
        return device_str

    @property
    def requires_grad(self) -> bool:
        """Get whether the tensor requires gradients."""
        return self._requires_grad

    def detach(self) -> 'EmberTensor':
        """Create a new tensor detached from the computation graph."""
        # For now, create a new tensor with requires_grad=False
        # Don't pass dtype to avoid calling the dtype property
        return EmberTensor(
            self._tensor,
            device=self.device,
            requires_grad=False
        )

    def numpy(self) -> Any:
        """Convert tensor to NumPy array."""
        return to_numpy(self._tensor)
    
    def zeros(self, shape: Union[int, Sequence[int]], dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
        """
        Create a tensor of zeros.
        
        Args:
            shape: Shape of the tensor
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor of zeros with the specified shape
        """
        tensor = zeros(shape, dtype=dtype)
        return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
    
    def ones(self, shape: Union[int, Sequence[int]], dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
        """
        Create a tensor of ones.
        
        Args:
            shape: Shape of the tensor
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor of ones with the specified shape
        """
        tensor = ones(shape, dtype=dtype)
        return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
    
    def zeros_like(self, x: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
        """
        Create a tensor of zeros with the same shape as the input.
        
        Args:
            x: Input tensor
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor of zeros with the same shape as x
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        tensor = zeros_like(x, dtype=dtype)
        return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
    
    def ones_like(self, x: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
        """
        Create a tensor of ones with the same shape as the input.
        
        Args:
            x: Input tensor
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor of ones with the same shape as x
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        tensor = ones_like(x, dtype=dtype)
        return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
    
    def eye(self, n: int, m: Optional[int] = None, dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
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
        tensor = eye(n, m, dtype=dtype)
        return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
    
    def arange(self, start: int, stop: Optional[int] = None, step: int = 1, dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
        """
        Create a tensor with evenly spaced values within a given interval.
        
        Args:
            start: Start of interval (inclusive)
            stop: End of interval (exclusive)
            step: Spacing between values
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor with evenly spaced values
        """
        tensor = arange(start, stop, step, dtype=dtype)
        return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
    
    def linspace(self, start: float, stop: float, num: int, dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
        """
        Create a tensor with evenly spaced values within a given interval.
        
        Args:
            start: Start of interval (inclusive)
            stop: End of interval (inclusive)
            num: Number of values to generate
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor with evenly spaced values
        """
        tensor = linspace(start, stop, num, dtype=dtype)
        return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
    
    def full(self, shape: Union[int, Sequence[int]], fill_value: Union[float, int], dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
        """
        Create a tensor filled with a scalar value.
        
        Args:
            shape: Shape of the tensor
            fill_value: Value to fill the tensor with
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor filled with the specified value
        """
        tensor = full(shape, fill_value, dtype=dtype)
        return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
    
    def full_like(self, x: Any, fill_value: Union[float, int], dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
        """
        Create a tensor filled with a scalar value with the same shape as the input.
        
        Args:
            x: Input tensor
            fill_value: Value to fill the tensor with
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor filled with the specified value with the same shape as x
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        tensor = full_like(x, fill_value, dtype=dtype)
        return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
    
    def reshape(self, x: Any, shape: Union[int, Sequence[int]]) -> 'EmberTensor':
        """
        Reshape a tensor to a new shape.
        
        Args:
            x: Input tensor
            shape: New shape
            
        Returns:
            Reshaped tensor
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        tensor = reshape(x, shape)
        # Don't pass dtype to avoid calling the dtype property
        return EmberTensor(tensor, device=self.device, requires_grad=self._requires_grad)
    
    def transpose(self, x: Any, axes: Optional[Sequence[int]] = None) -> 'EmberTensor':
        """
        Permute the dimensions of a tensor.
        
        Args:
            x: Input tensor
            axes: Optional permutation of dimensions
            
        Returns:
            Transposed tensor
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        tensor = transpose(x, axes)
        return EmberTensor(tensor, dtype=self.dtype, device=self.device, requires_grad=self._requires_grad)
    
    def concatenate(self, tensors: Sequence[Any], axis: int = 0) -> 'EmberTensor':
        """
        Concatenate tensors along a specified axis.
        
        Args:
            tensors: Sequence of tensors
            axis: Axis along which to concatenate
            
        Returns:
            Concatenated tensor
        """
        backend_tensors = [t.to_backend_tensor() if isinstance(t, EmberTensor) else t for t in tensors]
        tensor = concatenate(backend_tensors, axis)
        return EmberTensor(tensor, dtype=self.dtype, device=self.device, requires_grad=self._requires_grad)
    
    def stack(self, tensors: Sequence[Any], axis: int = 0) -> 'EmberTensor':
        """
        Stack tensors along a new axis.
        
        Args:
            tensors: Sequence of tensors
            axis: Axis along which to stack
            
        Returns:
            Stacked tensor
        """
        backend_tensors = [t.to_backend_tensor() if isinstance(t, EmberTensor) else t for t in tensors]
        tensor = stack(backend_tensors, axis)
        return EmberTensor(tensor, dtype=self.dtype, device=self.device, requires_grad=self._requires_grad)
    
    def split(self, x: Any, num_or_size_splits: Union[int, Sequence[int]], axis: int = 0) -> List[TensorInterface]:
        """
        Split a tensor into sub-tensors.
        
        Args:
            x: Input tensor
            num_or_size_splits: Number of splits or sizes of each split
            axis: Axis along which to split
            
        Returns:
            List of sub-tensors
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        tensors = split(x, num_or_size_splits, axis)
        return [EmberTensor(t, dtype=self.dtype, device=self.device, requires_grad=self._requires_grad) for t in tensors]
    
    def expand_dims(self, x: Any, axis: Union[int, Sequence[int]]) -> 'EmberTensor':
        """
        Insert new axes into a tensor's shape.
        
        Args:
            x: Input tensor
            axis: Position(s) where new axes should be inserted
            
        Returns:
            Tensor with expanded dimensions
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        tensor = expand_dims(x, axis)
        return EmberTensor(tensor, dtype=self.dtype, device=self.device, requires_grad=self._requires_grad)
    
    def squeeze(self, x: Any, axis: Optional[Union[int, Sequence[int]]] = None) -> 'EmberTensor':
        """
        Remove single-dimensional entries from a tensor's shape.
        
        Args:
            x: Input tensor
            axis: Position(s) where dimensions should be removed
            
        Returns:
            Tensor with squeezed dimensions
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        tensor = squeeze(x, axis)
        return EmberTensor(tensor, dtype=self.dtype, device=self.device, requires_grad=self._requires_grad)
    
    def tile(self, x: Any, reps: Sequence[int]) -> 'EmberTensor':
        """
        Construct a tensor by tiling a given tensor.
        
        Args:
            x: Input tensor
            reps: Number of repetitions along each dimension
            
        Returns:
            Tiled tensor
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        tensor = tile(x, reps)
        return EmberTensor(tensor, dtype=self.dtype, device=self.device, requires_grad=self._requires_grad)
    
    def gather(self, x: Any, indices: Any, axis: int = 0) -> 'EmberTensor':
        """
        Gather slices from a tensor along an axis.
        
        Args:
            x: Input tensor
            indices: Indices of slices to gather
            axis: Axis along which to gather
            
        Returns:
            Gathered tensor
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        if isinstance(indices, EmberTensor):
            indices = indices.to_backend_tensor()
        tensor = gather(x, indices, axis)
        return EmberTensor(tensor, dtype=self.dtype, device=self.device, requires_grad=self._requires_grad)
    
    def convert_to_tensor(self, x: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
        """
        Convert input to a tensor.
        
        Args:
            x: Input data (array, tensor, scalar)
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor representation of the input
        """
        tensor = _convert_to_backend_tensor(x, dtype=dtype)
        return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
    
    def cast(self, x: Any, dtype: Union[DType, str, Callable[[], Any]]) -> 'EmberTensor':
        """
        Cast a tensor to a different data type.
        
        Args:
            x: Input tensor
            dtype: Target data type (can be a DType, string, or callable)
            
        Returns:
            Tensor with the target data type
        """
        # Handle different dtype formats
        processed_dtype = None
        if dtype is not None:
            if callable(dtype):
                # If dtype is a callable (like the lambda functions in dtypes.py),
                # call it to get the actual dtype
                processed_dtype = dtype()
            elif isinstance(dtype, str):
                # If dtype is a string, use it directly
                processed_dtype = dtype
            else:
                # Otherwise, use it as is (assuming it's a DType or compatible)
                processed_dtype = dtype
                
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        tensor = cast(x, processed_dtype)
        return EmberTensor(tensor, dtype=processed_dtype, device=self.device, requires_grad=self._requires_grad)
    
    def copy(self, x: Any) -> 'EmberTensor':
        """
        Create a copy of a tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Copy of the tensor
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        tensor = copy(x)
        return EmberTensor(tensor, dtype=self.dtype, device=self.device, requires_grad=self._requires_grad)
    
    def var(self, x: Any, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> 'EmberTensor':
        """
        Compute the variance of a tensor along specified axes.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the variance
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Variance of the tensor
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        tensor = var(x, axis, keepdims)
        return EmberTensor(tensor, dtype=self.dtype, device=self.device, requires_grad=self._requires_grad)
    
    def sort(self, x: Any, axis: int = -1, descending: bool = False) -> 'EmberTensor':
        """
        Sort a tensor along a specified axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to sort
            descending: Whether to sort in descending order
            
        Returns:
            Sorted tensor
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        tensor = sort(x, axis, descending)
        return EmberTensor(tensor, dtype=self.dtype, device=self.device, requires_grad=self._requires_grad)
    
    def argsort(self, x: Any, axis: int = -1, descending: bool = False) -> 'EmberTensor':
        """
        Return the indices that would sort a tensor along a specified axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to sort
            descending: Whether to sort in descending order
            
        Returns:
            Indices that would sort the tensor
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        tensor = argsort(x, axis, descending)
        return EmberTensor(tensor, dtype=self.dtype, device=self.device, requires_grad=self._requires_grad)
    
    def slice(self, x: Any, starts: Sequence[int], sizes: Sequence[int]) -> 'EmberTensor':
        """
        Extract a slice from a tensor.
        
        Args:
            x: Input tensor
            starts: Starting indices for each dimension
            sizes: Size of the slice in each dimension. A value of -1 means "all remaining elements in this dimension"
            
        Returns:
            Sliced tensor
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        tensor = slice(x, starts, sizes)
        return EmberTensor(tensor, dtype=self.dtype, device=self.device, requires_grad=self._requires_grad)
    
    def slice_update(self, x: Any, slices: Union[List, Tuple], updates: Any) -> 'EmberTensor':
        """
        Update a tensor at specific indices.
        
        Args:
            x: Input tensor to update
            slices: List or tuple of slice objects or indices
            updates: Values to insert at the specified indices
            
        Returns:
            Updated tensor
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        if isinstance(updates, EmberTensor):
            updates = updates.to_backend_tensor()
        tensor = slice_update(x, slices, updates)
        return EmberTensor(tensor, dtype=self.dtype, device=self.device, requires_grad=self._requires_grad)
    
    def pad(self, x: Any, paddings: Sequence[Sequence[int]], constant_values: Union[int, float] = 0) -> 'EmberTensor':
        """
        Pad a tensor with a constant value.
        
        Args:
            x: Input tensor
            paddings: Sequence of sequences of integers specifying the padding for each dimension
                     Each inner sequence should contain two integers: [pad_before, pad_after]
            constant_values: Value to pad with
            
        Returns:
            Padded tensor
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        tensor = pad(x, paddings, constant_values)
        return EmberTensor(tensor, dtype=self.dtype, device=self.device, requires_grad=self._requires_grad)
    
    def tensor_scatter_nd_update(self, tensor: Any, indices: Any, updates: Any) -> 'EmberTensor':
        """
        Updates values of a tensor at specified indices.

        Args:
            tensor: Input tensor to update
            indices: Indices at which to update values (N-dimensional indices)
            updates: Values to insert at the specified indices

        Returns:
            Updated tensor
        """
        if isinstance(tensor, EmberTensor):
            tensor = tensor.to_backend_tensor()
        if isinstance(indices, EmberTensor):
            indices = indices.to_backend_tensor()
        if isinstance(updates, EmberTensor):
            updates = updates.to_backend_tensor()
        result = tensor_scatter_nd_update(tensor, indices, updates)
        return EmberTensor(result, dtype=self.dtype, device=self.device, requires_grad=self._requires_grad)
    
    def maximum(self, x1: Any, x2: Any) -> 'EmberTensor':
        """
        Element-wise maximum of two tensors.
        
        Args:
            x1: First input tensor
            x2: Second input tensor
            
        Returns:
            Element-wise maximum
        """
        if isinstance(x1, EmberTensor):
            x1 = x1.to_backend_tensor()
        if isinstance(x2, EmberTensor):
            x2 = x2.to_backend_tensor()
        tensor = maximum(x1, x2)
        return EmberTensor(tensor, dtype=self.dtype, device=self.device, requires_grad=self._requires_grad)
    
    def random_normal(self, shape: Union[int, Sequence[int]], mean: float = 0.0, stddev: float = 1.0,
                     dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
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
        tensor = random_normal(shape, mean, stddev, dtype, device)
        return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
    
    def random_uniform(self, shape: Union[int, Sequence[int]], minval: float = 0.0, maxval: float = 1.0,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
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
        tensor = random_uniform(shape, minval, maxval, dtype, device)
        return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
    
    def random_binomial(self, shape: Union[int, Sequence[int]], p: float = 0.5,
                       dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
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
        tensor = random_bernoulli(shape, p, dtype, device)
        return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
    
    def random_gamma(self, shape: Union[int, Sequence[int]], alpha: float = 1.0, beta: float = 1.0,
                    dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
        """
        Generate random values from a gamma distribution.
        
        Args:
            shape: Shape of the output tensor
            alpha: Shape parameter
            beta: Scale parameter
            dtype: Optional data type
            device: Optional device to place the tensor on
        
        Returns:
            Tensor with random values from a gamma distribution
        """
        tensor = random_gamma(shape, alpha, beta, dtype, device)
        return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
    
    def random_exponential(self, shape: Union[int, Sequence[int]], scale: float = 1.0,
                          dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
        """
        Generate random values from an exponential distribution.
        
        Args:
            shape: Shape of the output tensor
            scale: Scale parameter
            dtype: Optional data type
            device: Optional device to place the tensor on
        
        Returns:
            Tensor with random values from an exponential distribution
        """
        tensor = random_exponential(shape, scale, dtype, device)
        return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
    
    def random_poisson(self, shape: Union[int, Sequence[int]], lam: float = 1.0,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
        """
        Generate random values from a Poisson distribution.
        
        Args:
            shape: Shape of the output tensor
            lam: Rate parameter
            dtype: Optional data type
            device: Optional device to place the tensor on
        
        Returns:
            Tensor with random values from a Poisson distribution
        """
        tensor = random_poisson(shape, lam, dtype, device)
        return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
    
    def random_categorical(self, logits: Any, num_samples: int,
                          dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
        """
        Draw samples from a categorical distribution.
        
        Args:
            logits: 2D tensor with unnormalized log probabilities
            num_samples: Number of samples to draw
            dtype: Optional data type
            device: Optional device to place the tensor on
        
        Returns:
            Tensor with random categorical values
        """
        if isinstance(logits, EmberTensor):
            logits = logits.to_backend_tensor()
        tensor = random_categorical(logits, num_samples, dtype, device)
        return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
    
    def random_permutation(self, x: Union[int, Any], dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
        """
        Randomly permute a sequence or return a permuted range.
        
        Args:
            x: If x is an integer, randomly permute np.arange(x).
               If x is an array, make a copy and shuffle the elements randomly.
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Permuted tensor
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        tensor = random_permutation(x, dtype, device)
        return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
    
    def shuffle(self, x: Any) -> 'EmberTensor':
        """
        Randomly shuffle a tensor along the first dimension.
        
        Args:
            x: Input tensor
        
        Returns:
            Shuffled tensor
        """
        if isinstance(x, EmberTensor):
            x = x.to_backend_tensor()
        tensor = shuffle(x)
        return EmberTensor(tensor, dtype=self.dtype, device=self.device, requires_grad=self._requires_grad)
    
    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        set_seed(seed)
    
    def get_seed(self) -> Optional[int]:
        """
        Get the current random seed.
        
        Returns:
            Current random seed (None if not set)
        """
        return get_seed()
    
    def __setitem__(self, key, value):
        """
        Set values at specified indices.
        
        Args:
            key: Index or slice
            value: Value to set
        """
        if isinstance(value, EmberTensor):
            value = value.to_backend_tensor()
        self._tensor = slice_update(self._tensor, key, value)
        
    def __getstate__(self) -> dict:
        """
        Get the state of the tensor for serialization.
        
        Returns:
            Dictionary containing the tensor state
        """
        # Convert tensor to numpy for serialization
        tensor_data = to_numpy(self._tensor)
        
        # Get the dtype as a string
        dtype_str = str(self.dtype)
        
        # Return the state dictionary
        return {
            'tensor_data': tensor_data,
            'dtype': dtype_str,
            'device': self._device,
            'requires_grad': self._requires_grad,
            'backend': self._backend if hasattr(self, '_backend') else None
        }
    
    def __setstate__(self, state: dict) -> None:
        """
        Restore the tensor from a serialized state.
        
        Args:
            state: Dictionary containing the tensor state
        """
        # Extract state components
        tensor_data = state['tensor_data']
        dtype_str = state['dtype']
        device = state['device']
        requires_grad = state['requires_grad']
        backend = state.get('backend', None)
        
        # Convert the numpy array back to a backend tensor
        self._tensor = _convert_to_backend_tensor(tensor_data, dtype=dtype_str)
        
        # Set the other attributes
        self._device = device
        self._requires_grad = requires_grad
        self._dtype = DType(dtype_str.split('.')[-1]) if dtype_str else None
        self._backend = backend
    
    def __iter__(self):
        """
        Make the tensor iterable.
        
        Returns:
            Iterator over the tensor elements
        """
        # Convert to numpy and iterate over it
        return iter(to_numpy(self._tensor))