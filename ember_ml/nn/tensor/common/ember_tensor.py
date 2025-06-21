"""
Backend-agnostic EmberTensor implementation.

This module provides a common implementation of the tensor interface that works
with any backend (NumPy, PyTorch, MLX) using the backend abstraction layer.
"""


from typing import Any, Optional, List, Union, Tuple, Sequence, Callable, Iterator,TYPE_CHECKING
TensorLike = Any
DType = Any
if TYPE_CHECKING:
    from ember_ml.nn.tensor.types import DType
from ember_ml.nn.tensor.common.dtypes import EmberDType
from ember_ml.nn.tensor.interfaces import TensorInterface
from ember_ml.nn.tensor.common import (
    _convert_to_backend_tensor, to_numpy, item, shape, dtype, zeros, ones, zeros_like, ones_like,
    eye, arange, linspace, full, full_like, reshape, transpose, concatenate, stack, split,
    expand_dims, squeeze, tile, gather, scatter, tensor_scatter_nd_update, slice_tensor, slice_update,
    cast, copy, pad, maximum, random_normal, random_uniform,
    random_bernoulli, random_gamma, random_exponential, random_poisson,
    random_categorical, random_permutation, shuffle, set_seed, get_seed, tolist
)
from ember_ml import ops

class EmberTensor(TensorInterface):
    """
    A backend-agnostic tensor implementation using the backend abstraction layer.
    
    This implementation delegates all operations to the current backend through
    the backend abstraction layer, ensuring backend purity and compatibility.
    """

    def __repr__(self) -> str:
        """Return a string representation of the tensor."""
        return f"EmberTensor({to_numpy(self._tensor)})"
    
    def __str__(self) -> str:
        """Return a string representation of the tensor.
        
        Returns a consistent string representation across all backends.
        """
        # Create a consistent string representation that doesn't rely on backend-specific methods
        # Format: array([...], dtype=dtype)
        if len(self.shape) == 0:  # Scalar
            return f"array({self.item()}, dtype={self._dtype})"
        elif len(self.shape) == 1:  # 1D tensor
            # For 1D tensors, iterate through elements and format them
            elements = []
            for i in range(self.shape[0]):
                elements.append(str(self[i].item()))
            return f"array([{', '.join(elements)}], dtype={self._dtype})"
        else:  # Higher dimensional tensor
            # For higher dimensional tensors, just show shape and dtype
            return f"array(shape={self.shape}, dtype={self._dtype})"
 
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
        from ember_ml.nn.tensor import dtype as get_dtype
        # Figure out the dtype being inputted
        processed_dtype = None
        if dtype is not None:
            if isinstance(dtype, EmberDType):
                # If dtype is an EmberDType, use it directly
                processed_dtype = dtype
            elif isinstance(dtype, str):
                # If dtype is a string, use it directly
                processed_dtype = dtype
            else:
                # Otherwise, use it as is (assuming it's a DType or compatible)
                processed_dtype = dtype
            self._tensor = _convert_to_backend_tensor(data, dtype=processed_dtype)
        else:
            # If dtype is None, we need to determine the dtype from the data
            if data is not None:
                # Use the backend's dtype function to get the dtype
                self._tensor = _convert_to_backend_tensor(data)
                processed_dtype = get_dtype(self._tensor)
            else:
                # If no data is provided, initialize an empty tensor
                self._tensor = _convert_to_backend_tensor(data)
                backend_dtype = get_dtype(self._tensor) if dtype is None and self._tensor is not None else processed_dtype

        # Import get_backend_module directly for reliable access during init
        self._device = device if device is not None else ops.get_device()
        self._requires_grad = requires_grad
        self._backend = ops.get_backend() # get_backend is safe here
        


    def to_backend_tensor(self) -> Any:
        """Get the underlying backend tensor."""
        return self._tensor

    def __array__(self, dtype: Optional[DType] = None) -> Any:
        """Array interface.

        This method is part of NumPy's array interface protocol, which allows
        NumPy to convert objects to NumPy arrays. We use the to_numpy function
        from the backend abstraction layer to ensure backend purity.

        Args:
            dtype: The desired data type of the array.

        Returns:
            NumPy array representation of the tensor.
        """
        # Use to_numpy from the backend abstraction layer
        # This is a special case where we're allowed to use NumPy because
        # it's part of the NumPy array interface protocol
        if dtype is not None:
            return to_numpy(self._tensor).astype(dtype)
        return to_numpy(self._tensor)
    
    def __getitem__(self, key) -> Any: # Returns raw backend tensor or scalar
        """
        Get values at specified indices. Returns a raw backend tensor or scalar.
        
        Args:
            key: Index or slice
            
        Returns:
            Raw backend tensor or scalar value.
        """
        # common.slice_tensor is expected to return a raw backend tensor.
        # If the result of slicing is a 0-dim tensor that should be a scalar,
        # this will still return that 0-dim backend tensor. User can call .item() on it via EmberTensor(result).item().
        return slice_tensor(self._tensor, key)

    def item(self) -> Union[int, float, bool]:
        """Get the value of a scalar tensor."""
        return item(self._tensor)

    def tolist(self) -> List[Any]:
        """Convert tensor to a (nested) list."""
        return tolist(self._tensor)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the tensor."""
        return shape(self._tensor)

    @property
    def dtype(self) -> DType:
        """Get the dtype of the tensor."""
        # Use the stored EmberDType if available
        if hasattr(self, '_dtype') and self._dtype is not None:
            # Return the stored dtype directly
            return self._dtype
            
        # Otherwise, get it from the backend tensor
        backend_dtype = dtype(self._tensor)
        # If it's a backend-specific dtype, extract the name and create a string representation
        dtype_name = str(backend_dtype).split('.')[-1]
        return dtype_name

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
        # Return the stored backend name
        return str(self._backend)

    @property
    def requires_grad(self) -> bool:
        """Get whether the tensor requires gradients."""
        return self._requires_grad

    def detach(self) -> Any: # Returns raw backend tensor
        """Create a new tensor detached from the computation graph."""
        # Assumes common.detach exists and returns a backend tensor
        # If not, this might just return self._tensor
        from ember_ml.nn.tensor.common import detach as common_detach
        try:
            return common_detach(self._tensor)
        except (AttributeError, NotImplementedError):
            # Fallback if common.detach is not implemented or available
            # This means the "detach" is only conceptual at the EmberTensor wrapper level
            # by creating a new wrapper with requires_grad=False.
            # However, the goal is to return a backend tensor.
            # If the backend tensor itself cannot be detached, return it as is.
            logger.warning("common.detach not available, returning original backend tensor for detach().")
            return self._tensor


    def to_numpy(self) -> Any:
        """Convert tensor to NumPy array."""
        return to_numpy(self._tensor)

    @staticmethod
    def zeros(shape: Union[int, Sequence[int]], dtype: Optional[DType] = None, device: Optional[str] = None) -> Any: # Returns raw backend tensor
        """
        Create a tensor of zeros. Returns a raw backend tensor.
        """
        final_dtype = dtype if dtype is not None else EmberDType.float32
        final_device = device if device is not None else ops.get_device()
        return zeros(shape, dtype=final_dtype, device=final_device)

    @staticmethod
    def ones(shape: Union[int, Sequence[int]], dtype: Optional[DType] = None, device: Optional[str] = None) -> Any: # Returns raw backend tensor
        """
        Create a tensor of ones. Returns a raw backend tensor.
        """
        final_dtype = dtype if dtype is not None else EmberDType.float32
        final_device = device if device is not None else ops.get_device()
        return ones(shape, dtype=final_dtype, device=final_device)

    @staticmethod
    def zeros_like(other: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any: # other is backend tensor, returns backend
        """
        Create a tensor of zeros with the same shape as the input.
        Input `other` is expected to be a raw backend tensor. Returns a raw backend tensor.
        """
        # If 'other' could be EmberTensor, unwrap it first:
        # other_backend = other.to_backend_tensor() if isinstance(other, EmberTensor) else other
        # For this refactor, assuming 'other' is already a backend tensor if called from functional API.
        # If called as EmberTensor.zeros_like(et_instance), then __init__ of other ETs would handle it.
        # This static method on EmberTensor should expect backend tensor if it's consistent.
        # However, the previous version took EmberTensor. Let's keep that for now for this method's direct calls.
        other_backend = other.to_backend_tensor() if isinstance(other, EmberTensor) else other

        final_dtype = dtype if dtype is not None else globals()['dtype'](other_backend) # Use global dtype common function
        final_device = device if device is not None else globals()['ops'].get_device_of_tensor(other_backend) # Requires this op

        return zeros_like(other_backend, dtype=final_dtype, device=final_device)

    @staticmethod
    def ones_like(other: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any: # other is backend tensor, returns backend
        """
        Create a tensor of ones with the same shape as the input.
        Input `other` is expected to be a raw backend tensor. Returns a raw backend tensor.
        """
        other_backend = other.to_backend_tensor() if isinstance(other, EmberTensor) else other
        final_dtype = dtype if dtype is not None else globals()['dtype'](other_backend)
        final_device = device if device is not None else globals()['ops'].get_device_of_tensor(other_backend)
        return ones_like(other_backend, dtype=final_dtype, device=final_device)

    @staticmethod
    def eye(n: int, m: Optional[int] = None, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any: # Returns raw backend tensor
        """
        Create an identity matrix. Returns a raw backend tensor.
        """
        final_dtype = dtype if dtype is not None else EmberDType.float32
        final_device = device if device is not None else ops.get_device()
        return eye(n, m=m, dtype=final_dtype, device=final_device)

    @staticmethod
    def arange(start: int, stop: Optional[int] = None, step: int = 1, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any: # Returns raw backend tensor
        """
        Create a tensor with evenly spaced values. Returns a raw backend tensor.
        """
        final_dtype = dtype if dtype is not None else EmberDType.int64
        final_device = device if device is not None else ops.get_device()
        return arange(start, stop=stop, step=step, dtype=final_dtype, device=final_device)

    @staticmethod
    def linspace(start: float, stop: float, num: int, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any: # Returns raw backend tensor
        """
        Create a tensor with evenly spaced values. Returns a raw backend tensor.
        """
        final_dtype = dtype if dtype is not None else EmberDType.float32
        final_device = device if device is not None else ops.get_device()
        return linspace(start, stop, num, dtype=final_dtype, device=final_device)

    @staticmethod
    def full(shape: Union[int, Sequence[int]], fill_value: Union[float, int], dtype: Optional[DType] = None, device: Optional[str] = None) -> Any: # Returns raw backend tensor
        """
        Create a tensor filled with a scalar value. Returns a raw backend tensor.
        """
        final_dtype = dtype
        if final_dtype is None: # Infer from fill_value
            if isinstance(fill_value, int): final_dtype = EmberDType.int64
            elif isinstance(fill_value, float): final_dtype = EmberDType.float32
            else: final_dtype = EmberDType.float32
        final_device = device if device is not None else ops.get_device()
        return full(shape, fill_value, dtype=final_dtype, device=final_device)

    @staticmethod
    def full_like(other: Any, fill_value: Union[float, int], dtype: Optional[DType] = None, device: Optional[str] = None) -> Any: # other is backend, returns backend
        """
        Create a tensor filled with a scalar value, with same shape as input.
        Input `other` is expected to be a raw backend tensor. Returns a raw backend tensor.
        """
        other_backend = other.to_backend_tensor() if isinstance(other, EmberTensor) else other
        final_dtype = dtype if dtype is not None else globals()['dtype'](other_backend)
        final_device = device if device is not None else globals()['ops'].get_device_of_tensor(other_backend)
        return full_like(other_backend, fill_value, dtype=final_dtype, device=final_device)
    
    # --- Operational Instance Methods: To be removed or return backend tensors ---
    # def reshape(self, shape: Union[int, Sequence[int]]) -> "EmberTensor": # REMOVE
    # def transpose(self, axes: Optional[Sequence[int]] = None) -> "EmberTensor": # REMOVE
    # def cast(self, dtype: Union[DType, str, Callable[[], Any]]): # REMOVE
    
    def copy(self) -> Any: # Returns raw backend tensor
        """ Create a copy of the underlying backend tensor. """
        return copy(self._tensor)

    # def squeeze(self, axis: Optional[Union[int, Sequence[int]]] = None): # REMOVE
    # def expand_dims(self, axis: Union[int, Sequence[int]]): # REMOVE
    # def tile(self, reps: Sequence[int]): # REMOVE
    # def pad(self, paddings: Sequence[Sequence[int]], constant_values: Union[int, float] = 0): # REMOVE
    # def sort(self, axis: int = -1, descending: bool = False): # REMOVE
    # def argsort(self, axis: int = -1, descending: bool = False): # REMOVE
    # def slice(self, starts: Sequence[int], sizes: Sequence[int]): # REMOVE (use __getitem__ or functional)
    # def maximum(self, other: "EmberTensor"): # REMOVE (use functional ops.maximum)
    # def shuffle(self) -> "EmberTensor": # REMOVE (use functional tensor.shuffle)


    @staticmethod
    def concatenate(tensors: Sequence[Any], axis: int = 0) -> Any: # Takes backend tensors, returns backend tensor
        """ Concatenate backend tensors along a specified axis. """
        if not tensors:
            raise ValueError("Cannot concatenate empty sequence of tensors.")
        # Assuming inputs are already backend tensors or EmberTensor.to_backend_tensor() has been called
        # For consistency, ensure they are backend tensors
        backend_tensors = [t.to_backend_tensor() if isinstance(t, EmberTensor) else t for t in tensors]
        return concatenate(backend_tensors, axis)

    @staticmethod
    def stack(tensors: Sequence[Any], axis: int = 0) -> Any: # Takes backend tensors, returns backend tensor
        """ Stack backend tensors along a new axis. """
        if not tensors:
            raise ValueError("Cannot stack empty sequence of tensors.")
        backend_tensors = [t.to_backend_tensor() if isinstance(t, EmberTensor) else t for t in tensors]
        return stack(backend_tensors, axis)
    
    # def split(self, num_or_size_splits: Union[int, Sequence[int]], axis: int = 0) -> List["EmberTensor"]: # REMOVE (use functional)
    
    @staticmethod
    def gather(x: Any, indices: Any, axis: int = 0) -> Any: # Takes backend, returns backend
        """ Gather slices from a backend tensor along an axis. """
        x_backend = x.to_backend_tensor() if isinstance(x, EmberTensor) else x
        indices_backend = indices.to_backend_tensor() if isinstance(indices, EmberTensor) else indices
        return gather(x_backend, indices_backend, axis)

    @staticmethod
    def scatter(data: Any, indices: Any,
                dim_size: Optional[Any] = None, aggr: str = 'sum', axis: int = 0) -> Any: # Takes backend, returns backend
        """ Scatter data according to indices into a new backend tensor. """
        data_backend = data.to_backend_tensor() if isinstance(data, EmberTensor) else data
        indices_backend = indices.to_backend_tensor() if isinstance(indices, EmberTensor) else indices
        backend_dim_size = dim_size.to_backend_tensor() if isinstance(dim_size, EmberTensor) else dim_size
        return scatter(data_backend, indices_backend, backend_dim_size, aggr, axis)

    @staticmethod
    def convert_to_tensor(x: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any: # Returns raw backend tensor
        """ Convert input to a raw backend tensor for the active backend. """
        # The requires_grad flag is for the EmberTensor wrapper, not the backend tensor itself directly usually.
        return _convert_to_backend_tensor(x, dtype=dtype, device=device)

    @staticmethod
    def tensor_scatter_nd_update(target_tensor: Any, indices: Any, updates: Any) -> Any: # Takes backend, returns backend
        """ Updates values of a backend tensor at specified indices. """
        target_backend = target_tensor.to_backend_tensor() if isinstance(target_tensor, EmberTensor) else target_tensor
        indices_backend = indices.to_backend_tensor() if isinstance(indices, EmberTensor) else indices
        updates_backend = updates.to_backend_tensor() if isinstance(updates, EmberTensor) else updates
        return tensor_scatter_nd_update(target_backend, indices_backend, updates_backend)

    # @staticmethod # From previous refactor, static_maximum takes EmberTensors and returns EmberTensor.
    # def static_maximum(x1: "EmberTensor", x2: "EmberTensor") -> "EmberTensor":
    # This should be removed as per new direction. Use ops.maximum(et1.to_backend_tensor(), et2.to_backend_tensor())

    @staticmethod
    def random_normal(shape: Union[int, Sequence[int]], mean: float = 0.0, stddev: float = 1.0,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> Any: # Returns raw backend tensor
        """ Create a backend tensor with random values from a normal distribution. """
        final_dtype = dtype if dtype is not None else EmberDType.float32
        final_device = device if device is not None else ops.get_device()
        return random_normal(shape, mean, stddev, final_dtype, final_device)

    @staticmethod
    def random_uniform(shape: Union[int, Sequence[int]], minval: float = 0.0, maxval: float = 1.0,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> Any: # Returns raw backend tensor
        """ Create a backend tensor with random values from a uniform distribution. """
        final_dtype = dtype if dtype is not None else EmberDType.float32
        final_device = device if device is not None else ops.get_device()
        return random_uniform(shape, minval, maxval, final_dtype, final_device)

    @staticmethod
    def random_binomial(shape: Union[int, Sequence[int]], p: float = 0.5, # Renamed from random_bernoulli in common
                       dtype: Optional[DType] = None, device: Optional[str] = None) -> Any: # Returns raw backend tensor
        """ Create a backend tensor with random values from a bernoulli/binomial distribution. """
        # common.random_bernoulli is used here.
        final_dtype = dtype if dtype is not None else EmberDType.int32
        final_device = device if device is not None else ops.get_device()
        return random_bernoulli(shape, p, final_dtype, final_device)

    @staticmethod
    def random_gamma(shape: Union[int, Sequence[int]], alpha: float = 1.0, beta: float = 1.0,
                    dtype: Optional[DType] = None, device: Optional[str] = None) -> Any: # Returns raw backend tensor
        """ Generate random backend tensor values from a gamma distribution. """
        final_dtype = dtype if dtype is not None else EmberDType.float32
        final_device = device if device is not None else ops.get_device()
        return random_gamma(shape, alpha, beta, final_dtype, final_device)

    @staticmethod
    def random_exponential(shape: Union[int, Sequence[int]], scale: float = 1.0,
                          dtype: Optional[DType] = None, device: Optional[str] = None) -> Any: # Returns raw backend tensor
        """ Generate random backend tensor values from an exponential distribution. """
        final_dtype = dtype if dtype is not None else EmberDType.float32
        final_device = device if device is not None else ops.get_device()
        return random_exponential(shape, scale, final_dtype, final_device)

    @staticmethod
    def random_poisson(shape: Union[int, Sequence[int]], lam: float = 1.0,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> Any: # Returns raw backend tensor
        """ Generate random backend tensor values from a Poisson distribution. """
        final_dtype = dtype if dtype is not None else EmberDType.int32
        final_device = device if device is not None else ops.get_device()
        return random_poisson(shape, lam, final_dtype, final_device)

    @staticmethod
    def random_categorical(logits: Any, num_samples: int, # logits is backend tensor
                          dtype: Optional[DType] = None, device: Optional[str] = None) -> Any: # Returns raw backend tensor
        """ Draw samples from a categorical distribution. Logits is a backend tensor. """
        logits_backend = logits.to_backend_tensor() if isinstance(logits, EmberTensor) else logits
        # Categorical results are indices, typically int
        final_dtype = dtype if dtype is not None else EmberDType.int64
        final_device = device if device is not None else globals()['ops'].get_device_of_tensor(logits_backend)
        return random_categorical(logits_backend, num_samples, final_dtype, final_device)

    @staticmethod
    def random_permutation(x: Union[int, Any], dtype: Optional[DType] = None, device: Optional[str] = None) -> Any: # x can be int or backend tensor
        """ Randomly permute a sequence or return a permuted range, as a backend tensor. """
        backend_x_input = x.to_backend_tensor() if isinstance(x, EmberTensor) else x
        # Determine default dtype and device more carefully
        # If x is int, this creates a range permutation. Dtype is usually int.
        # If x is tensor, it shuffles it. Dtype/device should match x.

        # This needs to call common.random_permutation
        return random_permutation(backend_x_input, dtype, device)

    @staticmethod
    def set_seed(seed: int) -> None:
        """ Set the random seed for reproducibility (delegates to common.set_seed). """
        set_seed(seed)

    @staticmethod
    def get_seed() -> Optional[int]:
        """ Get the current random seed (delegates to common.get_seed). """
        return get_seed()
    
    def __setitem__(self, key, value: Union[Any, "EmberTensor"]):
        """
        Set values at specified indices.
        
        Args:
            key: Index or slice
            value: Value to set (can be scalar, list, numpy array, or another EmberTensor)
        """
        value_backend = value.to_backend_tensor() if isinstance(value, EmberTensor) else value

        # slice_update from common should handle converting 'value_backend' if it's not yet a backend tensor
        # e.g. if user passes a Python list or scalar.
        # The common.slice_update is expected to return a new backend tensor.
        self._tensor = slice_update(self._tensor, key, value_backend)
        
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
            'backend': self._backend
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
        backend = state.get('backend', None)  # Get backend if available, otherwise None
        
        # Convert the numpy array back to a backend tensor
        self._tensor = _convert_to_backend_tensor(tensor_data, dtype=dtype_str)
        
        # Set the other attributes
        self._device = device
        self._requires_grad = requires_grad
        self._dtype = DType(dtype_str.split('.')[-1]) if dtype_str else None
        
        # Set the backend if available
        if backend is not None:
            self._backend = backend
        else:
            # If backend is not available, get the current backend
            from ember_ml.backend import get_backend
            self._backend = get_backend()
    
    def __iter__(self) -> Iterator[Any]:
        """
        Make the tensor iterable.
        
        Returns:
            Iterator over the tensor elements, where each element is a raw tensor
        """
        # Iterate directly over the backend tensor
        for element in self._tensor:
            # Return the raw tensor element
            yield element