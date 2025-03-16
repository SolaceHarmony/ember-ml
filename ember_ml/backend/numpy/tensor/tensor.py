"""NumPy tensor implementation for ember_ml.

This module provides NumPy implementations of tensor operations.
"""

import numpy as np
from typing import Union, Optional, Sequence, Any, List, Tuple

from ember_ml.backend.numpy.tensor.dtype import NumpyDType

# Type aliases
Shape = Union[int, Sequence[int]]
DType = Any

def _convert_input(x: Any) -> np.ndarray:
    """Convert input to NumPy array."""
    if isinstance(x, np.ndarray):
        return x
    # Handle EmberTensor objects
    if isinstance(x, object) and getattr(x.__class__, '__name__', '') == 'EmberTensor':
        # For EmberTensor, extract the underlying NumPy array
        return getattr(x, '_tensor')
    # Convert other types to NumPy array
    try:
        return np.array(x)
    except:
        raise ValueError(f"Cannot convert {type(x)} to NumPy array")

def _validate_dtype(dtype_cls: NumpyDType, dtype: Optional[DType]) -> Optional[Any]:
    """
    Validate and convert dtype to NumPy format.
    
    Args:
        dtype_cls: NumpyDType instance for conversions
        dtype: Input dtype to validate
        
    Returns:
        Validated NumPy dtype or None
    """
    if dtype is None:
        return None
    
    # Handle string dtypes
    if isinstance(dtype, str):
        return dtype_cls.from_dtype_str(dtype)
        
    # Handle EmberDType objects
    if hasattr(dtype, 'name'):
        return dtype_cls.from_dtype_str(str(dtype.name))
        
    # If it's already a NumPy dtype, return as is
    if isinstance(dtype, np.dtype) or dtype in [np.float32, np.float64, np.int32, np.int64, 
                                               np.bool_, np.int8, np.int16, np.uint8, 
                                               np.uint16, np.uint32, np.uint64, np.float16]:
        return dtype
        
    raise ValueError(f"Invalid dtype: {dtype}")

class NumpyTensor:
    """NumPy implementation of tensor operations."""
    
    def __init__(self):
        """Initialize NumPy tensor operations."""
        self._dtype_cls = NumpyDType()
    
    def convert_to_tensor(self, data: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Convert data to a NumPy array.
        
        Args:
            data: The data to convert
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            NumPy array
        """
        tensor = _convert_input(data)
        if dtype is not None:
            numpy_dtype = _validate_dtype(self._dtype_cls, dtype)
            if numpy_dtype is not None:
                tensor = tensor.astype(numpy_dtype)
        # device parameter is ignored for NumPy backend
        return tensor
    
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """
        Convert a tensor to a NumPy array.
        
        Args:
            tensor: The tensor to convert
            
        Returns:
            NumPy array
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        # For non-array types, convert to NumPy array
        return self.convert_to_tensor(tensor)
    
    def item(self, tensor: Any) -> Union[int, float, bool]:
        """
        Get the value of a scalar tensor.
        
        Args:
            tensor: The tensor to get the value from
            
        Returns:
            The scalar value
        """
        if isinstance(tensor, np.ndarray):
            return tensor.item()
        # For non-array types, convert to scalar
        return self.convert_to_tensor(tensor).item()
    
    def shape(self, tensor: Any) -> Tuple[int, ...]:
        """
        Get the shape of a tensor.
        
        Args:
            tensor: The tensor to get the shape of
            
        Returns:
            The shape of the tensor
        """
        return self.convert_to_tensor(tensor).shape
    
    def dtype(self, tensor: Any) -> Any:
        """
        Get the data type of a tensor.
        
        Args:
            tensor: The tensor to get the data type of
            
        Returns:
            The data type of the tensor
        """
        return self.convert_to_tensor(tensor).dtype
    
    def zeros(self, shape: Shape, dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create a tensor of zeros.
        
        Args:
            shape: The shape of the tensor
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            Tensor of zeros
        """
        from ember_ml.backend.numpy.tensor.ops.creation import zeros as zeros_func
        return zeros_func(self, shape, dtype, device)
    
    def ones(self, shape: Shape, dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create a tensor of ones.
        
        Args:
            shape: The shape of the tensor
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            Tensor of ones
        """
        from ember_ml.backend.numpy.tensor.ops.creation import ones as ones_func
        return ones_func(self, shape, dtype, device)
    
    def zeros_like(self, tensor: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create a tensor of zeros with the same shape as the input.
        
        Args:
            tensor: The input tensor
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            Tensor of zeros with the same shape as the input
        """
        from ember_ml.backend.numpy.tensor.ops.creation import zeros_like as zeros_like_func
        return zeros_like_func(self, tensor, dtype, device)
    
    def ones_like(self, tensor: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create a tensor of ones with the same shape as the input.
        
        Args:
            tensor: The input tensor
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            Tensor of ones with the same shape as the input
        """
        from ember_ml.backend.numpy.tensor.ops.creation import ones_like as ones_like_func
        return ones_like_func(self, tensor, dtype, device)
    
    def eye(self, n: int, m: Optional[int] = None, dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create an identity matrix.
        
        Args:
            n: Number of rows
            m: Number of columns (default: n)
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            Identity matrix
        """
        from ember_ml.backend.numpy.tensor.ops.creation import eye as eye_func
        return eye_func(self, n, m, dtype, device)
    
    def reshape(self, tensor: Any, shape: Shape) -> np.ndarray:
        """
        Reshape a tensor.
        
        Args:
            tensor: The tensor to reshape
            shape: The new shape
            
        Returns:
            Reshaped tensor
        """
        from ember_ml.backend.numpy.tensor.ops.manipulation import reshape as reshape_func
        return reshape_func(self, tensor, shape)
    
    def transpose(self, tensor: Any, axes: Optional[Sequence[int]] = None) -> np.ndarray:
        """
        Transpose a tensor.
        
        Args:
            tensor: The tensor to transpose
            axes: Optional permutation of dimensions
            
        Returns:
            Transposed tensor
        """
        from ember_ml.backend.numpy.tensor.ops.manipulation import transpose as transpose_func
        return transpose_func(self, tensor, axes)
    
    def concatenate(self, tensors: Sequence[Any], axis: int = 0) -> np.ndarray:
        """
        Concatenate tensors along a specified axis.
        
        Args:
            tensors: The tensors to concatenate
            axis: The axis along which to concatenate
            
        Returns:
            Concatenated tensor
        """
        from ember_ml.backend.numpy.tensor.ops.manipulation import concatenate as concatenate_func
        return concatenate_func(self, tensors, axis)
    
    def stack(self, tensors: Sequence[Any], axis: int = 0) -> np.ndarray:
        """
        Stack tensors along a new axis.
        
        Args:
            tensors: The tensors to stack
            axis: The axis along which to stack
            
        Returns:
            Stacked tensor
        """
        from ember_ml.backend.numpy.tensor.ops.manipulation import stack as stack_func
        return stack_func(self, tensors, axis)
    
    def split(self, tensor: Any, num_or_size_splits: Union[int, Sequence[int]], axis: int = 0) -> List[np.ndarray]:
        """
        Split a tensor into sub-tensors.
        
        Args:
            tensor: The tensor to split
            num_or_size_splits: Number of splits or sizes of each split
            axis: The axis along which to split
            
        Returns:
            List of sub-tensors
        """
        from ember_ml.backend.numpy.tensor.ops.manipulation import split as split_func
        return split_func(self, tensor, num_or_size_splits, axis)
    
    def expand_dims(self, tensor: Any, axis: Union[int, Sequence[int]]) -> np.ndarray:
        """
        Insert a new axis into a tensor's shape.
        
        Args:
            tensor: The tensor to expand
            axis: The axis at which to insert the new dimension
            
        Returns:
            Expanded tensor
        """
        from ember_ml.backend.numpy.tensor.ops.manipulation import expand_dims as expand_dims_func
        return expand_dims_func(self, tensor, axis)
    
    def squeeze(self, tensor: Any, axis: Optional[Union[int, Sequence[int]]] = None) -> np.ndarray:
        """
        Remove single-dimensional entries from a tensor's shape.
        
        Args:
            tensor: The tensor to squeeze
            axis: The axis to remove
            
        Returns:
            Squeezed tensor
        """
        from ember_ml.backend.numpy.tensor.ops.manipulation import squeeze as squeeze_func
        return squeeze_func(self, tensor, axis)
    
    def cast(self, tensor: Any, dtype: DType) -> np.ndarray:
        """
        Cast a tensor to a different data type.
        
        Args:
            tensor: The tensor to cast
            dtype: The target data type
            
        Returns:
            Cast tensor
        """
        from ember_ml.backend.numpy.tensor.ops.casting import cast as cast_func
        return cast_func(self, tensor, dtype)
    
    def copy(self, tensor: Any) -> np.ndarray:
        """
        Create a copy of a tensor.
        
        Args:
            tensor: The tensor to copy
            
        Returns:
            Copy of the tensor
        """
        tensor_np = self.convert_to_tensor(tensor)
        return tensor_np.copy()
    
    def random_normal(self, shape: Shape, mean: float = 0.0, stddev: float = 1.0,
                     dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create a tensor with random values from a normal distribution.
        
        Args:
            shape: The shape of the tensor
            mean: The mean of the normal distribution
            stddev: The standard deviation of the normal distribution
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            Tensor with random values from a normal distribution
        """
        from ember_ml.backend.numpy.tensor.ops.random import random_normal as random_normal_func
        return random_normal_func(self, shape, mean, stddev, dtype, device)
    
    def random_uniform(self, shape: Shape, minval: float = 0.0, maxval: float = 1.0,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create a tensor with random values from a uniform distribution.
        
        Args:
            shape: The shape of the tensor
            minval: Minimum value
            maxval: Maximum value
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            Tensor with random values from a uniform distribution
        """
        from ember_ml.backend.numpy.tensor.ops.random import random_uniform as random_uniform_func
        return random_uniform_func(self, shape, minval, maxval, dtype, device)
    
    def random_binomial(self, shape: Shape, p: float = 0.5,
                       dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create a tensor with random values from a binomial distribution.
        
        Args:
            shape: The shape of the tensor
            p: Probability of success
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            Tensor with random values from a binomial distribution
        """
        from ember_ml.backend.numpy.tensor.ops.random import random_binomial as random_binomial_func
        return random_binomial_func(self, shape, p, dtype, device)
    
    def random_gamma(self, shape: Shape, alpha: float = 1.0, beta: float = 1.0,
                    dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Generate random values from a gamma distribution.
        
        Args:
            shape: Shape of the output array
            alpha: Shape parameter
            beta: Scale parameter
            dtype: Optional data type
            device: Ignored for NumPy backend
        
        Returns:
            NumPy array with random values from a gamma distribution
        """
        from ember_ml.backend.numpy.tensor.ops.random import random_gamma as random_gamma_func
        return random_gamma_func(self, shape, alpha, beta, dtype, device)
    
    def random_exponential(self, shape: Shape, scale: float = 1.0,
                          dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Generate random values from an exponential distribution.
        
        Args:
            shape: Shape of the output array
            scale: Scale parameter
            dtype: Optional data type
            device: Ignored for NumPy backend
        
        Returns:
            NumPy array with random values from an exponential distribution
        """
        from ember_ml.backend.numpy.tensor.ops.random import random_exponential as random_exponential_func
        return random_exponential_func(self, shape, scale, dtype, device)
    
    def random_poisson(self, shape: Shape, lam: float = 1.0,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Generate random values from a Poisson distribution.
        
        Args:
            shape: Shape of the output array
            lam: Rate parameter
            dtype: Optional data type
            device: Ignored for NumPy backend
        
        Returns:
            NumPy array with random values from a Poisson distribution
        """
        from ember_ml.backend.numpy.tensor.ops.random import random_poisson as random_poisson_func
        return random_poisson_func(self, shape, lam, dtype, device)
    
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
        from ember_ml.backend.numpy.tensor.ops.random import random_categorical as random_categorical_func
        return random_categorical_func(self, logits, num_samples, dtype, device)
    
    def random_permutation(self, x: Union[int, Any], dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Randomly permute a sequence or return a permuted range.
        
        Args:
            x: If x is an integer, randomly permute np.arange(x).
               If x is an array, make a copy and shuffle the elements randomly.
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            Permuted array
        """
        from ember_ml.backend.numpy.tensor.ops.random import random_permutation as random_permutation_func
        return random_permutation_func(self, x, dtype, device)
    
    def shuffle(self, x: Any) -> np.ndarray:
        """
        Randomly shuffle a NumPy array along its first dimension.
        
        Args:
            x: Input array
            
        Returns:
            Shuffled NumPy array
        """
        from ember_ml.backend.numpy.tensor.ops.random import shuffle as shuffle_func
        return shuffle_func(self, x)
    
    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        from ember_ml.backend.numpy.tensor.ops.random import set_seed as set_seed_func
        return set_seed_func(self, seed)
    
    def get_seed(self) -> Optional[int]:
        """
        Get the current random seed.
        
        Returns:
            Current random seed or None if not set
        """
        from ember_ml.backend.numpy.tensor.ops.random import get_seed as get_seed_func
        return get_seed_func(self)
    
    def full(self, shape: Shape, fill_value: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create a tensor filled with a scalar value.
        
        Args:
            shape: Shape of the tensor
            fill_value: Value to fill the tensor with
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            Tensor filled with the specified value
        """
        from ember_ml.backend.numpy.tensor.ops.creation import full as full_func
        return full_func(self, shape, fill_value, dtype, device)
    
    def full_like(self, tensor: Any, fill_value: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create a tensor filled with a scalar value with the same shape as the input.
        
        Args:
            tensor: Input tensor
            fill_value: Value to fill the tensor with
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            Tensor filled with the specified value with the same shape as tensor
        """
        from ember_ml.backend.numpy.tensor.ops.creation import full_like as full_like_func
        return full_like_func(self, tensor, fill_value, dtype, device)
    
    def arange(self, start: Union[int, float], stop: Optional[Union[int, float]] = None, step: int = 1,
              dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create a tensor with evenly spaced values within a given interval.
        
        Args:
            start: Start of interval (inclusive)
            stop: End of interval (exclusive)
            step: Spacing between values
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            Tensor with evenly spaced values
        """
        from ember_ml.backend.numpy.tensor.ops.creation import arange as arange_func
        return arange_func(self, start, stop, step, dtype, device)
    
    def linspace(self, start: Union[int, float], stop: Union[int, float], num: int,
                dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
        """
        Create a tensor with evenly spaced values within a given interval.
        
        Args:
            start: Start of interval (inclusive)
            stop: End of interval (inclusive)
            num: Number of values to generate
            dtype: Optional data type
            device: Ignored for NumPy backend
            
        Returns:
            Tensor with evenly spaced values
        """
        from ember_ml.backend.numpy.tensor.ops.creation import linspace as linspace_func
        return linspace_func(self, start, stop, num, dtype, device)
    
    def tile(self, tensor: Any, reps: Union[int, Sequence[int]]) -> np.ndarray:
        """
        Construct a tensor by tiling a given tensor.
        
        Args:
            tensor: Input tensor
            reps: Number of repetitions along each dimension
            
        Returns:
            Tiled tensor
        """
        from ember_ml.backend.numpy.tensor.ops.manipulation import tile as tile_func
        return tile_func(self, tensor, reps)
    
    def gather(self, tensor: Any, indices: Any, axis: int = 0) -> np.ndarray:
        """
        Gather slices from a tensor along an axis.
        
        Args:
            tensor: Input tensor
            indices: Indices of slices to gather
            axis: Axis along which to gather
            
        Returns:
            Gathered tensor
        """
        from ember_ml.backend.numpy.tensor.ops.indexing import gather as gather_func
        return gather_func(self, tensor, indices, axis)
    
    def slice(self, tensor: Any, starts: Sequence[int], sizes: Sequence[int]) -> np.ndarray:
        """
        Extract a slice from a tensor.
        
        Args:
            tensor: Input tensor
            starts: Starting indices for each dimension
            sizes: Size of the slice in each dimension. A value of -1 means "all remaining elements in this dimension"
            
        Returns:
            Sliced tensor
        """
        from ember_ml.backend.numpy.tensor.ops.indexing import slice_tensor as slice_tensor_func
        return slice_tensor_func(self, tensor, starts, sizes)
    
    def slice_update(self, tensor: Any, slices: Any, updates: Any) -> np.ndarray:
        """
        Update a slice of a tensor with new values.
        
        Args:
            tensor: Input tensor to update
            slices: Starting indices for each dimension
            updates: Values to insert at the specified indices
            
        Returns:
            Updated tensor
        """
        from ember_ml.backend.numpy.tensor.ops.indexing import slice_update as slice_update_func
        return slice_update_func(self, tensor, slices, updates)
    
    def pad(self, tensor: Any, paddings: Sequence[Sequence[int]], constant_values: int = 0) -> np.ndarray:
        """
        Pad a tensor with a constant value.
        
        Args:
            tensor: Input tensor
            paddings: Sequence of sequences of integers specifying the padding for each dimension
                    Each inner sequence should contain two integers: [pad_before, pad_after]
            constant_values: Value to pad with
            
        Returns:
            Padded tensor
        """
        from ember_ml.backend.numpy.tensor.ops.manipulation import pad as pad_func
        return pad_func(self, tensor, paddings, constant_values)
    
    def tensor_scatter_nd_update(self, tensor: Any, indices: Any, updates: Any) -> np.ndarray:
        """
        Updates values of a tensor at specified indices.
        
        Args:
            tensor: Input tensor to update
            indices: Indices at which to update values (N-dimensional indices)
            updates: Values to insert at the specified indices
            
        Returns:
            Updated tensor
        """
        from ember_ml.backend.numpy.tensor.ops.indexing import tensor_scatter_nd_update as tensor_scatter_nd_update_func
        return tensor_scatter_nd_update_func(self, tensor, indices, updates)
    
    def maximum(self, x: Any, y: Any) -> np.ndarray:
        """
        Element-wise maximum of two tensors.
        
        Args:
            x: First input tensor
            y: Second input tensor
            
        Returns:
            Element-wise maximum
        """
        x_np = self.convert_to_tensor(x)
        y_np = self.convert_to_tensor(y)
        return np.maximum(x_np, y_np)
    
    def sort(self, tensor: Any, axis: int = -1, descending: bool = False) -> np.ndarray:
        """
        Sort a tensor along a specified axis.
        
        Args:
            tensor: Input tensor
            axis: Axis along which to sort
            descending: Whether to sort in descending order
            
        Returns:
            Sorted tensor
        """
        tensor_np = self.convert_to_tensor(tensor)
        
        # Sort the tensor
        if descending:
            return -np.sort(-tensor_np, axis=axis)
        else:
            return np.sort(tensor_np, axis=axis)
    
    def argsort(self, tensor: Any, axis: int = -1, descending: bool = False) -> np.ndarray:
        """
        Return the indices that would sort a tensor along a specified axis.
        
        Args:
            tensor: Input tensor
            axis: Axis along which to sort
            descending: Whether to sort in descending order
            
        Returns:
            Indices that would sort the tensor
        """
        tensor_np = self.convert_to_tensor(tensor)
        
        # Get the indices that would sort the tensor
        if descending:
            return np.argsort(-tensor_np, axis=axis)
        else:
            return np.argsort(tensor_np, axis=axis)
    
    def scatter(self, src: Any, index: Any, dim_size: Optional[int] = None,
               aggr: str = "add", axis: int = 0) -> np.ndarray:
        """
        Scatter values from src into a new tensor of size dim_size along the given axis.
        
        Args:
            src: Source tensor containing values to scatter
            index: Indices where to scatter the values
            dim_size: Size of the output tensor along the given axis. If None, uses the maximum index + 1
            aggr: Aggregation method to use for duplicate indices ("add", "max", "mean", "softmax", "min")
            axis: Axis along which to scatter
            
        Returns:
            Tensor with scattered values
        """
        from ember_ml.backend.numpy.tensor.ops.indexing import scatter as scatter_func
        return scatter_func(self, src, index, dim_size, aggr, axis)
    
    def var(self, tensor: Any, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> np.ndarray:
        """
        Compute the variance of a tensor along specified axes.
        
        Args:
            tensor: Input tensor
            axis: Axis or axes along which to compute the variance
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Variance of the tensor
        """
        tensor_np = self.convert_to_tensor(tensor)
        return np.var(tensor_np, axis=axis, keepdims=keepdims)