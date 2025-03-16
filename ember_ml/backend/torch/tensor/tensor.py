"""
PyTorch tensor implementation for ember_ml.

This module provides PyTorch implementations of tensor operations.
"""

from ember_ml.backend.torch.tensor.dtype import TorchDType


class TorchTensor:
    """PyTorch implementation of tensor operations."""
    
    def __init__(self):
        """Initialize PyTorch tensor operations."""
        self._dtype_cls = TorchDType()
    
    def convert_to_tensor(self, data, dtype=None, device=None):
        """
        Convert data to a PyTorch tensor.
        
        Args:
            data: The data to convert
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            PyTorch tensor
        """
        from ember_ml.backend.torch.tensor.ops.utility import convert_to_tensor as convert_to_tensor_func
        return convert_to_tensor_func(self, data, dtype, device)
    
    def to_numpy(self, tensor):
        """
        Convert a PyTorch tensor to a NumPy-compatible array.
        
        Args:
            tensor: The tensor to convert
            
        Returns:
            NumPy-compatible array
        """
        from ember_ml.backend.torch.tensor.ops.utility import to_numpy as to_numpy_func
        return to_numpy_func(self, tensor)
    
    def item(self, tensor):
        """
        Get the value of a scalar tensor.
        
        Args:
            tensor: The tensor to get the value from
            
        Returns:
            The scalar value
        """
        from ember_ml.backend.torch.tensor.ops.utility import item as item_func
        return item_func(self, tensor)
    
    def shape(self, tensor):
        """
        Get the shape of a tensor.
        
        Args:
            tensor: The tensor to get the shape of
            
        Returns:
            The shape of the tensor
        """
        from ember_ml.backend.torch.tensor.ops.utility import shape as shape_func
        return shape_func(self, tensor)
    
    def dtype(self, tensor):
        """
        Get the data type of a tensor.
        
        Args:
            tensor: The tensor to get the data type of
            
        Returns:
            The data type of the tensor
        """
        from ember_ml.backend.torch.tensor.ops.utility import dtype as dtype_func
        return dtype_func(self, tensor)
    
    def zeros(self, shape, dtype=None, device=None):
        """
        Create a tensor of zeros.
        
        Args:
            shape: The shape of the tensor
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor of zeros
        """
        from ember_ml.backend.torch.tensor.ops.creation import zeros as zeros_func
        return zeros_func(self, shape, dtype, device)
    
    def ones(self, shape, dtype=None, device=None):
        """
        Create a tensor of ones.
        
        Args:
            shape: The shape of the tensor
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor of ones
        """
        from ember_ml.backend.torch.tensor.ops.creation import ones as ones_func
        return ones_func(self, shape, dtype, device)
    
    def zeros_like(self, tensor, dtype=None, device=None):
        """
        Create a tensor of zeros with the same shape as the input.
        
        Args:
            tensor: The input tensor
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor of zeros with the same shape as the input
        """
        from ember_ml.backend.torch.tensor.ops.creation import zeros_like as zeros_like_func
        return zeros_like_func(self, tensor, dtype, device)
    
    def ones_like(self, tensor, dtype=None, device=None):
        """
        Create a tensor of ones with the same shape as the input.
        
        Args:
            tensor: The input tensor
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor of ones with the same shape as the input
        """
        from ember_ml.backend.torch.tensor.ops.creation import ones_like as ones_like_func
        return ones_like_func(self, tensor, dtype, device)
    
    def eye(self, n, m=None, dtype=None, device=None):
        """
        Create an identity matrix.
        
        Args:
            n: Number of rows
            m: Number of columns (default: n)
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Identity matrix
        """
        from ember_ml.backend.torch.tensor.ops.creation import eye as eye_func
        return eye_func(self, n, m, dtype, device)
    
    def reshape(self, tensor, shape):
        """
        Reshape a tensor.
        
        Args:
            tensor: The tensor to reshape
            shape: The new shape
            
        Returns:
            Reshaped tensor
        """
        from ember_ml.backend.torch.tensor.ops.manipulation import reshape as reshape_func
        return reshape_func(self, tensor, shape)
    
    def transpose(self, tensor, axes=None):
        """
        Transpose a tensor.
        
        Args:
            tensor: The tensor to transpose
            axes: Optional permutation of dimensions
            
        Returns:
            Transposed tensor
        """
        from ember_ml.backend.torch.tensor.ops.manipulation import transpose as transpose_func
        return transpose_func(self, tensor, axes)
    
    def concatenate(self, tensors, axis=0):
        """
        Concatenate tensors along a specified axis.
        
        Args:
            tensors: The tensors to concatenate
            axis: The axis along which to concatenate
            
        Returns:
            Concatenated tensor
        """
        from ember_ml.backend.torch.tensor.ops.manipulation import concatenate as concatenate_func
        return concatenate_func(self, tensors, axis)
    
    def stack(self, tensors, axis=0):
        """
        Stack tensors along a new axis.
        
        Args:
            tensors: The tensors to stack
            axis: The axis along which to stack
            
        Returns:
            Stacked tensor
        """
        from ember_ml.backend.torch.tensor.ops.manipulation import stack as stack_func
        return stack_func(self, tensors, axis)
    
    def split(self, tensor, num_or_size_splits, axis=0):
        """
        Split a tensor into sub-tensors.
        
        Args:
            tensor: The tensor to split
            num_or_size_splits: Number of splits or sizes of each split
            axis: The axis along which to split
            
        Returns:
            List of sub-tensors
        """
        from ember_ml.backend.torch.tensor.ops.manipulation import split as split_func
        return split_func(self, tensor, num_or_size_splits, axis)
    
    def expand_dims(self, tensor, axis):
        """
        Insert a new axis into a tensor's shape.
        
        Args:
            tensor: The tensor to expand
            axis: The axis at which to insert the new dimension
            
        Returns:
            Expanded tensor
        """
        from ember_ml.backend.torch.tensor.ops.manipulation import expand_dims as expand_dims_func
        return expand_dims_func(self, tensor, axis)
    
    def squeeze(self, tensor, axis=None):
        """
        Remove single-dimensional entries from a tensor's shape.
        
        Args:
            tensor: The tensor to squeeze
            axis: The axis to remove
            
        Returns:
            Squeezed tensor
        """
        from ember_ml.backend.torch.tensor.ops.manipulation import squeeze as squeeze_func
        return squeeze_func(self, tensor, axis)
    
    def cast(self, tensor, dtype):
        """
        Cast a tensor to a different data type.
        
        Args:
            tensor: The tensor to cast
            dtype: The target data type
            
        Returns:
            Cast tensor
        """
        from ember_ml.backend.torch.tensor.ops.casting import cast as cast_func
        return cast_func(self, tensor, dtype)
    
    def copy(self, tensor):
        """
        Create a copy of a tensor.
        
        Args:
            tensor: The tensor to copy
            
        Returns:
            Copy of the tensor
        """
        from ember_ml.backend.torch.tensor.ops.utility import copy as copy_func
        return copy_func(self, tensor)
    
    def random_uniform(self, shape, minval=0.0, maxval=1.0, dtype=None, device=None):
        """
        Create a torch array with random values from a uniform distribution.
        
        Args:
            shape: Shape of the array
            minval: Minimum value
            maxval: Maximum value
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Torch array with random uniform values
        """
        from ember_ml.backend.torch.tensor.ops.random import random_uniform as random_uniform_func
        return random_uniform_func(self, shape, minval, maxval, dtype, device)
    
    def random_normal(self, shape, mean=0.0, stddev=1.0, dtype=None, device=None):
        """
        Create a tensor with random values from a normal distribution.
        
        Args:
            shape: The shape of the tensor
            mean: The mean of the normal distribution
            stddev: The standard deviation of the normal distribution
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor with random values from a normal distribution
        """
        from ember_ml.backend.torch.tensor.ops.random import random_normal as random_normal_func
        return random_normal_func(self, shape, mean, stddev, dtype, device)
    
    def arange(self, start, stop=None, step=1, dtype=None, device=None):
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
        from ember_ml.backend.torch.tensor.ops.creation import arange as arange_func
        return arange_func(self, start, stop, step, dtype, device)
    
    def linspace(self, start, stop, num, dtype=None, device=None):
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
        from ember_ml.backend.torch.tensor.ops.creation import linspace as linspace_func
        return linspace_func(self, start, stop, num, dtype, device)
    
    def full(self, shape, fill_value, dtype=None, device=None):
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
        from ember_ml.backend.torch.tensor.ops.creation import full as full_func
        return full_func(self, shape, fill_value, dtype, device)
    
    def full_like(self, tensor, fill_value, dtype=None, device=None):
        """
        Create a tensor filled with a scalar value with the same shape as the input.
        
        Args:
            tensor: Input tensor
            fill_value: Value to fill the tensor with
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor filled with the specified value with the same shape as tensor
        """
        from ember_ml.backend.torch.tensor.ops.creation import full_like as full_like_func
        return full_like_func(self, tensor, fill_value, dtype, device)
    
    def tile(self, tensor, reps):
        """
        Construct a tensor by tiling a given tensor.
        
        Args:
            tensor: Input tensor
            reps: Number of repetitions along each dimension
            
        Returns:
            Tiled tensor
        """
        from ember_ml.backend.torch.tensor.ops.manipulation import tile as tile_func
        return tile_func(self, tensor, reps)
    
    def gather(self, tensor, indices, axis=0):
        """
        Gather slices from a tensor along an axis.
        
        Args:
            tensor: Input tensor
            indices: Indices of slices to gather
            axis: Axis along which to gather
            
        Returns:
            Gathered tensor
        """
        from ember_ml.backend.torch.tensor.ops.indexing import gather as gather_func
        return gather_func(self, tensor, indices, axis)
    
    def var(self, tensor, axis=None, keepdims=False):
        """
        Compute the variance of a tensor along specified axes.
        
        Args:
            tensor: Input tensor
            axis: Axis or axes along which to compute the variance
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Variance of the tensor
        """
        from ember_ml.backend.torch.tensor.ops.utility import var as var_func
        return var_func(self, tensor, axis, keepdims)
    
    def sort(self, tensor, axis=-1, descending=False):
        """
        Sort a tensor along a specified axis.
        
        Args:
            tensor: Input tensor
            axis: Axis along which to sort
            descending: Whether to sort in descending order
            
        Returns:
            Sorted tensor
        """
        from ember_ml.backend.torch.tensor.ops.utility import sort as sort_func
        return sort_func(self, tensor, axis, descending)
    
    def argsort(self, tensor, axis=-1, descending=False):
        """
        Return the indices that would sort a tensor along a specified axis.
        
        Args:
            tensor: Input tensor
            axis: Axis along which to sort
            descending: Whether to sort in descending order
            
        Returns:
            Indices that would sort the tensor
        """
        from ember_ml.backend.torch.tensor.ops.utility import argsort as argsort_func
        return argsort_func(self, tensor, axis, descending)
    
    def slice(self, tensor, starts, sizes):
        """
        Extract a slice from a tensor.
        
        Args:
            tensor: Input tensor
            starts: Starting indices for each dimension
            sizes: Size of the slice in each dimension. A value of -1 means "all remaining elements in this dimension"
            
        Returns:
            Sliced tensor
        """
        from ember_ml.backend.torch.tensor.ops.indexing import slice_tensor as slice_tensor_func
        return slice_tensor_func(self, tensor, starts, sizes)
    
    def slice_update(self, tensor, slices, updates):
        """
        Update a tensor at specific indices.
        
        Args:
            tensor: Input tensor to update
            slices: List or tuple of slice objects or indices
            updates: Values to insert at the specified indices
            
        Returns:
            Updated tensor
        """
        from ember_ml.backend.torch.tensor.ops.indexing import slice_update as slice_update_func
        return slice_update_func(self, tensor, slices, updates)
    
    def pad(self, tensor, paddings, constant_values=0):
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
        from ember_ml.backend.torch.tensor.ops.manipulation import pad as pad_func
        return pad_func(self, tensor, paddings, constant_values)
    
    def tensor_scatter_nd_update(self, tensor, indices, updates):
        """
        Updates values of a tensor at specified indices.
        
        Args:
            tensor: Input tensor to update
            indices: Indices at which to update values (N-dimensional indices)
            updates: Values to insert at the specified indices
            
        Returns:
            Updated tensor
        """
        from ember_ml.backend.torch.tensor.ops.indexing import tensor_scatter_nd_update as tensor_scatter_nd_update_func
        return tensor_scatter_nd_update_func(self, tensor, indices, updates)
    
    def maximum(self, x, y):
        """
        Element-wise maximum of two tensors.
        
        Args:
            x: First input tensor
            y: Second input tensor
            
        Returns:
            Element-wise maximum
        """
        from ember_ml.backend.torch.tensor.ops.utility import maximum as maximum_func
        return maximum_func(self, x, y)
    
    def random_binomial(self, shape, p=0.5, dtype=None, device=None):
        """
        Create a tensor with random values from a binomial distribution.
        
        Args:
            shape: Shape of the tensor
            p: Probability of success
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor with random values from a binomial distribution
        """
        from ember_ml.backend.torch.tensor.ops.random import random_binomial as random_binomial_func
        return random_binomial_func(self, shape, p, dtype, device)
    
    def random_permutation(self, x, dtype=None, device=None):
        """
        Generate a random permutation.
        
        Args:
            x: If an integer, randomly permute integers from 0 to x-1.
               If a tensor, randomly permute along the first axis.
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor with random permutation
        """
        from ember_ml.backend.torch.tensor.ops.random import random_permutation as random_permutation_func
        return random_permutation_func(self, x, dtype, device)
    
    def random_exponential(self, shape, scale=1.0, dtype=None, device=None):
        """
        Generate random values from an exponential distribution.
        
        Args:
            shape: Shape of the output tensor
            scale: Scale parameter
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            PyTorch tensor with random values from an exponential distribution
        """
        from ember_ml.backend.torch.tensor.ops.random import random_exponential as random_exponential_func
        return random_exponential_func(self, shape, scale, dtype, device)
    
    def random_gamma(self, shape, alpha=1.0, beta=1.0, dtype=None, device=None):
        """
        Generate random values from a gamma distribution.
        
        Args:
            shape: Shape of the output tensor
            alpha: Shape parameter
            beta: Scale parameter
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            PyTorch tensor with random values from a gamma distribution
        """
        from ember_ml.backend.torch.tensor.ops.random import random_gamma as random_gamma_func
        return random_gamma_func(self, shape, alpha, beta, dtype, device)
    
    def random_poisson(self, shape, lam=1.0, dtype=None, device=None):
        """
        Generate random values from a Poisson distribution.
        
        Args:
            shape: Shape of the output tensor
            lam: Rate parameter
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            PyTorch tensor with random values from a Poisson distribution
        """
        from ember_ml.backend.torch.tensor.ops.random import random_poisson as random_poisson_func
        return random_poisson_func(self, shape, lam, dtype, device)
    
    def random_categorical(self, logits, num_samples, dtype=None, device=None):
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
        from ember_ml.backend.torch.tensor.ops.random import random_categorical as random_categorical_func
        return random_categorical_func(self, logits, num_samples, dtype, device)
    
    def shuffle(self, x):
        """
        Randomly shuffle a tensor along the first dimension.
        
        Args:
            x: Input tensor
            
        Returns:
            Shuffled tensor
        """
        from ember_ml.backend.torch.tensor.ops.random import shuffle as shuffle_func
        return shuffle_func(self, x)
    
    def set_seed(self, seed):
        """
        Set the random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        from ember_ml.backend.torch.tensor.ops.random import set_seed as set_seed_func
        return set_seed_func(self, seed)
    
    def get_seed(self):
        """
        Get the current random seed.
        
        Returns:
            Current random seed or None if not set
        """
        from ember_ml.backend.torch.tensor.ops.random import get_seed as get_seed_func
        return get_seed_func(self)