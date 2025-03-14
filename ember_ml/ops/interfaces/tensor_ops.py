"""
Tensor operations interface.

This module defines the abstract interface for tensor operations.
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Optional, Tuple, Any, List

# Type aliases
Shape = Union[int, Sequence[int]]
DType = Any  # Each backend will define its own dtype

class TensorOps(ABC):
    """Abstract interface for tensor operations."""
    
    @abstractmethod
    def zeros(self, shape: Shape, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor of zeros.
        
        Args:
            shape: Shape of the tensor
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor of zeros with the specified shape
        """
        pass
    
    @abstractmethod
    def ones(self, shape: Shape, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor of ones.
        
        Args:
            shape: Shape of the tensor
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor of ones with the specified shape
        """
        pass
    
    @abstractmethod
    def zeros_like(self, x: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor of zeros with the same shape as the input.
        
        Args:
            x: Input tensor
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor of zeros with the same shape as x
        """
        pass
    
    @abstractmethod
    def ones_like(self, x: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor of ones with the same shape as the input.
        
        Args:
            x: Input tensor
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor of ones with the same shape as x
        """
        pass
    
    @abstractmethod
    def eye(self, n: int, m: Optional[int] = None, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
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
        pass
    
    @abstractmethod
    def arange(self, start: int, stop: Optional[int] = None, step: int = 1, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
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
        pass
    
    @abstractmethod
    def linspace(self, start: float, stop: float, num: int, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
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
        pass
    
    @abstractmethod
    def full(self, shape: Shape, fill_value: Union[float, int], dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
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
        pass
    
    @abstractmethod
    def full_like(self, x: Any, fill_value: Union[float, int], dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
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
        pass
    
    @abstractmethod
    def reshape(self, x: Any, shape: Shape) -> Any:
        """
        Reshape a tensor to a new shape.
        
        Args:
            x: Input tensor
            shape: New shape
            
        Returns:
            Reshaped tensor
        """
        pass
    
    @abstractmethod
    def transpose(self, x: Any, axes: Optional[Sequence[int]] = None) -> Any:
        """
        Permute the dimensions of a tensor.
        
        Args:
            x: Input tensor
            axes: Optional permutation of dimensions
            
        Returns:
            Transposed tensor
        """
        pass
    
    @abstractmethod
    def concatenate(self, tensors: Sequence[Any], axis: int = 0) -> Any:
        """
        Concatenate tensors along a specified axis.
        
        Args:
            tensors: Sequence of tensors
            axis: Axis along which to concatenate
            
        Returns:
            Concatenated tensor
        """
        pass
    
    @abstractmethod
    def stack(self, tensors: Sequence[Any], axis: int = 0) -> Any:
        """
        Stack tensors along a new axis.
        
        Args:
            tensors: Sequence of tensors
            axis: Axis along which to stack
            
        Returns:
            Stacked tensor
        """
        pass
    
    @abstractmethod
    def split(self, x: Any, num_or_size_splits: Union[int, Sequence[int]], axis: int = 0) -> List[Any]:
        """
        Split a tensor into sub-tensors.
        
        Args:
            x: Input tensor
            num_or_size_splits: Number of splits or sizes of each split
            axis: Axis along which to split
            
        Returns:
            List of sub-tensors
        """
        pass
    
    @abstractmethod
    def expand_dims(self, x: Any, axis: Union[int, Sequence[int]]) -> Any:
        """
        Insert new axes into a tensor's shape.
        
        Args:
            x: Input tensor
            axis: Position(s) where new axes should be inserted
            
        Returns:
            Tensor with expanded dimensions
        """
        pass
    
    @abstractmethod
    def squeeze(self, x: Any, axis: Optional[Union[int, Sequence[int]]] = None) -> Any:
        """
        Remove single-dimensional entries from a tensor's shape.
        
        Args:
            x: Input tensor
            axis: Position(s) where dimensions should be removed
            
        Returns:
            Tensor with squeezed dimensions
        """
        pass
    
    @abstractmethod
    def tile(self, x: Any, reps: Sequence[int]) -> Any:
        """
        Construct a tensor by tiling a given tensor.
        
        Args:
            x: Input tensor
            reps: Number of repetitions along each dimension
            
        Returns:
            Tiled tensor
        """
        pass
    
    @abstractmethod
    def gather(self, x: Any, indices: Any, axis: int = 0) -> Any:
        """
        Gather slices from a tensor along an axis.
        
        Args:
            x: Input tensor
            indices: Indices of slices to gather
            axis: Axis along which to gather
            
        Returns:
            Gathered tensor
        """
        pass
    
    @abstractmethod
    def convert_to_tensor(self, x: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
        """
        Convert input to a tensor.
        
        Args:
            x: Input data (array, tensor, scalar)
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor representation of the input
        """
        pass
    
    @abstractmethod
    def shape(self, x: Any) -> Tuple[int, ...]:
        """
        Get the shape of a tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Shape of the tensor
        """
        pass
    
    @abstractmethod
    def dtype(self, x: Any) -> DType:
        """
        Get the data type of a tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Data type of the tensor
        """
        pass
    
    @abstractmethod
    def cast(self, x: Any, dtype: DType) -> Any:
        """
        Cast a tensor to a different data type.
        
        Args:
            x: Input tensor
            dtype: Target data type
            
        Returns:
            Tensor with the target data type
        """
        pass
    
    @abstractmethod
    def copy(self, x: Any) -> Any:
        """
        Create a copy of a tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Copy of the tensor
        """
        pass
    
    @abstractmethod
    def var(self, x: Any, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Any:
        """
        Compute the variance of a tensor along specified axes.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the variance
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Variance of the tensor
        """
        pass
    
    @abstractmethod
    def sort(self, x: Any, axis: int = -1, descending: bool = False) -> Any:
        """
        Sort a tensor along a specified axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to sort
            descending: Whether to sort in descending order
            
        Returns:
            Sorted tensor
        """
        pass
    
    @abstractmethod
    def argsort(self, x: Any, axis: int = -1, descending: bool = False) -> Any:
        """
        Return the indices that would sort a tensor along a specified axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to sort
            descending: Whether to sort in descending order
            
        Returns:
            Indices that would sort the tensor
        """
        pass
    
    @abstractmethod
    def item(self, x: Any) -> Union[int, float, bool]:
        """
        Extract the scalar value from a tensor.
        
        This method extracts the scalar value from a tensor containing a single element.
        It works across all backends (NumPy, PyTorch, MLX) and returns a standard Python scalar.
        
        Args:
            x: Input tensor containing a single element
            
        Returns:
            Standard Python scalar (int, float, or bool)
        """
        pass
    
    @abstractmethod
    def slice(self, x: Any, starts: Sequence[int], sizes: Sequence[int]) -> Any:
        """
        Extract a slice from a tensor.
        
        Args:
            x: Input tensor
            starts: Starting indices for each dimension
            sizes: Size of the slice in each dimension. A value of -1 means "all remaining elements in this dimension"
            
        Returns:
            Sliced tensor
        """
        pass
    
    @abstractmethod
    def slice_update(self, x: Any, slices: Union[List, Tuple], updates: Any) -> Any:
        """
        Update a tensor at specific indices.
        
        Args:
            x: Input tensor to update
            slices: List or tuple of slice objects or indices
            updates: Values to insert at the specified indices
            
        Returns:
            Updated tensor
        """
        pass
    
    @abstractmethod
    def pad(self, x: Any, paddings: Sequence[Sequence[int]], constant_values: Union[int, float] = 0) -> Any:
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
        pass

    @abstractmethod
    def tensor_scatter_nd_update(self, tensor: Any, indices: Any, updates: Any) -> Any:
        """
        Updates values of a tensor at specified indices.

        Args:
            tensor: Input tensor to update
            indices: Indices at which to update values (N-dimensional indices)
            updates: Values to insert at the specified indices

        Returns:
            Updated tensor
        """
        pass

    @abstractmethod
    def to_numpy(self, x: Any) -> Any:
        """
        Convert a tensor to a numpy array.

        Args:
            x: Input tensor

        Returns:
            NumPy array containing the tensor data
        """
        pass