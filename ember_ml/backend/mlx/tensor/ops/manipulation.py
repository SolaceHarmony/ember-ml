"""MLX tensor manipulation operations."""

import mlx.core as mx
from typing import Optional,Union

from ember_ml.backend.mlx.tensor import MLXTensor
from ember_ml.backend.mlx.types import TensorLike,ShapeLike,Shape
Tensor = MLXTensor()

def reshape(tensor: TensorLike, shape: ShapeLike) -> mx.array:
    """
    Reshape an MLX array to a new shape.
    
    Args:
        tensor: Input array
        shape: New shape
        
    Returns:
        Reshaped MLX array
    """
    # Ensure shape is a sequence

    if isinstance(shape, int):
        shape = (shape,)
    return mx.reshape(Tensor.convert_to_tensor(tensor), shape)

def transpose(tensor: TensorLike, axes: Optional[Shape]=None):
    """
    Permute the dimensions of an MLX array.
    
    Args:
        tensor: Input array
        axes: Optional permutation of dimensions
        
    Returns:
        Transposed MLX array
    """
    tensor_array = Tensor.convert_to_tensor(tensor)

    if axes is None:
        # Default transpose behavior (swap last two dimensions)
        ndim = len(tensor_array.shape)
        if ndim <= 1:
            return tensor_array
        axes = list(range(ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1]

    return mx.transpose(tensor_array, axes)

def concatenate(tensors: list[TensorLike], axis: Optional[int]=0):
    """
    Concatenate MLX arrays along a specified axis.

    Args:
        tensors: Sequence of arrays
        axis: Axis along which to concatenate

    Returns:
        Concatenated MLX array
    """
    return mx.concatenate([Tensor.convert_to_tensor(arr) for arr in tensors], axis=axis)

def stack(tensors : list[TensorLike], axis: Optional[int]=0):
    """
    Stack MLX arrays along a new axis.

    Args:
        tensors: Sequence of arrays
        axis: Axis along which to stack

    Returns:
        Stacked MLX array
    """
    return mx.stack([Tensor.convert_to_tensor(arr) for arr in tensors], axis=axis)

def split(tensor : TensorLike, num_or_size_splits: Union[int,list[int]], axis=0) -> list[mx.array]:
    """
    Split an MLX array into sub-arrays.

    Args:
        tensor: Input array
        num_or_size_splits: Number of splits or sizes of each split
        axis: Axis along which to split

    Returns:
        List of sub-arrays
    """
    tensor_array = Tensor.convert_to_tensor(tensor)

    # MLX split returns an array or a tuple of arrays
    result = mx.split(tensor_array, indices_or_sections=num_or_size_splits, axis=axis)

    # Convert to list if it's not already a list
    if isinstance(result, list):
        return result
    elif isinstance(result, tuple):
        return list(result)
    else:
        # If it's a single array, return a list with that array
        return [result]

def expand_dims(tensor : TensorLike, axis: ShapeLike) -> mx.array:
    """
    Insert new axes into an MLX array's shape.

    Args:
        tensor: Input array
        axis: Position(s) where new axes should be inserted

    Returns:
        MLX array with expanded dimensions
    """
    tensor_array = Tensor.convert_to_tensor(tensor)

    if isinstance(axis, (list, tuple)):
        for ax in sorted(axis):
            tensor_array = mx.expand_dims(tensor_array, ax)
        return tensor_array

    return mx.expand_dims(tensor_array, axis)

def squeeze(tensor: TensorLike, axis : Union[None,ShapeLike]=None):
    """
    Remove single-dimensional entries from an MLX array's shape.

    Args:
        tensor: Input array
        axis: Position(s) where dimensions should be removed

    Returns:
        MLX array with squeezed dimensions
    """
    tensor_array = Tensor.convert_to_tensor(tensor)

    if axis is None:
        return mx.squeeze(tensor_array)

    return mx.squeeze(tensor_array, axis)

def tile(tensor : TensorLike, reps : ShapeLike) -> mx.array:
    """
    Construct an MLX array by tiling a given array.
    
    Args:
        tensor_obj: MLXTensor instance
        tensor: Input array
        reps: Number of repetitions for each dimension
        
    Returns:
        Tiled MLX array
    """
    tensor_array = Tensor.convert_to_tensor(tensor)
    return mx.tile(tensor_array, reps)

def pad(tensor : TensorLike, paddings, constant_values=0):
    """
    Pad a tensor with a constant value.

    Args:
        tensor_obj: MLXTensor instance
        tensor: Input tensor
        paddings: Sequence of sequences of integers specifying the padding for each dimension
                Each inner sequence should contain two integers: [pad_before, pad_after]
        constant_values: Value to pad with

    Returns:
        Padded tensor
    """
    tensor_array = Tensor.convert_to_tensor(tensor)

    # Convert paddings to the format expected by mx.pad
    # MLX expects a tuple of (pad_before, pad_after) for each dimension
    pad_width = tuple(tuple(p) for p in paddings)

    # Pad the tensor
    return mx.pad(tensor_array, pad_width, constant_values)