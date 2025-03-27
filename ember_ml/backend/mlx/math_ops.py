"""
MLX math operations for ember_ml.

This module provides MLX implementations of math operations.
"""

import mlx.core as mx
from typing import Union, Optional, List, Tuple, Literal
from ember_ml.backend.mlx.tensor.ops import cast
from ember_ml.backend.mlx.tensor import MLXDType
from ember_ml.backend.mlx.types import TensorLike, ShapeLike
# We avoid creating global instances to prevent circular imports
# Each function will create its own instances when needed

def gather(x: TensorLike, indices: TensorLike, axis: int = 0) -> mx.array:
    """
    Gather slices from x along the specified axis according to indices.
    
    Args:
        x: The input array from which to gather values
        indices: The indices of the values to extract
        axis: The axis along which to index (default: 0)
        
    Returns:
        Array of gathered values with the same type as x
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    x_array = tensor_ops.convert_to_tensor(x)
    indices_array = tensor_ops.convert_to_tensor(indices)
    
    # Ensure indices are integer type
    # Check if the dtype contains 'int' in its name
    if not 'int' in str(indices_array.dtype).lower():
        dtype_ops = MLXDType()
        indices_array = cast(indices_array, dtype_ops.int32)
    
    # In MLX, we can use take with the axis parameter
    return mx.take(x_array, indices_array, axis=axis)
dtype_obj = MLXDType()

def add(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Add two MLX arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Element-wise sum
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    return mx.add(tensor_ops.convert_to_tensor(x), tensor_ops.convert_to_tensor(y))


def subtract(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Subtract two MLX arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Element-wise difference
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    return mx.subtract(tensor_ops.convert_to_tensor(x), tensor_ops.convert_to_tensor(y))


def multiply(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Multiply two MLX arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Element-wise product
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    x_tensor = tensor_ops.convert_to_tensor(x)
    y_tensor = tensor_ops.convert_to_tensor(y)
    
    # Handle empty arrays
    if x_tensor.size == 0 or y_tensor.size == 0:
        # If either array is empty, return an empty array with appropriate shape
        if x_tensor.size == 0:
            return x_tensor  # Return the empty array
        else:
            return mx.zeros((0,), dtype=y_tensor.dtype)
    
    return mx.multiply(x_tensor, y_tensor)


def divide(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Divide two MLX arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Element-wise quotient
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    return mx.divide(tensor_ops.convert_to_tensor(x), tensor_ops.convert_to_tensor(y))


def dot(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Compute the dot product of two MLX arrays.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Dot product
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    x_array = tensor_ops.convert_to_tensor(x)
    y_array = tensor_ops.convert_to_tensor(y)
    
    # Handle different dimensions
    if mx.equal(mx.array(len(x_array.shape)), mx.array(1)) and mx.equal(mx.array(len(y_array.shape)), mx.array(1)):
        return mx.sum(mx.multiply(x_array, y_array))
    else:
        return mx.matmul(x_array, y_array)


def matmul(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Compute the matrix product of two MLX arrays.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Matrix product
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    return mx.matmul(tensor_ops.convert_to_tensor(x), tensor_ops.convert_to_tensor(y))

# Note: MLX does not support broadcasting for mean, sum, var, min, and max
def mean(x: TensorLike, axis: Optional[ShapeLike] = None, keepdims: bool = False) -> mx.array:
    """
    Compute the mean of an MLX array along specified axes.
    
    Args:
        x: Input array
        axis: Axis or axes along which to compute the mean
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Mean of the array
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    x_array = MLXTensor().convert_to_tensor(x)
    
    if axis is None:
        return mx.mean(x_array, keepdims=keepdims)
    if isinstance(axis, (list, tuple)):
        # MLX doesn't support multiple axes directly, so we need to do it sequentially
        result = x_array
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = mx.mean(result, axis=ax, keepdims=keepdims)
        return result
    return mx.mean(x_array, axis=axis, keepdims=keepdims)


def sum(x: TensorLike, axis: Optional[ShapeLike] = None, keepdims: bool = False) -> mx.array:
    """
    Compute the sum of an MLX array along specified axes.
    
    Args:
        x: Input array
        axis: Axis or axes along which to compute the sum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Sum of the array
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()    
    x_array = tensor_ops.convert_to_tensor(x)
    
    if axis is None:
        return mx.sum(x_array, keepdims=keepdims)
    if isinstance(axis, (list, tuple)):
        # MLX doesn't support multiple axes directly, so we need to do it sequentially
        result = x_array
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = mx.sum(result, axis=ax, keepdims=keepdims)
        return result
    else:
        return mx.sum(x_array, axis=axis, keepdims=keepdims)

def var(x: TensorLike, axis: Optional[ShapeLike] = None, keepdims: bool = False) -> mx.array:
    """
    Compute the variance of an MLX array along specified axes.
    
    Args:
        x: Input array
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Variance of the array
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    x_array = MLXTensor().convert_to_tensor(x)
    
    if axis is None:
        return mx.var(x_array, keepdims=keepdims)
    
    if isinstance(axis, (list, tuple)):
        # MLX doesn't support multiple axes directly, so we need to do it sequentially
        result = x_array
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = mx.var(result, axis=ax, keepdims=keepdims)
        return result
    else:
        return mx.var(x_array, axis=axis, keepdims=keepdims)


def exp(x: TensorLike) -> mx.array:
    """
    Compute the exponential of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise exponential
    """
    # Create instances for each call to avoid circular imports 
    from ember_ml.backend.mlx.tensor import MLXTensor   
    return mx.exp(MLXTensor().convert_to_tensor(x))


def log(x: TensorLike) -> mx.array:
    """
    Compute the natural logarithm of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise logarithm
    """
    # Create instances for each call to avoid circular imports    
    from ember_ml.backend.mlx.tensor import MLXTensor   
    return mx.log(MLXTensor().convert_to_tensor(x))

# Note: MLX does not support broadcasting for log1p
def pow(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Compute x raised to the power of y element-wise.
    
    Args:
        x: Base array
        y: Exponent array
        
    Returns:
        Element-wise power
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor   
    tensor_ops = MLXTensor()
    return mx.power(tensor_ops.convert_to_tensor(x), tensor_ops.convert_to_tensor(y))


def sqrt(x: TensorLike) -> mx.array:
    """
    Compute the square root of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise square root
    """
    from ember_ml.backend.mlx.tensor import MLXTensor   
    return mx.sqrt(MLXTensor().convert_to_tensor(x))

def clip(x: TensorLike,
         min_val: TensorLike,
         max_val: TensorLike) -> mx.array:
    """
    Clip the values of an MLX array to a specified range.
    
    Args:
        x: Input array
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clipped array
    """
    from ember_ml.backend.mlx.tensor import MLXTensor   
    Tensor = MLXTensor()
    x_array = Tensor.convert_to_tensor(x)
    min_val = Tensor.convert_to_tensor(min_val)
    max_val = Tensor.convert_to_tensor(max_val)
    
    return mx.clip(x_array, min_val, max_val)


def sigmoid(x: mx.array) -> mx.array:
    """
    Compute sigmoid function on all elements.
    
    The sigmoid function is defined as:
    sigmoid(x) = 1 / (1 + exp(-x))
    
    It transforms each element of input tensor to a value between 0 and 1.
    
    Args:
        x: Input array
        
    Returns:
        Tensor with sigmoid applied to each element
    """
    from ember_ml.backend.mlx.tensor import MLXTensor  # inserted missing import
    x_array = MLXTensor().convert_to_tensor(x)
    x_safe = clip(x_array, -88.0, 88.0)  # Prevent overflow
    return mx.sigmoid(x_safe)


def softplus(x: TensorLike) -> mx.array:
    """
    Compute the softplus of an MLX array element-wise.
    
    The softplus function is defined as log(1 + exp(x)).
    
    Args:
        x: Input array
        
    Returns:
        Element-wise softplus
    """
    from ember_ml.backend.mlx.tensor import MLXTensor   
    x_array = MLXTensor().convert_to_tensor(x)
    x_safe = clip(x_array, -88.0, 88.0)  # Prevent overflow
    # softplus(x) = log(1 + exp(x))
    return mx.log(mx.add(1.0, mx.exp(x_safe)))


def tanh(x: TensorLike) -> mx.array:
    """
    Compute the hyperbolic tangent of an MLX array element-wise.

    Args:
        x: Input array
        
    Returns:
        Element-wise tanh
    """
    from ember_ml.backend.mlx.tensor import MLXTensor  # inserted missing import
    return mx.tanh(MLXTensor().convert_to_tensor(x))


def relu(x: TensorLike) -> mx.array:
    """
    Apply Rectified Linear Unit function element-wise.
    
    ReLU(x) = max(0, x)
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with ReLU applied to each element
    """
    from ember_ml.backend.mlx.tensor import MLXTensor   
    return mx.maximum(0, MLXTensor().convert_to_tensor(x))

def abs(x: TensorLike) -> mx.array:
    """
    Compute the absolute value of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise absolute value
    """
    from ember_ml.backend.mlx.tensor import MLXTensor   
    return mx.abs(MLXTensor().convert_to_tensor(x))


def negative(x: TensorLike) -> mx.array:
    """
    Compute the negative of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with negated values
    """
    from ember_ml.backend.mlx.tensor import MLXTensor   
    return mx.negative(MLXTensor().convert_to_tensor(x))


def sign(x: TensorLike) -> mx.array:
    """
    Compute the sign of tensor elements.
    
    Returns -1 for negative values, 0 for zero, and 1 for positive values.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with sign values
    """
    from ember_ml.backend.mlx.tensor import MLXTensor   
    x_tensor = MLXTensor().convert_to_tensor(x)
    # Compute sign using comparisons
    positive = mx.array(1.0, dtype=x_tensor.dtype)
    negative = mx.array(-1.0, dtype=x_tensor.dtype)
    zero = mx.array(0.0, dtype=x_tensor.dtype)
    
    # x > 0 -> 1, x < 0 -> -1, else 0
    return mx.where(x_tensor > 0, positive, mx.where(x_tensor < 0, negative, zero))


def argmax(x: TensorLike, axis: Optional[int] = None, keepdims: bool = False) -> mx.array:
    """
    Return the indices of the maximum values along the specified axis.
    
    Args:
        x: Input array
        axis: Axis along which to find maximum values. If None, the argmax of
            the flattened array is returned.
        keepdims: If True, the reduced axes are kept as dimensions with size one.
            
    Returns:
        Indices of the maximum values along the specified axis.
    """
    from ember_ml.backend.mlx.tensor import MLXTensor   
    x_tensor = MLXTensor().convert_to_tensor(x)
    return mx.argmax(x_tensor, axis=axis, keepdims=keepdims)


def sin(x: TensorLike) -> mx.array:
    """
    Compute the sine of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise sine
    """
    from ember_ml.backend.mlx.tensor import MLXTensor   
    return mx.sin(MLXTensor().convert_to_tensor(x))


def cos(x: TensorLike) -> mx.array:
    """Compute the cosine of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise cosine
    """
    from ember_ml.backend.mlx.tensor import MLXTensor  # inserted missing import
    return mx.cos(MLXTensor().convert_to_tensor(x))


def tan(x: TensorLike) -> mx.array:
    """
    Compute the tangent of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise tangent
    """
    from ember_ml.backend.mlx.tensor import MLXTensor   
    x_tensor = MLXTensor().convert_to_tensor(x)
    # tan(x) = sin(x) / cos(x)
    return mx.divide(mx.sin(x_tensor), mx.cos(x_tensor)) if mx.cos(x_tensor) != 0 else mx.array(float('inf'))  # Avoid division by zero


def sinh(x: TensorLike) -> mx.array:
    """
    Compute the hyperbolic sine of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise hyperbolic sine
    """
    from ember_ml.backend.mlx.tensor import MLXTensor   
    x_tensor = MLXTensor().convert_to_tensor(x)
    # sinh(x) = (exp(x) - exp(-x)) / 2
    return mx.divide(mx.subtract(mx.exp(x_tensor), mx.exp(mx.negative(x_tensor))), 2.0)


def cosh(x: TensorLike) -> mx.array:
    """
    Compute the hyperbolic cosine of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise hyperbolic cosine
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    x_tensor = tensor_ops.convert_to_tensor(x)
    # cosh(x) = (exp(x) + exp(-x)) / 2
    return mx.divide(mx.add(mx.exp(x_tensor), mx.exp(mx.negative(x_tensor))), 2.0)


def log10(x: TensorLike) -> mx.array:
    """
    Compute the base-10 logarithm of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise base-10 logarithm
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    x_tensor = tensor_ops.convert_to_tensor(x)
    # log10(x) = log(x) / log(10)
    return mx.divide(mx.log(x_tensor), mx.log(mx.array(10.0)))


def log2(x: TensorLike) -> mx.array:
    """
    Compute the base-2 logarithm of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise base-2 logarithm
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    x_tensor = tensor_ops.convert_to_tensor(x)
    # log2(x) = log(x) / log(2)
    return mx.divide(mx.log(x_tensor), mx.log(mx.array(2.0)))


def square(x: TensorLike) -> mx.array:
    """
    Compute the square of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise square
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    return mx.square(MLXTensor().convert_to_tensor(x))


def mod(x: TensorLike, y: TensorLike) -> mx.array:
    """Compute the remainder of division of x by y element-wise.
    
    Args:
        x: Input array (dividend)
        y: Input array (divisor)
        
    Returns:
        Element-wise remainder
    """
    from ember_ml.backend.mlx.tensor import MLXTensor  # inserted missing import
    Tensor = MLXTensor()
    x_tensor = Tensor.convert_to_tensor(x)
    y_tensor = Tensor.convert_to_tensor(y)
    _, remainder = mx.divmod(x_tensor, y_tensor)
    return remainder


def floor_divide(x: TensorLike, y: TensorLike) -> mx.array:
    """Element-wise integer division.
    
    If either array is a floating point type then it is equivalent to calling floor() after divide().
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Element-wise integer quotient (a // b)
    """
    from ember_ml.backend.mlx.tensor import MLXTensor  # inserted missing import
    Tensor = MLXTensor()

    x_tensor = Tensor.convert_to_tensor(x)
    y_tensor = Tensor.convert_to_tensor(y)
    
    # Use floor_divide from MLX
    return mx.floor_divide(x_tensor, y_tensor)


def min(x: TensorLike, axis: Optional[ShapeLike] = None, keepdims: bool = False) -> mx.array:
    """
    Compute the minimum value of an MLX array along specified axes.
    
    Args:
        x: Input array
        axis: Axis or axes along which to compute the minimum

        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Minimum value of the array
    """
    from ember_ml.backend.mlx.tensor import MLXTensor  # inserted missing import
    x_array = MLXTensor().convert_to_tensor(x)
    
    if axis is None:
        # Find the minimum of the flattened array
        return mx.min(x_array)
    return mx.min(x_array, axis=axis, keepdims=keepdims)


def max(x: TensorLike, axis: Optional[ShapeLike] = None, keepdims: bool = False) -> mx.array:
    """
    Compute the maximum value of an MLX array along specified axes.
    
    Args:
        x: Input tensor
        axis: Axis along which to find maximum.
            If None, find maximum over all elements.
        keepdims: Whether the output tensor has dim retained or not.
        
    Returns:
        Maximum value of the array
    """
    from ember_ml.backend.mlx.tensor import MLXTensor  # inserted missing import
    x_array = MLXTensor().convert_to_tensor(x)
    
    if axis is None:
        return mx.max(x_array)
    return mx.max(x_array, axis=axis, keepdims=keepdims)

def softmax(x: TensorLike, axis: int = -1) -> mx.array:
    """Compute the softmax of an MLX array along a specified axis.
    
    Args:
        x: Input array
        axis: Axis along which to compute the softmax
        
    Returns:
        Softmax of the array
    """
    from ember_ml.backend.mlx.tensor import MLXTensor  # inserted missing import
    x_array = MLXTensor().convert_to_tensor(x)
    x_max = mx.max(x_array, axis=axis, keepdims=True)
    exp_x = mx.exp(mx.subtract(x_array, x_max))
    return mx.divide(exp_x, mx.sum(exp_x, axis=axis, keepdims=True))


def sort(x: TensorLike, axis: int = -1) -> mx.array:
    """
    Sort an MLX array along a specified axis.
    
    Args:
        x: Input array
        axis: Axis along which to sort
        
    Returns:
        Sorted array
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    x_array = tensor_ops.convert_to_tensor(x)
    return mx.sort(x_array, axis=axis)


def gradient(f: TensorLike, 
             *varargs, axis: Optional[ShapeLike] = None,
            edge_order: Literal[1, 2] = 1) -> Union[mx.array, List[mx.array]]:
    """
    Return the gradient of an N-dimensional array.
    
    The gradient is computed using second order accurate central differences in the interior
    points and either first or second order accurate one-sides (forward or backwards)
    differences at the boundaries. The returned gradient hence has the same shape as the input array.
    
    Args:
        f: An N-dimensional array containing samples of a scalar function.
        *varargs: Spacing between f values. Default unitary spacing for all dimensions.
        axis: Gradient is calculated only along the given axis or axes.
            The default (axis = None) is to calculate the gradient for all the axes of the input array.
        edge_order: Gradient is calculated using N-th order accurate differences at the boundaries.
            Must be 1 or 2.
            
    Returns:
        A tensor or tuple of tensors corresponding to the derivatives of f with respect to each dimension.
        Each derivative has the same shape as f.
    """
    from ember_ml.backend.mlx.tensor.ops import scatter
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    f_array = tensor_ops.convert_to_tensor(f)
    
    # Get the shape of the input array
    f_shape = f_array.shape
    ndim = len(f_shape)
    
    # Process spacing arguments
    if len(varargs) == 0:
# Default spacing is 1 for all dimensions
        dx = [mx.array(1.0) for _ in range(ndim)]
    else:
        dx = [tensor_ops.convert_to_tensor(arg) for arg in varargs]
    
    # Determine which axes to calculate gradient along
    axes = list(range(ndim)) if axis is None else ([axis] if isinstance(axis, int) else list(axis))
    
# Calculate gradient for each axis
    grad = []
    for i in axes:
        # Create slices for forward and backward differences
        slice_prev = [slice(None)] * ndim
        slice_next = [slice(None)] * ndim
        
        # Calculate gradient along axis i
        if edge_order == 1:
            # First-order accurate differences at the boundaries
            # Forward difference at the beginning
            slice_prev[i] = slice(0, 1)
            slice_next[i] = slice(1, 2)
            begin_indices = [0]  # Index for the beginning as a Python list
            begin = mx.divide(mx.subtract(f_array[tuple(slice_next)], f_array[tuple(slice_prev)]), dx[i])
            
            # Central difference in the interior
            slice_prev[i] = slice(0, -2)
            slice_next[i] = slice(2, None)
            middle_size = f_shape[i] - 2  # Size of the interior
            middle_indices = list(range(1, middle_size + 1))  # Indices for the interior as a Python list
            middle = mx.divide(mx.subtract(f_array[tuple(slice_next)], f_array[tuple(slice_prev)]),
                              mx.multiply(mx.array(2.0), dx[i]))
            
            # Backward difference at the end
            slice_prev[i] = slice(-2, -1)
            slice_next[i] = slice(-1, None)
            end_indices = [f_shape[i] - 1]  # Index for the end as a Python list
            end = mx.divide(mx.subtract(f_array[tuple(slice_next)], f_array[tuple(slice_prev)]), dx[i])
            
            # Create gradient tensor using scatter operations
            grad_i = mx.zeros_like(f_array)
            
            # Create a flattened version of the gradient tensor
            grad_i_flat = mx.reshape(grad_i, (-1,))
            
            # Scatter the beginning values
            if begin is not None and len(begin_indices) > 0:
                begin_flat = mx.reshape(begin, (-1,))
                grad_i_flat = scatter(begin_flat, mx.array(begin_indices), grad_i_flat.shape[0], aggr="add")
                grad_i = mx.reshape(grad_i_flat, f_array.shape)
            
            # Scatter the middle values
            if middle is not None and len(middle_indices) > 0:
                middle_flat = mx.reshape(middle, (-1,))
                grad_i_flat = mx.reshape(grad_i, (-1,))
                grad_i_flat = scatter(middle_flat, mx.array(middle_indices), grad_i_flat.shape[0], aggr="add")
                grad_i = mx.reshape(grad_i_flat, f_array.shape)
            
            # Scatter the end values
            if end is not None and len(end_indices) > 0:
                end_flat = mx.reshape(end, (-1,))
                grad_i_flat = mx.reshape(grad_i, (-1,))
                grad_i_flat = scatter(end_flat, mx.array(end_indices), grad_i_flat.shape[0], aggr="add")
                grad_i = mx.reshape(grad_i_flat, f_array.shape)
            
        elif edge_order == 2:
            # Second-order accurate differences at the boundaries
            # TODO: Implement second-order accurate differences
            # For now, fall back to first-order accurate differences
            slice_prev[i] = slice(0, 1)
            slice_next[i] = slice(1, 2)
            begin_indices = [0]  # Index for the beginning as a Python list
            begin = mx.divide(mx.subtract(f_array[tuple(slice_next)], f_array[tuple(slice_prev)]), dx[i])
            
            # Central difference in the interior
            slice_prev[i] = slice(0, -2)
            slice_next[i] = slice(2, None)
            middle_size = f_shape[i] - 2  # Size of the interior
            middle_indices = list(range(1, middle_size + 1))  # Indices for the interior as a Python list
            middle = mx.divide(mx.subtract(f_array[tuple(slice_next)], f_array[tuple(slice_prev)]),
                              mx.multiply(mx.array(2.0), dx[i]))
            
            # Backward difference at the end
            slice_prev[i] = slice(-2, -1)
            slice_next[i] = slice(-1, None)
            end_indices = [f_shape[i] - 1]  # Index for the end as a Python list
            end = mx.divide(mx.subtract(f_array[tuple(slice_next)], f_array[tuple(slice_prev)]), dx[i])
            
            # Create gradient tensor using scatter operations
            grad_i = mx.zeros_like(f_array)
            
            # Create a flattened version of the gradient tensor
            grad_i_flat = mx.reshape(grad_i, (-1,))
            
            # Scatter the beginning values
            if begin is not None and len(begin_indices) > 0:
                begin_flat = mx.reshape(begin, (-1,))
                grad_i_flat = scatter(begin_flat, mx.array(begin_indices), grad_i_flat.shape[0], aggr="add")
                grad_i = mx.reshape(grad_i_flat, f_array.shape)
            
            # Scatter the middle values
            if middle is not None and len(middle_indices) > 0:
                middle_flat = mx.reshape(middle, (-1,))
                grad_i_flat = mx.reshape(grad_i, (-1,))
                grad_i_flat = scatter(middle_flat, mx.array(middle_indices), grad_i_flat.shape[0], aggr="add")
                grad_i = mx.reshape(grad_i_flat, f_array.shape)
            
            # Scatter the end values
            if end is not None and len(end_indices) > 0:
                end_flat = mx.reshape(end, (-1,))
                grad_i_flat = mx.reshape(grad_i, (-1,))
                grad_i_flat = scatter(end_flat, mx.array(end_indices), grad_i_flat.shape[0], aggr="add")
                grad_i = mx.reshape(grad_i_flat, f_array.shape)
        
        grad.append(grad_i)
    
    # Return the gradient
    if len(grad) == 1:
        return grad[0]
    else:
        return grad

def cumsum(x: mx.array, axis: Optional[int] = None) -> mx.array:
    """
    Compute the cumulative sum of an MLX array along a specified axis.
    
    Args:
        x: Input array
        axis: Axis along which to compute the cumulative sum
        
    Returns:
        Array with cumulative sums
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    x_array = tensor_ops.convert_to_tensor(x)
    return mx.cumsum(x_array, axis=axis)


def eigh(a: mx.array) -> Tuple[mx.array, mx.array]:
    """
    Compute the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.
    
    Args:
        a: Input Hermitian or symmetric matrix
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    a_array = tensor_ops.convert_to_tensor(a)
    
    # Ensure the input is a 2D matrix
    if len(a_array.shape) != 2 or a_array.shape[0] != a_array.shape[1]:
        raise ValueError("Input must be a square matrix.")
    

    eigenvalues, eigenvectors = mx.linalg.eig(a_array)
    return eigenvalues, eigenvectors

# Define the pi constant using Chudnovsky algorithm
def _calculate_pi_value(precision_digits=15):
    """
    Calculate pi using the Chudnovsky algorithm.
    
    The Chudnovsky algorithm is one of the most efficient algorithms for calculating π,
    with a time complexity of O(n log(n)^3). It converges much faster than other series.
    
    Formula:
    1/π = (12/426880√10005) * Σ (6k)!(13591409 + 545140134k) / ((3k)!(k!)^3 * (-640320)^(3k))
    
    Args:
        precision_digits: Number of decimal places to calculate
        
    Returns:
        Value of pi with the specified precision
    """
    # Constants in the Chudnovsky algorithm
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    C = mx.array(640320)
    C3_OVER_24 = mx.divide(mx.power(C, 3), mx.array(24))
    DIGITS_PER_TERM = mx.array(14.1816474627254776555)  # Approx. digits per iteration
    
    def binary_split(a, b):
        """Recursive binary split for the Chudnovsky algorithm."""
        from ember_ml.backend.mlx.tensor import MLXTensor
        inner_tensor = MLXTensor()
        a_tensor = inner_tensor.convert_to_tensor(a)
        b_tensor = inner_tensor.convert_to_tensor(b)
        diff = mx.subtract(b_tensor, a_tensor)
        
        if mx.equal(diff, mx.array(1)):
            # Base case
            if mx.equal(a_tensor, mx.array(0)):
                Pab = mx.array(1)
                Qab = mx.array(1)
            else:
                term1 = mx.subtract(mx.multiply(mx.array(6), a_tensor), mx.array(5))
                term2 = mx.subtract(mx.multiply(mx.array(2), a_tensor), mx.array(1))
                term3 = mx.subtract(mx.multiply(mx.array(6), a_tensor), mx.array(1))
                Pab = mx.multiply(mx.multiply(term1, term2), term3)
                Qab = mx.multiply(mx.power(a_tensor, 3), C3_OVER_24)
            
            base_term = mx.array(13591409)
            multiplier = mx.array(545140134)
            term = mx.add(base_term, mx.multiply(multiplier, a_tensor))
            Tab = mx.multiply(Pab, term)
            
            # Check if a is odd using divmod
            _, remainder = mx.divmod(a_tensor, mx.array(2))
            is_odd = mx.equal(remainder, mx.array(1))
            
            # If a is odd, negate Tab
            Tab = mx.where(is_odd, mx.negative(Tab), Tab)
            
            return Pab, Qab, Tab
        
        # Recursive case
        m = mx.divide(mx.add(a_tensor, b_tensor), mx.array(2))
        m = mx.floor(m)  # Ensure m is an integer
        
        Pam, Qam, Tam = binary_split(a, m)
        Pmb, Qmb, Tmb = binary_split(m, b)
        
        Pab = mx.multiply(Pam, Pmb)
        Qab = mx.multiply(Qam, Qmb)
        term1 = mx.multiply(Qmb, Tam)
        term2 = mx.multiply(Pam, Tmb)
        Tab = mx.add(term1, term2)
        
        return Pab, Qab, Tab
    
    # Number of terms needed for the desired precision
    precision_tensor = tensor_ops.convert_to_tensor(precision_digits)
    terms_float = mx.divide(precision_tensor, DIGITS_PER_TERM)
    terms_float = mx.add(terms_float, mx.array(1))
    terms = mx.floor(terms_float)  # Convert to integer
    terms_int = cast(terms, mx.int32)
    
    # Compute the binary split
    P, Q, T = binary_split(0, terms_int)
    
    # Calculate pi
    sqrt_10005 = mx.sqrt(mx.array(10005))
    numerator = mx.multiply(Q, mx.array(426880))
    numerator = mx.multiply(numerator, sqrt_10005)
    pi_approx = mx.divide(numerator, T)
    
    # Return as MLX array with shape (1,)
    return mx.reshape(pi_approx, (1,))

# Calculate pi with appropriate precision for MLX (float32)
# Ensure it's a scalar with shape (1,) as per MLX conventions
PI_CONSTANT = _calculate_pi_value(15)  # Increased precision to match reference value

pi : mx.array = mx.array([PI_CONSTANT], dtype=mx.float32)



def binary_split(a: TensorLike, b: TensorLike) -> Tuple[mx.array, mx.array]:
    """
    Recursive binary split for the Chudnovsky algorithm.
    
    This is used in the implementation of PI calculation for the PyTorch backend.
    
    Args:
        a: Start value
        b: End value
        
    Returns:
        Tuple of intermediate values for PI calculation
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    a_tensor = tensor_ops.convert_to_tensor(a)
    b_tensor = tensor_ops.convert_to_tensor(b)
    
    # Use mx operations
    diff = mx.subtract(b_tensor, a_tensor)
    
    if mx.equal(diff, mx.array(1)):
        # Base case
        if mx.equal(a_tensor, mx.array(0)):
            Pab = mx.array(1)
            Qab = mx.array(1)
        else:
            # Calculate terms using mx operations
            term1 = mx.subtract(mx.multiply(mx.array(6), a_tensor), mx.array(5))
            term2 = mx.subtract(mx.multiply(mx.array(2), a_tensor), mx.array(1))
            term3 = mx.subtract(mx.multiply(mx.array(6), a_tensor), mx.array(1))
            Pab = mx.multiply(mx.multiply(term1, term2), term3)
            
            # Define C3_OVER_24
            C = mx.array(640320.0)
            C3_OVER_24 = mx.divide(mx.power(C, mx.array(3.0)), mx.array(24.0))
            
            Qab = mx.multiply(mx.power(a_tensor, mx.array(3.0)), C3_OVER_24)
        
        return Pab, Qab
    else:
        # Recursive case
        m = mx.divide(mx.add(a_tensor, b_tensor), mx.array(2.0))
        Pam, Qam = binary_split(a_tensor, m)
        Pmb, Qmb = binary_split(m, b_tensor)
        
        Pab = mx.multiply(Pam, Pmb)
        Qab = mx.add(mx.multiply(Qam, Pmb), mx.multiply(Pam, Qmb))
        
        return Pab, Qab

class MLXMathOps:
    """MLX implementation of math operations."""
    
    def gather(self, x, indices, axis=0):
        """Gather slices from array according to indices."""
        return gather(x, indices, axis)
    
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
    
    def var(self, x, axis=None, keepdims=False):
        """Compute the variance of a tensor along specified axes."""
        return var(x, axis=axis, keepdims=keepdims)
    
    def exp(self, x):
        """Compute the exponential of a tensor element-wise."""
        return exp(x)
    
    def log(self, x):
        """Compute the natural logarithm of a tensor element-wise."""
        return log(x)
    
    def pow(self, x, y):
        """Compute x raised to the power of y element-wise."""
        return pow(x, y)
    
    def sqrt(self, x):
        """Compute the square root of a tensor element-wise."""
        return sqrt(x)
    
    def clip(self, x, min_val, max_val):
        """Clip the values of a tensor to a specified range."""
        return clip(x, min_val, max_val)
    
    def sigmoid(self, x):
        """Compute the sigmoid of a tensor element-wise."""
        return sigmoid(x)
    
    def softplus(self, x):
        """Compute the softplus of a tensor element-wise."""
        return softplus(x)
    
    def tanh(self, x):
        """Compute the hyperbolic tangent of a tensor element-wise."""
        return tanh(x)
    
    def relu(self, x):
        """Compute the rectified linear unit of a tensor element-wise."""
        return relu(x)
    
    def abs(self, x):
        """Compute the absolute value of a tensor element-wise."""
        return abs(x)
    
    def negative(self, x):
        """Compute the negative of a tensor element-wise."""
        return negative(x)
    
    def square(self, x):
        """Compute the square of a tensor element-wise."""
        return square(x)
    
    def min(self, x, axis=None, keepdims=False):
        """Compute the minimum value of a tensor along specified axes."""
        return min(x, axis=axis, keepdims=keepdims)
    
    def max(self, x, axis=None, keepdims=False):
        """Compute the maximum value of a tensor along specified axes."""
        return max(x, axis=axis, keepdims=keepdims)
    
    def softmax(self, x, axis=-1):
        """Compute the softmax of a tensor along a specified axis."""
        return softmax(x, axis=axis)
    
    def sin(self, x):
        """Compute the sine of a tensor element-wise."""
        return sin(x)
    
    def cos(self, x):
        """Compute the cosine of a tensor element-wise."""
        return cos(x)
    
    def tan(self, x):
        """Compute the tangent of a tensor element-wise."""
        return tan(x)
    
    def sinh(self, x):
        """Compute the hyperbolic sine of a tensor element-wise."""
        return sinh(x)
    
    def cosh(self, x):
        """Compute the hyperbolic cosine of a tensor element-wise."""
        return cosh(x)
    
    def sign(self, x):
        """Compute the sign of a tensor element-wise."""
        return sign(x)
    
    def argmax(self, x, axis=None, keepdims=False):
        """Return the indices of the maximum values along the specified axis."""
        return argmax(x, axis=axis, keepdims=keepdims)
    
    def log10(self, x):
        """Compute the base-10 logarithm of a tensor element-wise."""
        return log10(x)
    
    def log2(self, x):
        """Compute the base-2 logarithm of a tensor element-wise."""
        return log2(x)
    
    def mod(self, x, y):
        """Compute the remainder of division of x by y element-wise."""
        return mod(x, y)
    
    def floor_divide(self, x, y):
        """Element-wise integer division."""
        return floor_divide(x, y)
    
    def sort(self, x, axis=-1):
        """Sort an array along a specified axis."""
        return sort(x, axis=axis)
        
    def gradient(self, f, *varargs, axis=None, edge_order: Literal[1, 2] = 1):
        """Return the gradient of an N-dimensional array."""
        if edge_order not in (1, 2):
            raise ValueError("edge_order must be 1 or 2")
        return gradient(f, *varargs, axis=axis, edge_order=edge_order)
    
    def cumsum(self, x, axis=None):
        """Compute the cumulative sum of a tensor along a specified axis."""
        return cumsum(x, axis=axis)
    
    def eigh(self, a):
        """Compute the eigenvalues and eigenvectors of a Hermitian or symmetric matrix."""
        return eigh(a)

    def binary_split(self, a, b):
        """Recursive binary split for the Chudnovsky algorithm."""
        return binary_split(a, b)

    @property
    def pi(self):
        """Return the value of pi."""
        return mx.array([PI_CONSTANT], dtype=mx.float32)


