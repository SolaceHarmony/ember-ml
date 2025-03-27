"""
PyTorch math operations for ember_ml.

This module provides PyTorch implementations of math operations.
"""

import torch
from typing import Union, Optional, List, Tuple

from ember_ml.backend.torch.types import TensorLike, ShapeLike

# We avoid creating global instances to prevent circular imports
# Each function will create its own instances when needed

def gather(x: TensorLike, indices: TensorLike, axis: int = 0) -> torch.Tensor:
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
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_array = tensor_ops.convert_to_tensor(x)
    indices_array = tensor_ops.convert_to_tensor(indices)
    
    # Ensure indices are integer type
    # Check if the dtype contains 'int' in its name
    if isinstance(indices_array, torch.Tensor) and indices_array.dtype != torch.int64:
        # If not, cast to int64
        indices_array = indices_array.to(torch.int64)
    
    # Use torch.index_select to gather values along the specified axis
    return torch.index_select(x_array, dim=axis, index=indices_array)

def add(x: TensorLike, y: TensorLike) -> torch.Tensor:
    """
    Add two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise sum
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    return torch.add(tensor_ops.convert_to_tensor(x), tensor_ops.convert_to_tensor(y))


def subtract(x: TensorLike, y: TensorLike) -> torch.Tensor:
    """
    Subtract two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise difference
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    return torch.subtract(tensor.convert_to_tensor(x), tensor.convert_to_tensor(y))


def multiply(x: TensorLike, y: TensorLike) -> torch.Tensor:
    """
    Multiply two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise product
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    return torch.mul(tensor.convert_to_tensor(x), tensor.convert_to_tensor(y))


def divide(x: TensorLike, y: TensorLike) -> torch.Tensor:
    """
    Divide two tensors element-wise.
    
    Args:
        x: First tensor (numerator)
        y: Second tensor (denominator)
        
    Returns:
        Element-wise quotient
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    return torch.div(tensor.convert_to_tensor(x), tensor.convert_to_tensor(y))


def dot(x: TensorLike, y: TensorLike) -> torch.Tensor:
    """
    Compute the dot product of two PyTorch tensors.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Dot product
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_array = tensor_ops.convert_to_tensor(x)
    y_array = tensor_ops.convert_to_tensor(y)
    
    # Handle different dimensions
    if torch.equal(torch.tensor(len(x_array.shape)), torch.tensor(1)) and torch.equal(torch.tensor(len(y_array.shape)), torch.tensor(1)):
        return torch.sum(torch.multiply(x_array, y_array))
    else:
        return torch.matmul(x_array, y_array)


def matmul(x: TensorLike, y: TensorLike) -> torch.Tensor:
    """
    Multiply two tensors as matrices.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Matrix product
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    return torch.matmul(tensor.convert_to_tensor(x), tensor.convert_to_tensor(y))


def mean(x: TensorLike, axis: Optional[ShapeLike] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute mean of tensor elements along specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute mean.
            If None, compute mean over all elements.
        keepdims: Whether the output tensor has dim retained or not.
        
    Returns:
        Mean of tensor elements
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)
    
    # Ensure float dtype for mean calculation if input is integer
    if not x_tensor.dtype.is_floating_point and not x_tensor.dtype.is_complex:
        x_tensor = x_tensor.to(torch.float32) # Cast to float32

    if axis is None:
        # torch.mean doesn't accept keepdim when axis is None (mean over all elements)
        result = torch.mean(x_tensor)
        # If keepdims is True, we need to manually reshape the scalar result
        if keepdims:
            # Create a shape of all ones with the same ndim as input
            target_shape = (1,) * x_tensor.ndim
            return result.reshape(target_shape)
        else:
            return result
    elif isinstance(axis, tuple):
        result = x_tensor
        for dim in sorted(axis, reverse=True):
            result = torch.mean(result, dim=dim, keepdim=keepdims)
        return result
    else:
        return torch.mean(x_tensor, dim=axis, keepdim=keepdims)


def sum(x: TensorLike, axis: Optional[ShapeLike] = None, keepdim: bool = False) -> torch.Tensor:
    """
    Compute sum of tensor elements along specified axis.
    
    Args:
        x: Input array
        axis: Axis or axes along which to compute sum.
            If None, compute sum over all elements.
        keepdims: Whether the output tensor has dim retained or not.
        
    Returns:
        Sum of tensor elements
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)
    
    if axis is None:
        return torch.sum(x_tensor)
    elif isinstance(axis, tuple):
        # Sort axes in reverse order to avoid reshaping issues
        result = x_tensor
        # Sort axes in descending order to avoid dimension shifting
        for dim in sorted(axis, reverse=True):
            result = torch.sum(result, dim=dim, keepdim=keepdim)
        return result
    else:
        return torch.sum(x_tensor, dim=axis, keepdim=keepdim)

def var(x: TensorLike, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> torch.Tensor:
    """
    Compute variance of tensor elements along specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute variance.
            If None, compute variance over all elements.
        keepdims: Whether the output tensor has dim retained or not.
        
    Returns:
        Variance of tensor elements
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)
    
    if axis is None:
        result = torch.var(x_tensor, unbiased=False)
        # Handle keepdims for None axis
        if keepdim:
            # Add back all dimensions as size 1
            for _ in range(x_tensor.dim()):
                result = result.unsqueeze(0)
        return result
    
    elif isinstance(axis, tuple):
        # Sort axes in reverse order to avoid reshaping issues
        result = x_tensor
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = torch.var(result, dim=ax, unbiased=False, keepdim=keepdim)
        return result
    else:
        return torch.var(x_tensor, dim=axis, unbiased=False, keepdim=keepdim)


def exp(x: TensorLike) -> torch.Tensor:
    """
    Compute exponential of all elements in the input tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Exponential of each element in the input tensor
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.exp(TorchTensor().convert_to_tensor(x))


def log(x: TensorLike) -> torch.Tensor:
    """
    Compute natural logarithm of all elements in the input tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Natural logarithm of each element in the input tensor
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.log(TorchTensor().convert_to_tensor(x))


def pow(x: TensorLike, y: TensorLike) -> torch.Tensor:
    """
    Compute x raised to the power of y for all elements.
    
    Args:
        x: Base tensor
        y: Exponent tensor or scalar
        
    Returns:
        Tensor with elements of x raised to the power of y
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    x_tensor = tensor.convert_to_tensor(x)
    y_tensor = tensor.convert_to_tensor(y)
    return torch.pow(x_tensor, y_tensor)


def sqrt(x: TensorLike) -> torch.Tensor:
    """
    Compute the square root of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Square root of each element in the input tensor
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.sqrt(TorchTensor().convert_to_tensor(x))


def clip(x: TensorLike, 
         min_val: TensorLike, 
         max_val: TensorLike) -> torch.Tensor:
    """
    Clip tensor elements to a specified range.
    
    Args:
        x: Input tensor
        min_val: Minimum value for clipping, can be None for no lower bound
        max_val: Maximum value for clipping, can be None for no upper bound
        
    Returns:
        Tensor with clipped values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    x_array = tensor.convert_to_tensor(x)
    min_val = tensor.convert_to_tensor(min_val)
    max_val = tensor.convert_to_tensor(max_val)

    return torch.clamp(x_array, min=min_val, max=max_val)


def sigmoid(x: TensorLike) -> torch.Tensor:
    """
    Compute sigmoid function on all elements.
    
    The sigmoid function is defined as:
    sigmoid(x) = 1 / (1 + exp(-x))
    
    It transforms each element of input tensor to a value between 0 and 1.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with sigmoid applied to each element
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)
    x_safe = clip(x_tensor, -88.0, 88.0)  # Prevent overflow
    return torch.sigmoid(x_safe)


def softplus(x: TensorLike) -> torch.Tensor:
    """
    Compute the softplus of tensor elements element-wise.
    
    The softplus function is defined as log(1 + exp(x)).
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with softplus applied to each element
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)
    x_safe = torch.clamp(x_tensor, min=-88.0, max=88.0)  # Prevent overflow
    return torch.log(torch.add(torch.tensor(1.0), torch.exp(x_safe)))


def tanh(x: TensorLike) -> torch.Tensor:
    """
    Compute hyperbolic tangent of all elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise tanh
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.tanh(TorchTensor().convert_to_tensor(x))


def relu(x: TensorLike) -> torch.Tensor:
    """
    Apply Rectified Linear Unit function element-wise.


        
    Args:
        x: Input tensor
        
    Returns:
        Tensor with ReLU applied to each element
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.relu(TorchTensor().convert_to_tensor(x))

def abs(x: TensorLike) -> torch.Tensor:
    """
    Compute absolute value of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with absolute values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.abs(TorchTensor().convert_to_tensor(x))


def negative(x: TensorLike) -> torch.Tensor:
    """
    Compute the negative of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with negated values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.negative(TorchTensor().convert_to_tensor(x))


def sign(x: TensorLike) -> torch.Tensor:
    """
    Compute the sign of tensor elements.
    
    Returns -1 for negative values, 0 for zero, and 1 for positive values.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with sign values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.sign(TorchTensor().convert_to_tensor(x))




def argmax(x: TensorLike, axis: Optional[int] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Returns the indices of the maximum values along an axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to find the indices of maximum values.
            If None, the index is for the flattened tensor.
        keepdims: Whether the output tensor has dim retained or not.
        
    Returns:
        Indices of maximum values along the specified axis
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)
    
    if axis is None:
        result = torch.argmax(x_tensor.flatten())
        # For None axis with keepdims=True, we need a scalar tensor with shape (1,)
        if keepdims:
            return result.reshape(1)
        return result
    else:
        result = torch.argmax(x_tensor, dim=axis)
        # Handle keepdims by unsqueezing the reduced dimension if needed
        if keepdims:
            return result.unsqueeze(dim=axis)
        return result


def sin(x: TensorLike) -> torch.Tensor:
    """
    Compute sine of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with sine of values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.sin(TorchTensor().convert_to_tensor(x))


def cos(x: TensorLike) -> torch.Tensor:
    """
    Compute cosine of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with cosine of values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.cos(TorchTensor().convert_to_tensor(x))


def tan(x: TensorLike) -> torch.Tensor:
    """
    Compute tangent of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with tangent of values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.tan(TorchTensor().convert_to_tensor(x))




def sinh(x: TensorLike) -> torch.Tensor:
    """
    Compute hyperbolic sine of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with hyperbolic sine of values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.sinh(TorchTensor().convert_to_tensor(x))


def cosh(x: TensorLike) -> torch.Tensor:
    """
    Compute hyperbolic cosine of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with hyperbolic cosine of values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.cosh(TorchTensor().convert_to_tensor(x))



def log10(x: TensorLike) -> torch.Tensor:
    """
    Compute base-10 logarithm of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with base-10 logarithm of values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.log10(TorchTensor().convert_to_tensor(x))




def log2(x: TensorLike) -> torch.Tensor:
    """
    Compute base-2 logarithm of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with base-2 logarithm of values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.log2(TorchTensor().convert_to_tensor(x))


def square(x: TensorLike) -> torch.Tensor:
    """
    Compute square of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with squared values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.square(TorchTensor().convert_to_tensor(x))

def mod(x: TensorLike, y: TensorLike) -> torch.Tensor:
    """
    Compute element-wise remainder of division.
    
    Args:
        x: Input tensor (dividend)
        y: Input tensor (divisor)
        
    Returns:
        Tensor with element-wise remainder
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    x_tensor = tensor.convert_to_tensor(x)
    y_tensor = tensor.convert_to_tensor(y)
    
    return torch.remainder(x_tensor, y_tensor)

def floor_divide(x: TensorLike, y: TensorLike) -> torch.Tensor:
    """
    Compute element-wise integer division.
    
    Args:
        x: Input tensor (dividend)
        y: Input tensor (divisor)
        
    Returns:
        Tensor with element-wise integer division result
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()

    x_tensor = tensor.convert_to_tensor(x)
    y_tensor = tensor.convert_to_tensor(y)

    # Use floor_divide to perform integer division    
    return torch.floor_divide(x_tensor, y_tensor)


def min(x: TensorLike, axis: Optional[int] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute minimum of tensor elements along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to find minimum.
            If None, find minimum over all elements.
        keepdims: Whether the output tensor has dim retained or not.
        
    Returns:
        Minimum values along the specified axis
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)
    
    if axis is None:
        # Find the minimum
        return torch.min(x_tensor)
    return torch.min(x_tensor, dim=axis, keepdim=keepdims).values


def max(x: TensorLike, axis: Optional[int] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute maximum of tensor elements along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to find maximum.
            If None, find maximum over all elements.
        keepdims: Whether the output tensor has dim retained or not.
        
    Returns:
        Maximum values along the specified axis
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)

    if axis is None:
        return torch.max(x_tensor)
    return torch.max(x_tensor, dim=axis, keepdim=keepdims).values

def softmax(x: TensorLike, axis: int = -1) -> torch.Tensor:
    """
    Compute softmax values for the last dimension of the tensor.
    
    softmax(x) = exp(x_i) / sum(exp(x_j))
    
    Args:
        x: Input tensor
        axis: Dimension along which softmax will be computed (default: -1, the last dimension)
        
    Returns:
        Tensor with softmax applied along the specified dimension
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)
    
    # For numerical stability, subtract the maximum value before taking the exponential
    if x_tensor.dim() > 1:
        max_vals = torch.max(x_tensor, dim=axis, keepdim=True).values
        exp_vals = torch.exp(torch.subtract(x_tensor, max_vals))
        sum_exp = torch.sum(exp_vals, dim=axis, keepdim=True)
        return torch.div(exp_vals, sum_exp)
    else:
        # Handle 1D case
        max_val = torch.max(x_tensor)
        exp_vals = torch.exp(torch.subtract(x_tensor, max_val))
        sum_exp = torch.sum(exp_vals)
        return torch.div(exp_vals, sum_exp)


def sort(x: TensorLike, axis: int = -1) -> torch.Tensor:
    """
    Sort tensor along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to sort (default: -1)
        
    Returns:
        Sorted tensor
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.sort(TorchTensor().convert_to_tensor(x), dim=axis).values 


def gradient(f: TensorLike, *varargs, axis: Optional[Union[int, Tuple[int, ...]]] = None,
             edge_order: int = 1) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Return the gradient of an N-dimensional tensor.
    
    The gradient is computed using finite differences. This implementation
    approximates numpy.gradient using PyTorch operations.
    
    Args:
        f: An N-dimensional tensor containing samples of a scalar function.
        *varargs: Spacing between f values. Default unitary spacing for all dimensions.
        axis: Gradient is calculated only along the given axis or axes.
            The default (axis = None) is to calculate the gradient for all the axes.
        edge_order: Gradient is calculated using N-th order accurate differences at the boundaries.
            Must be 1 or 2.
            
    Returns:
        A tensor or tuple of tensors corresponding to the derivatives of f with respect to each dimension.
        Each derivative has the same shape as f.
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    f_array = tensor_ops.convert_to_tensor(f)

    # Get the shape of the input tensor
    f_shape = f_array.shape
    ndim = len(f_shape)

    # Determine the axes along which to compute the gradient
    if axis is None:
        axes = tuple(range(f_array.dim()))
    elif isinstance(axis, int):
        axes = (axis,)
    else:
        axes = axis
        
    # Initialize spacing for each dimension
    spacings = []
    if len(varargs) == 0:
        # Default: unitary spacing for all dimensions
        spacings = [torch.tensor(1.0)] * len(axes)
    elif len(varargs) == 1:
        # Same spacing for all dimensions
        spacings = [tensor_ops.convert_to_tensor(varargs[0])] * len(axes)
    else:
        # Different spacing for each dimension
        if len(varargs) != len(axes):
            raise ValueError("If spacing is specified for each axis, the number of "
                            "spacing values must match the number of axes.")
        spacings = [tensor_ops.convert_to_tensor(spacing) for spacing in varargs]
    
    # Compute the gradient along each specified axis
    result = []
    
    for i, axis_i in enumerate(axes):
        # Get the spacing for this axis
        dx = spacings[i]
        
        # Create slices for forward and backward differences
        slice_prev = [slice(None)] * f_array.dim()
        slice_next = [slice(None)] * f_array.dim()
        slice_center = [slice(None)] * f_array.dim()
        
        # Compute the gradient using finite differences
        if edge_order == 1:
            # Forward difference at the beginning
            slice_prev[axis_i] = slice(0, 1)
            slice_next[axis_i] = slice(1, 2)
            
            # Use torch operations instead of Python operators
            forward_diff = torch.div(
                torch.subtract(f_array[tuple(slice_next)], f_array[tuple(slice_prev)]), 
                dx
            )
            
            # Backward difference at the end
            slice_prev[axis_i] = slice(-2, -1)
            slice_next[axis_i] = slice(-1, None)
            
            # Use torch operations instead of Python operators
            backward_diff = torch.div(
                torch.subtract(f_array[tuple(slice_next)], f_array[tuple(slice_prev)]), 
                dx
            )
            
            # Central difference in the middle
            slice_prev[axis_i] = slice(0, -2)
            slice_center[axis_i] = slice(1, -1)
            slice_next[axis_i] = slice(2, None)
            
            # Use torch operations instead of Python operators
            central_diff = torch.div(
                torch.subtract(f_array[tuple(slice_next)], f_array[tuple(slice_prev)]), 
                torch.multiply(torch.tensor(2.0), dx)
            )
            
            # Combine the differences
            grad = torch.zeros_like(f_array)
            
            # Assign the forward difference at the beginning
            slice_center[axis_i] = slice(0, 1)
            grad[tuple(slice_center)] = forward_diff
            
            # Assign the central difference in the middle
            slice_center[axis_i] = slice(1, -1)
            grad[tuple(slice_center)] = central_diff
            
            # Assign the backward difference at the end
            slice_center[axis_i] = slice(-1, None)
            grad[tuple(slice_center)] = backward_diff
            
        elif edge_order == 2:
            # Second-order accurate differences
            # For simplicity, we'll implement a basic version here
            # A more accurate implementation would use higher-order finite differences
            
            # Central difference for interior points
            slice_prev[axis_i] = slice(0, -2)
            slice_center[axis_i] = slice(1, -1)
            slice_next[axis_i] = slice(2, None)
            
            # Use torch operations instead of Python operators
            central_diff = torch.div(
                torch.subtract(f_array[tuple(slice_next)], f_array[tuple(slice_prev)]),
                torch.multiply(torch.tensor(2.0), dx)
            )
            
            # Second-order accurate differences at the boundaries
            # For the beginning
            slice_0 = [slice(None)] * f_array.dim()
            slice_1 = [slice(None)] * f_array.dim()
            slice_2 = [slice(None)] * f_array.dim()
            
            slice_0[axis_i] = slice(0, 1)
            slice_1[axis_i] = slice(1, 2)
            slice_2[axis_i] = slice(2, 3)
            
            if f_shape[axis_i] > 2:
                # Use torch operations instead of Python operators
                term1 = torch.multiply(torch.tensor(-3.0), f_array[tuple(slice_0)])
                term2 = torch.multiply(torch.tensor(4.0), f_array[tuple(slice_1)])
                term3 = torch.negative(f_array[tuple(slice_2)])
                
                sum_terms = torch.add(torch.add(term1, term2), term3)
                forward_diff = torch.div(sum_terms, torch.multiply(torch.tensor(2.0), dx))
            else:
                # Use torch operations instead of Python operators
                forward_diff = torch.div(
                    torch.subtract(f_array[tuple(slice_1)], f_array[tuple(slice_0)]),
                    dx
                )
            
            # For the end
            slice_n2 = [slice(None)] * f_array.dim()
            slice_n1 = [slice(None)] * f_array.dim()
            slice_n = [slice(None)] * f_array.dim()
            
            slice_n2[axis_i] = slice(-3, -2)
            slice_n1[axis_i] = slice(-2, -1)
            slice_n[axis_i] = slice(-1, None)
            
            if f_shape[axis_i] > 2:
                # Use torch operations instead of Python operators
                term1 = torch.multiply(torch.tensor(3.0), f_array[tuple(slice_n)])
                term2 = torch.multiply(torch.tensor(-4.0), f_array[tuple(slice_n1)])
                term3 = f_array[tuple(slice_n2)]
                
                sum_terms = torch.add(torch.add(term1, term2), term3)
                backward_diff = torch.div(sum_terms, torch.multiply(torch.tensor(2.0), dx))
            else:
                # Use torch operations instead of Python operators
                backward_diff = torch.div(
                    torch.subtract(f_array[tuple(slice_n)], f_array[tuple(slice_n1)]),
                    dx
                )
            
            # Combine the differences
            grad = torch.zeros_like(f_array)
            
            # Assign the forward difference at the beginning
            slice_center[axis_i] = slice(0, 1)
            grad[tuple(slice_center)] = forward_diff
            
            # Assign the central difference in the middle
            slice_center[axis_i] = slice(1, -1)
            grad[tuple(slice_center)] = central_diff
            
            # Assign the backward difference at the end
            slice_center[axis_i] = slice(-1, None)
            grad[tuple(slice_center)] = backward_diff
            
        else:
            raise ValueError("Edge order must be 1 or 2.")
        
        result.append(grad)
    
    # Return a single tensor if only one axis was specified
    if len(result) == 1:
        return result[0]
    else:
        return result
    

def cumsum(x: TensorLike, axis: Optional[int] = None) -> torch.Tensor:
    """
    Compute the cumulative sum of tensor elements along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to compute the cumulative sum. If None, the
            flattened tensor's cumulative sum is returned.
        
    Returns:
        Tensor with cumulative sum along the specified axis
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)
    
    if axis is None:
        return torch.cumsum(x_tensor.flatten(), dim=0)
    else:
        return torch.cumsum(x_tensor, dim=axis)


def eigh(a: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.
    
    Args:
        a: Input Hermitian or symmetric matrix
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    a_tensor = TorchTensor().convert_to_tensor(a)
    return torch.linalg.eigh(a_tensor)


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
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    C = torch.tensor(640320)
    C3_OVER_24 = torch.divide(torch.pow(C, 3), torch.tensor(24))
    DIGITS_PER_TERM = torch.tensor(14.1816474627254776555)  # Approx. digits per iteration
    
    def binary_split(a, b):
        """Recursive binary split for the Chudnovsky algorithm."""
        from ember_ml.backend.torch.tensor import TorchTensor
        inner_tensor = TorchTensor()
        a_tensor = inner_tensor.convert_to_tensor(a)
        b_tensor = inner_tensor.convert_to_tensor(b)
        diff = torch.subtract(b_tensor, a_tensor)
        
        if torch.equal(diff, torch.tensor(1)):
            # Base case
            if torch.equal(a_tensor, torch.tensor(0)):
                Pab = torch.tensor(1)
                Qab = torch.tensor(1)
            else:
                term1 = torch.subtract(torch.multiply(torch.tensor(6), a_tensor), torch.tensor(5))
                term2 = torch.subtract(torch.multiply(torch.tensor(2), a_tensor), torch.tensor(1))
                term3 = torch.subtract(torch.multiply(torch.tensor(6), a_tensor), torch.tensor(1))
                Pab = torch.multiply(torch.multiply(term1, term2), term3)
                Qab = torch.multiply(torch.pow(a_tensor, 3), C3_OVER_24)
            
            base_term = torch.tensor(13591409)
            multiplier = torch.tensor(545140134)
            term = torch.add(base_term, torch.multiply(multiplier, a_tensor))
            Tab = torch.multiply(Pab, term)
            
            # Check if a is odd using remainder comparison
            remainder = torch.remainder(a_tensor, torch.tensor(2))
            is_odd = torch.eq(remainder, torch.tensor(1))
            
            # If a is odd, negate Tab
            Tab = torch.where(is_odd, torch.negative(Tab), Tab)
            
            return Pab, Qab, Tab
        
        # Recursive case
        m = torch.divide(torch.add(a_tensor, b_tensor), torch.tensor(2))
        m = torch.floor(m)  # Ensure m is an integer
        
        Pam, Qam, Tam = binary_split(a, m)
        Pmb, Qmb, Tmb = binary_split(m, b)
        
        Pab = torch.multiply(Pam, Pmb)
        Qab = torch.multiply(Qam, Qmb)
        term1 = torch.multiply(Qmb, Tam)
        term2 = torch.multiply(Pam, Tmb)
        Tab = torch.add(term1, term2)
        
        return Pab, Qab, Tab
    
    # Number of terms needed for the desired precision
    precision_tensor = tensor_ops.convert_to_tensor(precision_digits)
    terms_float = torch.divide(precision_tensor, DIGITS_PER_TERM)
    terms_float = torch.add(terms_float, torch.tensor(1))
    terms = torch.floor(terms_float)  # Convert to integer
    terms_int = terms.to(torch.int32)  # Convert to int32 using PyTorch's to() method
    
    # Compute the binary split
    P, Q, T = binary_split(0, terms_int)
    
    # Calculate pi
    sqrt_10005 = torch.sqrt(torch.tensor(10005))
    numerator = torch.multiply(Q, torch.tensor(426880))
    numerator = torch.multiply(numerator, sqrt_10005)
    pi_approx = torch.divide(numerator, T)
    
    # Return as PyTorch tensor with shape (1,)
    return torch.reshape(pi_approx, (1,))

# Calculate pi with appropriate precision for PyTorch (float32)
# Ensure it's a scalar with shape (1,) as per PyTorch conventions
PI_CONSTANT = _calculate_pi_value(15)  # Increased precision to match reference value

pi : torch.tensor = torch.tensor([PI_CONSTANT], dtype=torch.float32)


def binary_split(a: TensorLike, b: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recursive binary split for the Chudnovsky algorithm.
    
    This is used in the implementation of PI calculation for the PyTorch backend.
    
    Args:
        a: Start value
        b: End value
        
    Returns:
        Tuple of intermediate values for PI calculation
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    a_tensor = tensor.convert_to_tensor(a)
    b_tensor = tensor.convert_to_tensor(b)
    
    # Use torch operations
    diff = torch.subtract(b_tensor, a_tensor)
    
    if torch.equal(diff, torch.tensor(1.0)):
        # Base case
        if torch.equal(a_tensor, torch.tensor(0.0)):
            Pab = torch.tensor(1.0)
            Qab = torch.tensor(1.0)
        else:
            # Calculate terms using torch operations
            term1 = torch.subtract(torch.multiply(torch.tensor(6.0), a_tensor), torch.tensor(5.0))
            term2 = torch.subtract(torch.multiply(torch.tensor(2.0), a_tensor), torch.tensor(1.0))
            term3 = torch.subtract(torch.multiply(torch.tensor(6.0), a_tensor), torch.tensor(1.0))
            Pab = torch.multiply(torch.multiply(term1, term2), term3)
            
            # Define C3_OVER_24
            C = torch.tensor(640320.0)
            C3_OVER_24 = torch.div(torch.pow(C, torch.tensor(3.0)), torch.tensor(24.0))
            
            Qab = torch.multiply(torch.pow(a_tensor, torch.tensor(3.0)), C3_OVER_24)
        
        return Pab, Qab
    else:
        # Recursive case
        m = torch.div(torch.add(a_tensor, b_tensor), torch.tensor(2.0))
        Pam, Qam = binary_split(a_tensor, m)
        Pmb, Qmb = binary_split(m, b_tensor)
        
        Pab = torch.multiply(Pam, Pmb)
        Qab = torch.add(torch.multiply(Qam, Pmb), torch.multiply(Pam, Qmb))
        
        return Pab, Qab

class TorchMathOps:
    """PyTorch implementation of math operations."""
    
    def gather(self, x, indices, axis=0):
        """Gather slices from tensor according to indices."""
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
        # Correct implementation using torch.sum and handling keepdims
        from ember_ml.backend.torch.tensor import TorchTensor # Import needed here
        x_tensor = TorchTensor().convert_to_tensor(x)
        
        if axis is None:
            # torch.sum doesn't accept keepdim when axis is None
            result = torch.sum(x_tensor)
            if keepdims:
                # Manually reshape to keep dimensions if keepdims=True
                target_shape = (1,) * x_tensor.ndim
                return result.reshape(target_shape)
            else:
                return result
        elif isinstance(axis, tuple):
            # Handle tuple axis by summing iteratively
            result = x_tensor
            for dim in sorted(axis, reverse=True):
                result = torch.sum(result, dim=dim, keepdim=keepdims)
            return result
        else:
            # Handle single axis
            return torch.sum(x_tensor, dim=axis, keepdim=keepdims)
        
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
        
    def gradient(self, f, *varargs, axis=None, edge_order=1):
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
        return torch.tensor([PI_CONSTANT], dtype=torch.float32)


