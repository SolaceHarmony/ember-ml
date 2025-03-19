"""
PyTorch math operations for ember_ml.

This module provides PyTorch implementations of math operations.
"""

import torch
from typing import Union, Optional, Tuple

# Import from tensor_ops
from ember_ml.backend.torch.tensor import TorchTensor

convert_to_tensor = TorchTensor().convert_to_tensor

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Add two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise sum
    """
    return torch.add(convert_to_tensor(x), convert_to_tensor(y))


def subtract(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Subtract two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise difference
    """
    return torch.subtract(convert_to_tensor(x), convert_to_tensor(y))


def multiply(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Multiply two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise product
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    
    # Ensure both tensors are on the same device
    if x_tensor.device != y_tensor.device:
        y_tensor = y_tensor.to(x_tensor.device)
    
    return torch.mul(x_tensor, y_tensor)


def divide(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Divide two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise quotient
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    
    # Ensure both tensors are on the same device
    if x_tensor.device != y_tensor.device:
        y_tensor = y_tensor.to(x_tensor.device)
    
    return torch.div(x_tensor, y_tensor)


def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the matrix product of two tensors.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Matrix product
    """
    return torch.matmul(convert_to_tensor(x), convert_to_tensor(y))


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the dot product of two tensors.
    
    Args:
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Dot product of x and y
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return torch.matmul(x_tensor.flatten(), y_tensor.flatten())


def mean(x: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute the mean of a tensor along specified axes.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the mean
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Mean of the tensor
    """
    x_tensor = convert_to_tensor(x)
    
    # Cast to float32 if the tensor is an integer type
    if x_tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool]:
        x_tensor = x_tensor.to(torch.float32)
    
    if axis is None:
        return torch.mean(x_tensor)
    
    if isinstance(axis, (list, tuple)):
        # PyTorch doesn't support multiple axes directly, so we need to do it sequentially
        result = x_tensor
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = torch.mean(result, dim=ax, keepdim=keepdims)
        return result
    
    return torch.mean(x_tensor, dim=axis, keepdim=keepdims)


def var(x: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute the variance of a tensor along specified axes.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Variance of the tensor
    """
    x_tensor = convert_to_tensor(x)
    
    if axis is None:
        return torch.var(x_tensor)
    
    if isinstance(axis, (list, tuple)):
        # PyTorch doesn't support multiple axes directly, so we need to do it sequentially
        result = x_tensor
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = torch.var(result, dim=ax, keepdim=keepdims)
        return result
    
    return torch.var(x_tensor, dim=axis, keepdim=keepdims)


def sum(x: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute the sum of a tensor along specified axes.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the sum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Sum of the tensor
    """
    x_tensor = convert_to_tensor(x)
    
    if axis is None:
        return torch.sum(x_tensor)
    
    if isinstance(axis, (list, tuple)):
        # PyTorch doesn't support multiple axes directly, so we need to do it sequentially
        result = x_tensor
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = torch.sum(result, dim=ax, keepdim=keepdims)
        return result
    
    return torch.sum(x_tensor, dim=axis, keepdim=keepdims)


def exp(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the exponential of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise exponential
    """
    return torch.exp(convert_to_tensor(x))


def log(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the natural logarithm of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise logarithm
    """
    return torch.log(convert_to_tensor(x))


def pow(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute x raised to the power of y element-wise.
    
    Args:
        x: Base tensor
        y: Exponent tensor
        
    Returns:
        Element-wise power
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    
    # Ensure both tensors are on the same device
    if x_tensor.device != y_tensor.device:
        y_tensor = y_tensor.to(x_tensor.device)
    
    return torch.pow(x_tensor, y_tensor)


def sqrt(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the square root of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise square root
    """
    return torch.sqrt(convert_to_tensor(x))


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the sigmoid of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise sigmoid
    """
    return torch.sigmoid(convert_to_tensor(x))


def softplus(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    """
    Compute the softplus of a tensor element-wise.
    
    The softplus function is defined as (1/beta) * log(1 + exp(beta * x)).
    For numerical stability, the implementation reverts to a linear function
    when input * beta > threshold.
    
    Args:
        x: Input tensor
        beta: The beta value for the softplus formulation. Default: 1.0
        threshold: Values above this revert to a linear function. Default: 20.0
        
    Returns:
        Element-wise softplus
    """
    x_tensor = convert_to_tensor(x)
    # Use PyTorch's built-in softplus function with parameters
    return torch.nn.functional.softplus(x_tensor, beta=beta, threshold=threshold)


def tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the hyperbolic tangent of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise tanh
    """
    return torch.tanh(convert_to_tensor(x))


def relu(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the rectified linear unit of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise ReLU
    """
    return torch.relu(convert_to_tensor(x))


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """
    Compute the softmax of a tensor along a specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to compute the softmax
        
    Returns:
        Softmax of the tensor
    """
    x_tensor = convert_to_tensor(x)
    
    # Handle the case where axis is None (not supported by PyTorch)
    if axis is None:
        # Flatten the tensor and apply softmax along the only dimension
        x_flat = x_tensor.reshape(-1)
        x_max = torch.max(x_flat)
        exp_x = torch.exp(torch.subtract(x_flat, x_max))
        return torch.div(exp_x, torch.sum(exp_x))
    
    # Manual implementation of softmax to avoid issues with PyTorch's softmax
    # First, find the maximum value along the specified axis for numerical stability
    x_max = torch.max(x_tensor, dim=axis, keepdim=True)[0]
    
    # Subtract the maximum value and compute the exponential
    exp_x = torch.exp(torch.subtract(x_tensor, x_max))
    
    # Compute the sum along the specified axis
    sum_exp_x = torch.sum(exp_x, dim=axis, keepdim=True)
    
    # Compute the softmax
    return torch.div(exp_x, sum_exp_x)


def max(x: torch.Tensor, axis: Optional[int] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute the maximum of a tensor along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the maximum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Maximum of x along the specified axis
    """
    x_tensor = convert_to_tensor(x)
    if axis is None:
        return torch.max(x_tensor)
    return torch.max(x_tensor, dim=axis, keepdim=keepdims)[0]


def min(x: torch.Tensor, axis: Optional[int] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute the minimum of a tensor along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the minimum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Minimum of x along the specified axis
    """
    x_tensor = convert_to_tensor(x)
    if axis is None:
        return torch.min(x_tensor)
    return torch.min(x_tensor, dim=axis, keepdim=keepdims)[0]

def abs(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Absolute value of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.abs(x_tensor)


def negative(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the negative of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise negative
    """
    x_tensor = convert_to_tensor(x)
    return torch.negative(x_tensor)


def sign(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the sign of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Sign of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.sign(x_tensor)


def sin(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the sine of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Sine of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.sin(x_tensor)


def cos(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Cosine of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.cos(x_tensor)


def tan(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the tangent of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Tangent of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.tan(x_tensor)


def sinh(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the hyperbolic sine of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Hyperbolic sine of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.sinh(x_tensor)


def cosh(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the hyperbolic cosine of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Hyperbolic cosine of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.cosh(x_tensor)


def log10(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the base-10 logarithm of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Base-10 logarithm of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.log10(x_tensor)


def log2(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the base-2 logarithm of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Base-2 logarithm of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.log2(x_tensor)


def square(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the square of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Square of x
    """
    x_tensor = convert_to_tensor(x)
    return torch.square(x_tensor)


def clip(x: torch.Tensor, min_val: Optional[float] = None, max_val: Optional[float] = None) -> torch.Tensor:
    """
    Clip the values of a tensor to the specified range.
    
    Args:
        x: Input tensor
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clipped tensor
    """
    x_tensor = convert_to_tensor(x)
    return torch.clamp(x_tensor, min=min_val, max=max_val)


def mod(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the remainder of division of x by y element-wise.
    
    Args:
        x: Input tensor (dividend)
        y: Input tensor (divisor)
        
    Returns:
        Element-wise remainder
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    
    # Ensure both tensors are on the same device
    if x_tensor.device != y_tensor.device:
        y_tensor = y_tensor.to(x_tensor.device)
    
    # Use remainder to get the modulo result
    return torch.remainder(x_tensor, y_tensor)


def floor_divide(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Element-wise integer division.
    
    If either array is a floating point type then it is equivalent to calling floor() after divide().
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise integer quotient (a // b)
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    
    # Ensure both tensors are on the same device
    if x_tensor.device != y_tensor.device:
        y_tensor = y_tensor.to(x_tensor.device)
    
    return torch.floor_divide(x_tensor, y_tensor)


def sort(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """
    Sort a PyTorch tensor along a specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to sort
        
    Returns:
        Sorted tensor
    """
    x_tensor = convert_to_tensor(x)
    return torch.sort(x_tensor, dim=axis)[0]


from typing import Literal



def gradient(f: torch.Tensor, *varargs, axis: Optional[Union[int, Tuple[int, ...]]] = None,
            edge_order: Literal[1, 2] = 1) -> Union[torch.Tensor, list[torch.Tensor]]:
    """
    Return the gradient of an N-dimensional tensor.
    
    The gradient is computed using second order accurate central differences in the interior
    points and either first or second order accurate one-sides (forward or backwards)
    differences at the boundaries. The returned gradient hence has the same shape as the input tensor.
    
    Args:
        f: An N-dimensional tensor containing samples of a scalar function.
        *varargs: Spacing between f values. Default unitary spacing for all dimensions.
        axis: Gradient is calculated only along the given axis or axes.
            The default (axis = None) is to calculate the gradient for all the axes of the input tensor.
        edge_order: Gradient is calculated using N-th order accurate differences at the boundaries.
            Must be 1 or 2.
            
    Returns:
        A tensor or tuple of tensors corresponding to the derivatives of f with respect to each dimension.
        Each derivative has the same shape as f.
    """
    f_tensor = convert_to_tensor(f)
    
    # Process spacing arguments
    spacings = []
    for arg in varargs:
        if isinstance(arg, torch.Tensor):
            spacings.append(arg)
        else:
            spacings.append(torch.tensor(arg, device=f_tensor.device))
    
    # Default spacing is 1.0
    if not spacings:
        # Create a list of spacing tensors without using Python multiplication
        dim_count = f_tensor.dim()
        spacings = []
        for _ in range(dim_count):
            spacings.append(torch.tensor(1.0, device=f_tensor.device))
    
    # If axis is None, compute gradient for all dimensions
    if axis is None:
        # Create a list of axes without using Python range
        dim_count = f_tensor.dim()
        axes = []
        for i in range(dim_count):
            axes.append(i)
    elif isinstance(axis, (list, tuple)):
        # Convert to list without using list() constructor
        axes = []
        for ax in axis:
            axes.append(ax)
    else:
        # Create a single-element list without using list brackets
        axes = []
        axes.append(axis)
    
    results = []
    for i, ax in enumerate(axes):
        # Get spacing for this axis
        spacing = spacings[i] if i < len(spacings) else torch.tensor(1.0, device=f_tensor.device)
        
        # Create slices for forward and backward differences
        ndim = f_tensor.dim()
        # Create lists without using Python multiplication
        slice_prev = []
        slice_next = []
        slice_result = []
        for _ in range(ndim):
            slice_prev.append(slice(None))
            slice_next.append(slice(None))
            slice_result.append(slice(None))
        
        # Compute gradient along this axis
        if edge_order == 1:
            # First-order accurate differences at the boundaries
            # Forward difference at the beginning
            slice_prev[ax] = slice(0, 1)
            slice_next[ax] = slice(1, 2)
            slice_result[ax] = slice(0, 1)
            forward_diff = torch.div(
                torch.subtract(f_tensor[tuple(slice_next)], f_tensor[tuple(slice_prev)]),
                spacing
            )
            
            # Central difference in the interior
            slice_prev[ax] = slice(0, -2)
            slice_next[ax] = slice(2, None)
            slice_result[ax] = slice(1, -1)
            central_diff = torch.div(
                torch.subtract(f_tensor[tuple(slice_next)], f_tensor[tuple(slice_prev)]),
                torch.multiply(torch.tensor(2.0, device=f_tensor.device), spacing)
            )
            
            # Backward difference at the end
            slice_prev[ax] = slice(-2, -1)
            slice_next[ax] = slice(-1, None)
            slice_result[ax] = slice(-1, None)
            backward_diff = torch.div(
                torch.subtract(f_tensor[tuple(slice_next)], f_tensor[tuple(slice_prev)]),
                spacing
            )
            
            # Combine the differences
            result = torch.zeros_like(f_tensor)
            result[tuple(slice_result)] = forward_diff
            slice_result[ax] = slice(1, -1)
            result[tuple(slice_result)] = central_diff
            slice_result[ax] = slice(-1, None)
            result[tuple(slice_result)] = backward_diff
            
        elif edge_order == 2:
            # Second-order accurate differences at the boundaries
            # Forward difference at the beginning
            if f_tensor.shape[ax] > 2:
                # Define all slices needed without using Python multiplication
                slice_prev = []
                slice_mid = []
                slice_next = []
                slice_result = []
                for _ in range(ndim):
                    slice_prev.append(slice(None))
                    slice_mid.append(slice(None))
                    slice_next.append(slice(None))
                    slice_result.append(slice(None))
                
                # Forward difference at the beginning
                slice_prev[ax] = slice(0, 1)
                slice_mid[ax] = slice(1, 2)
                slice_next[ax] = slice(2, 3)
                slice_result[ax] = slice(0, 1)
                
                # Calculate: (-3 * prev + 4 * mid - next) / (2 * spacing)
                neg_three = torch.tensor(-3.0, device=f_tensor.device)
                four = torch.tensor(4.0, device=f_tensor.device)
                neg_one = torch.tensor(-1.0, device=f_tensor.device)
                two = torch.tensor(2.0, device=f_tensor.device)
                
                term1 = torch.multiply(neg_three, f_tensor[tuple(slice_prev)])
                term2 = torch.multiply(four, f_tensor[tuple(slice_mid)])
                term3 = torch.multiply(neg_one, f_tensor[tuple(slice_next)])
                
                sum_terms = torch.add(torch.add(term1, term2), term3)
                forward_diff = torch.div(sum_terms, torch.multiply(two, spacing))
                
                # Central difference in the interior
                slice_prev[ax] = slice(0, -2)
                slice_next[ax] = slice(2, None)
                slice_result[ax] = slice(1, -1)
                
                # Calculate: (next - prev) / (2 * spacing)
                diff_terms = torch.subtract(f_tensor[tuple(slice_next)], f_tensor[tuple(slice_prev)])
                central_diff = torch.div(diff_terms, torch.multiply(two, spacing))
                
                # Backward difference at the end
                slice_prev[ax] = slice(-3, -2)
                slice_mid[ax] = slice(-2, -1)
                slice_next[ax] = slice(-1, None)
                slice_result[ax] = slice(-1, None)
                
                # Calculate: (prev - 4 * mid + 3 * next) / (2 * spacing)
                three = torch.tensor(3.0, device=f_tensor.device)
                
                term1 = f_tensor[tuple(slice_prev)]
                term2 = torch.multiply(neg_one, torch.multiply(four, f_tensor[tuple(slice_mid)]))
                term3 = torch.multiply(three, f_tensor[tuple(slice_next)])
                
                sum_terms = torch.add(torch.add(term1, term2), term3)
                backward_diff = torch.div(sum_terms, torch.multiply(two, spacing))
                
                # Combine the differences
                result = torch.zeros_like(f_tensor)
                result[tuple(slice_result)] = forward_diff
                slice_result[ax] = slice(1, -1)
                result[tuple(slice_result)] = central_diff
                slice_result[ax] = slice(-1, None)
                result[tuple(slice_result)] = backward_diff
            else:
                # Fall back to first-order for small tensors
                # Use PyTorch's gradient function with spacing as a list
                spacing_list = [spacing]
                result = torch.gradient(f_tensor, spacing=spacing_list, dim=ax)[0]
        else:
            raise ValueError("edge_order must be 1 or 2")
        
        results.append(result)
    
    # Return a single tensor if only one axis, otherwise a list
    if len(results) == 1:
        return results[0]
    else:
        return results


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
    from ember_ml.backend.torch.config import DEFAULT_DEVICE
    
    # Create all constants on the default device
    C = torch.tensor(640320.0, device=DEFAULT_DEVICE)
    C3_OVER_24 = torch.div(torch.pow(C, 3), torch.tensor(24.0, device=DEFAULT_DEVICE))
    DIGITS_PER_TERM = torch.tensor(14.1816474627254776555, device=DEFAULT_DEVICE)  # Approx. digits per iteration
    
    def binary_split(a, b):
        """Recursive binary split for the Chudnovsky algorithm."""
        a_tensor = convert_to_tensor(a)
        b_tensor = convert_to_tensor(b)
        diff = torch.subtract(b_tensor, a_tensor)
        
        # Create comparison tensor on the same device as diff
        one_tensor = torch.tensor(1.0, device=diff.device)
        if torch.equal(diff, one_tensor):
            # Base case
            # Create tensors on the same device as a_tensor
            zero_tensor = torch.tensor(0.0, device=a_tensor.device)
            one_tensor = torch.tensor(1.0, device=a_tensor.device)
            if torch.equal(a_tensor, zero_tensor):
                Pab = torch.tensor(1.0, device=a_tensor.device)
                Qab = torch.tensor(1.0, device=a_tensor.device)
            else:
                six_tensor = torch.tensor(6.0, device=a_tensor.device)
                five_tensor = torch.tensor(5.0, device=a_tensor.device)
                two_tensor = torch.tensor(2.0, device=a_tensor.device)
                term1 = torch.subtract(torch.multiply(six_tensor, a_tensor), five_tensor)
                term2 = torch.subtract(torch.multiply(two_tensor, a_tensor), one_tensor)
                term3 = torch.subtract(torch.multiply(six_tensor, a_tensor), one_tensor)
                Pab = torch.multiply(torch.multiply(term1, term2), term3)
                Qab = torch.multiply(torch.pow(a_tensor, 3), C3_OVER_24)
            
            base_term = torch.tensor(13591409.0, device=a_tensor.device)
            multiplier = torch.tensor(545140134.0, device=a_tensor.device)
            term = torch.add(base_term, torch.multiply(multiplier, a_tensor))
            Tab = torch.multiply(Pab, term)
            
            # Check if a is odd
            two_tensor = torch.tensor(2.0, device=a_tensor.device)
            one_tensor = torch.tensor(1.0, device=a_tensor.device)
            remainder = torch.remainder(a_tensor, two_tensor)
            is_odd = torch.eq(remainder, one_tensor)
            
            # If a is odd, negate Tab
            Tab = torch.where(is_odd, torch.negative(Tab), Tab)
            
            return Pab, Qab, Tab
        
        # Recursive case
        two_tensor = torch.tensor(2.0, device=a_tensor.device)
        m = torch.div(torch.add(a_tensor, b_tensor), two_tensor)
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
    precision_tensor = convert_to_tensor(precision_digits)
    terms_float = torch.div(precision_tensor, DIGITS_PER_TERM)
    one_tensor = torch.tensor(1.0, device=terms_float.device)
    terms_float = torch.add(terms_float, one_tensor)
    terms = torch.floor(terms_float)  # Convert to integer
    terms_int = terms.to(torch.int32)
    
    # Compute the binary split
    P, Q, T = binary_split(0, terms_int)
    
    # Calculate pi
    sqrt_10005 = torch.sqrt(torch.tensor(10005.0, device=Q.device))
    numerator = torch.multiply(Q, torch.tensor(426880.0, device=Q.device))
    numerator = torch.multiply(numerator, sqrt_10005)
    pi_approx = torch.div(numerator, T)
    
    # Return as PyTorch tensor with shape (1,)
    return pi_approx.reshape(1)

# Calculate pi with appropriate precision for PyTorch (float32)
# Ensure it's a scalar with shape (1,) as per PyTorch conventions
pi = torch.tensor([_calculate_pi_value(15)], dtype=torch.float32)  # Increased precision to match reference value

class TorchMathOps:
    """PyTorch implementation of math operations."""
    
    # Reference the module-level pi
    pi = pi
    
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
    
    def max(self, x, axis=None, keepdims=False):
        """Compute the maximum of a tensor along specified axes."""
        return max(x, axis=axis, keepdims=keepdims)
    
    def min(self, x, axis=None, keepdims=False):
        """Compute the minimum of a tensor along specified axes."""
        return min(x, axis=axis, keepdims=keepdims)
    
    def exp(self, x):
        """Compute the exponential of a tensor."""
        return exp(x)
    
    def log(self, x):
        """Compute the natural logarithm of a tensor."""
        return log(x)
    
    def log10(self, x):
        """Compute the base-10 logarithm of a tensor."""
        return log10(x)
    
    def log2(self, x):
        """Compute the base-2 logarithm of a tensor."""
        return log2(x)
    
    def pow(self, x, y):
        """Compute x raised to the power of y."""
        return pow(x, y)
    
    def sqrt(self, x):
        """Compute the square root of a tensor."""
        return sqrt(x)
    
    def square(self, x):
        """Compute the square of a tensor."""
        return square(x)
    
    def abs(self, x):
        """Compute the absolute value of a tensor."""
        return abs(x)
    
    def negative(self, x):
        """Compute the negative of a tensor element-wise."""
        return negative(x)
    
    def sign(self, x):
        """Compute the sign of a tensor."""
        return sign(x)
    
    def sin(self, x):
        """Compute the sine of a tensor."""
        return sin(x)
    
    def cos(self, x):
        """Compute the cosine of a tensor."""
        return cos(x)
    
    def tan(self, x):
        """Compute the tangent of a tensor."""
        return tan(x)
    
    def sinh(self, x):
        """Compute the hyperbolic sine of a tensor."""
        return sinh(x)
    
    def cosh(self, x):
        """Compute the hyperbolic cosine of a tensor."""
        return cosh(x)
    
    def tanh(self, x):
        """Compute the hyperbolic tangent of a tensor."""
        return tanh(x)
    
    def sigmoid(self, x):
        """Compute the sigmoid of a tensor."""
        return sigmoid(x)
    
    def softplus(self, x, beta=1.0, threshold=20.0):
        """Compute the softplus of a tensor."""
        return softplus(x, beta=beta, threshold=threshold)
    
    def relu(self, x):
        """Compute the rectified linear unit of a tensor."""
        return relu(x)
    
    def softmax(self, x, axis=-1):
        """Compute the softmax of a tensor."""
        return softmax(x, axis=axis)
    
    def clip(self, x, min_val=None, max_val=None):
        """Clip the values of a tensor."""
        return clip(x, min_val=min_val, max_val=max_val)
    
    def var(self, x, axis=None, keepdims=False):
        """Compute the variance of a tensor."""
        return var(x, axis=axis, keepdims=keepdims)
    
    def mod(self, x, y):
        """Compute the remainder of division of x by y element-wise."""
        return mod(x, y)
    
    def floor_divide(self, x, y):
        """Element-wise integer division."""
        return floor_divide(x, y)
    
    def sort(self, x, axis=-1):
        """Sort a tensor along a specified axis."""
        return sort(x, axis=axis)
        
    def gradient(self, f, *varargs, axis=None, edge_order: Literal[1, 2] = 1):
        """Return the gradient of an N-dimensional tensor."""
        if edge_order not in (1, 2):
            raise ValueError("edge_order must be 1 or 2")
        return gradient(f, *varargs, axis=axis, edge_order=edge_order)
    
    def cumsum(self, x: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
        """
        Compute the cumulative sum of a tensor along a specified axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to compute the cumulative sum
            
        Returns:
            Tensor with cumulative sums
        """
        x_tensor = convert_to_tensor(x)
        return torch.cumsum(x_tensor, dim=axis)
    
    def eigh(self, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.
        
        Args:
            a: Input Hermitian or symmetric matrix
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        a_tensor = convert_to_tensor(a)
        return torch.linalg.eigh(a_tensor)
