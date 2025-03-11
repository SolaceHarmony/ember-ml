"""
MLX implementation of math operations.

This module provides MLX implementations of math operations.
"""

import mlx.core as mx
from typing import Union, Sequence, Optional, Any, List, Literal
from ember_ml.backend.mlx.tensor_ops import convert_to_tensor, cast

# Type aliases
ArrayLike = Union[mx.array, float, int, list, tuple]
Shape = Union[int, Sequence[int]]
DType = Any

def add(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Add two MLX arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Element-wise sum
    """
    return mx.add(convert_to_tensor(x), convert_to_tensor(y))

def subtract(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Subtract two MLX arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Element-wise difference
    """
    return mx.subtract(convert_to_tensor(x), convert_to_tensor(y))

def multiply(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Multiply two MLX arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Element-wise product
    """
    return mx.multiply(convert_to_tensor(x), convert_to_tensor(y))

def divide(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Divide two MLX arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Element-wise quotient
    """
    return mx.divide(convert_to_tensor(x), convert_to_tensor(y))

def dot(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Compute the dot product of two MLX arrays.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Dot product
    """
    x_array = convert_to_tensor(x)
    y_array = convert_to_tensor(y)
    
    # Handle different dimensions
    if mx.equal(mx.array(len(x_array.shape)), mx.array(1)) and mx.equal(mx.array(len(y_array.shape)), mx.array(1)):
        return mx.sum(mx.multiply(x_array, y_array))
    else:
        return mx.matmul(x_array, y_array)

def matmul(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Compute the matrix product of two MLX arrays.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Matrix product
    """
    return mx.matmul(convert_to_tensor(x), convert_to_tensor(y))

def mean(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> mx.array:
    """
    Compute the mean of an MLX array along specified axes.
    
    Args:
        x: Input array
        axis: Axis or axes along which to compute the mean
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Mean of the array
    """
    x_array = convert_to_tensor(x)
    
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

def sum(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> mx.array:
    """
    Compute the sum of an MLX array along specified axes.
    
    Args:
        x: Input array
        axis: Axis or axes along which to compute the sum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Sum of the array
    """
    x_array = convert_to_tensor(x)
    
    if axis is None:
        return mx.sum(x_array, keepdims=keepdims)
    
    if isinstance(axis, (list, tuple)):
        # MLX doesn't support multiple axes directly, so we need to do it sequentially
        result = x_array
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = mx.sum(result, axis=ax, keepdims=keepdims)
        return result
    
    return mx.sum(x_array, axis=axis, keepdims=keepdims)

def var(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> mx.array:
    """
    Compute the variance of an MLX array along specified axes.
    
    Args:
        x: Input array
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Variance of the array
    """
    x_array = convert_to_tensor(x)
    
    if axis is None:
        return mx.var(x_array, keepdims=keepdims)
    
    if isinstance(axis, (list, tuple)):
        # MLX doesn't support multiple axes directly, so we need to do it sequentially
        result = x_array
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = mx.var(result, axis=ax, keepdims=keepdims)
        return result
    
    return mx.var(x_array, axis=axis, keepdims=keepdims)

def exp(x: ArrayLike) -> mx.array:
    """
    Compute the exponential of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise exponential
    """
    return mx.exp(convert_to_tensor(x))

def log(x: ArrayLike) -> mx.array:
    """
    Compute the natural logarithm of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise logarithm
    """
    return mx.log(convert_to_tensor(x))

def pow(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Compute x raised to the power of y element-wise.
    
    Args:
        x: Base array
        y: Exponent array
        
    Returns:
        Element-wise power
    """
    return mx.power(convert_to_tensor(x), convert_to_tensor(y))

def sqrt(x: ArrayLike) -> mx.array:
    """
    Compute the square root of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise square root
    """
    return mx.sqrt(convert_to_tensor(x))

def clip(x: ArrayLike, min_val: Union[float, ArrayLike], max_val: Union[float, ArrayLike]) -> mx.array:
    """
    Clip the values of an MLX array to a specified range.
    
    Args:
        x: Input array
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clipped array
    """
    x_array = convert_to_tensor(x)
    min_val = convert_to_tensor(min_val)
    max_val = convert_to_tensor(max_val)
    
    return mx.clip(x_array, min_val, max_val)

def sigmoid(x: ArrayLike) -> mx.array:
    """
    Compute the sigmoid of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise sigmoid
    """
    x_array = convert_to_tensor(x)
    x_safe = clip(x_array, -88.0, 88.0)  # Prevent overflow
    return mx.sigmoid(x_safe)

def softplus(x: ArrayLike) -> mx.array:
    """
    Compute the softplus of an MLX array element-wise.
    
    The softplus function is defined as log(1 + exp(x)).
    
    Args:
        x: Input array
        
    Returns:
        Element-wise softplus
    """
    x_array = convert_to_tensor(x)
    x_safe = clip(x_array, -88.0, 88.0)  # Prevent overflow
    # softplus(x) = log(1 + exp(x))
    return mx.log(mx.add(1.0, mx.exp(x_safe)))

def tanh(x: ArrayLike) -> mx.array:
    """
    Compute the hyperbolic tangent of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise tanh
    """
    return mx.tanh(convert_to_tensor(x))

def relu(x: ArrayLike) -> mx.array:
    """
    Compute the rectified linear unit of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise ReLU
    """
    return mx.maximum(0, convert_to_tensor(x))


def abs(x: ArrayLike) -> mx.array:
    """
    Compute the absolute value of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise absolute value
    """
    return mx.abs(convert_to_tensor(x))


def sign(x: ArrayLike) -> mx.array:
    """
    Compute the sign of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise sign (-1 for negative, 0 for zero, 1 for positive)
    """
    x_tensor = convert_to_tensor(x)
    # Compute sign using comparisons
    positive = mx.array(1.0, dtype=x_tensor.dtype)
    negative = mx.array(-1.0, dtype=x_tensor.dtype)
    zero = mx.array(0.0, dtype=x_tensor.dtype)
    
    # x > 0 -> 1, x < 0 -> -1, else 0
    return mx.where(x_tensor > 0, positive, mx.where(x_tensor < 0, negative, zero))


def sin(x: ArrayLike) -> mx.array:
    """
    Compute the sine of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise sine
    """
    return mx.sin(convert_to_tensor(x))


def cos(x: ArrayLike) -> mx.array:
    """
    Compute the cosine of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise cosine
    """
    return mx.cos(convert_to_tensor(x))


def tan(x: ArrayLike) -> mx.array:
    """
    Compute the tangent of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise tangent
    """
    x_tensor = convert_to_tensor(x)
    # tan(x) = sin(x) / cos(x)
    return mx.divide(mx.sin(x_tensor), mx.cos(x_tensor))


def sinh(x: ArrayLike) -> mx.array:
    """
    Compute the hyperbolic sine of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise hyperbolic sine
    """
    x_tensor = convert_to_tensor(x)
    # sinh(x) = (exp(x) - exp(-x)) / 2
    return mx.divide(mx.subtract(mx.exp(x_tensor), mx.exp(mx.negative(x_tensor))), 2.0)


def cosh(x: ArrayLike) -> mx.array:
    """
    Compute the hyperbolic cosine of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise hyperbolic cosine
    """
    x_tensor = convert_to_tensor(x)
    # cosh(x) = (exp(x) + exp(-x)) / 2
    return mx.divide(mx.add(mx.exp(x_tensor), mx.exp(mx.negative(x_tensor))), 2.0)


def log10(x: ArrayLike) -> mx.array:
    """
    Compute the base-10 logarithm of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise base-10 logarithm
    """
    x_tensor = convert_to_tensor(x)
    # log10(x) = log(x) / log(10)
    return mx.divide(mx.log(x_tensor), mx.log(mx.array(10.0)))


def log2(x: ArrayLike) -> mx.array:
    """
    Compute the base-2 logarithm of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise base-2 logarithm
    """
    x_tensor = convert_to_tensor(x)
    # log2(x) = log(x) / log(2)
    return mx.divide(mx.log(x_tensor), mx.log(mx.array(2.0)))


def square(x: ArrayLike) -> mx.array:
    """
    Compute the square of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Element-wise square
    """
    x_tensor = convert_to_tensor(x)
    return mx.multiply(x_tensor, x_tensor)


def mod(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Compute the remainder of division of x by y element-wise.
    
    Args:
        x: Input array (dividend)
        y: Input array (divisor)
        
    Returns:
        Element-wise remainder
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    
    # Use divmod to get the remainder
    _, remainder = mx.divmod(x_tensor, y_tensor)
    return remainder


def floor_divide(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Element-wise integer division.
    
    If either array is a floating point type then it is equivalent to calling floor() after divide().
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Element-wise integer quotient (a // b)
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    
    # Use floor_divide from MLX
    return mx.floor_divide(x_tensor, y_tensor)


def min(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> mx.array:
    """
    Compute the minimum value of an MLX array along specified axes.
    
    Args:
        x: Input array
        axis: Axis or axes along which to compute the minimum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Minimum value of the array
    """
    x_array = convert_to_tensor(x)
    
    if axis is None:
        # Flatten the array and find the minimum
        x_flat = mx.reshape(x_array, (-1,))
        return mx.min(x_flat)
    
    if isinstance(axis, (list, tuple)):
        # MLX doesn't support multiple axes directly, so we need to do it sequentially
        result = x_array
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = mx.min(result, axis=ax, keepdims=keepdims)
        return result
    
    return mx.min(x_array, axis=axis, keepdims=keepdims)


def max(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> mx.array:
    """
    Compute the maximum value of an MLX array along specified axes.
    
    Args:
        x: Input array
        axis: Axis or axes along which to compute the maximum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Maximum value of the array
    """
    x_array = convert_to_tensor(x)
    
    if axis is None:
        # Flatten the array and find the maximum
        x_flat = mx.reshape(x_array, (-1,))
        return mx.max(x_flat)
    
    if isinstance(axis, (list, tuple)):
        # MLX doesn't support multiple axes directly, so we need to do it sequentially
        result = x_array
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = mx.max(result, axis=ax, keepdims=keepdims)
        return result
    
    return mx.max(x_array, axis=axis, keepdims=keepdims)

def softmax(x: ArrayLike, axis: int = -1) -> mx.array:
    """
    Compute the softmax of an MLX array along a specified axis.
    
    Args:
        x: Input array
        axis: Axis along which to compute the softmax
        
    Returns:
        Softmax of the array
    """
    x_array = convert_to_tensor(x)
    x_max = mx.max(x_array, axis=axis, keepdims=True)
    exp_x = mx.exp(mx.subtract(x_array, x_max))
    return mx.divide(exp_x, mx.sum(exp_x, axis=axis, keepdims=True))

def sort(x: ArrayLike, axis: int = -1) -> mx.array:
    """
    Sort an MLX array along a specified axis.
    
    Args:
        x: Input array
        axis: Axis along which to sort
        
    Returns:
        Sorted array
    """
    x_array = convert_to_tensor(x)
    return mx.sort(x_array, axis=axis)


def gradient(f: ArrayLike, *varargs, axis: Optional[Union[int, Sequence[int]]] = None,
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
    import numpy as np
    
    f_array = convert_to_tensor(f)
    
    # Convert to NumPy array for calculation
    f_numpy = f_array.tolist()
    f_numpy = np.array(f_numpy)
    
    # Process spacing arguments
    spacing_args = []
    for arg in varargs:
        if isinstance(arg, mx.array):
            arg_numpy = arg.tolist()
            spacing_args.append(np.array(arg_numpy))
        else:
            spacing_args.append(arg)
    
    # Calculate gradient using NumPy's gradient function
    result_numpy = np.gradient(f_numpy, *spacing_args, axis=axis, edge_order=edge_order)
    
    # Convert back to MLX array
    if isinstance(result_numpy, np.ndarray):
        return mx.array(result_numpy)
    else:
        return [mx.array(arr) for arr in result_numpy]

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
    C = mx.array(640320)
    C3_OVER_24 = mx.divide(mx.power(C, 3), mx.array(24))
    DIGITS_PER_TERM = mx.array(14.1816474627254776555)  # Approx. digits per iteration
    
    def binary_split(a, b):
        """Recursive binary split for the Chudnovsky algorithm."""
        a_tensor = convert_to_tensor(a)
        b_tensor = convert_to_tensor(b)
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
    precision_tensor = convert_to_tensor(precision_digits)
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


def pi() -> mx.array:
    """
    Return the mathematical constant pi calculated using the Chudnovsky algorithm.
    
    This implementation uses the Chudnovsky algorithm, which is one of the most
    efficient algorithms for calculating π. The value is calculated with precision
    appropriate for MLX's float32 data type and returned as a scalar array with
    shape (1,) as per MLX conventions.
    
    Returns:
        MLX array containing the value of pi with shape (1,)
    """
    # Return pi as a scalar with shape (1,) as per MLX conventions
    return PI_CONSTANT


class MLXMathOps:
    """MLX implementation of math operations."""
    
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
    
    @property
    def pi(self):
        """Return the mathematical constant pi."""
        return pi
    
    def pi_func(self):
        """Return the mathematical constant pi as a function."""
        return pi()