"""
NumPy math operations for ember_ml.

This module provides NumPy implementations of math operations.
"""

import numpy as np
from typing import Optional, Union, Sequence, List, Literal, Tuple
from ember_ml.backend.numpy.tensor.tensor import NumpyTensor

# Create a tensor instance for convert_to_tensor
_tensor_ops = NumpyTensor()
convert_to_tensor = _tensor_ops.convert_to_tensor

# Type aliases
ArrayLike = Union[np.ndarray, float, int, list, tuple]
Shape = Union[int, Sequence[int]]
DType = Union[np.dtype, str, None]


def add(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Add two NumPy arrays element-wise.

    Args:
        x: First array
        y: Second array

    Returns:
        Element-wise sum
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return np.add(x_tensor, y_tensor)


def subtract(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Subtract two NumPy arrays element-wise.

    Args:
        x: First array
        y: Second array

    Returns:
        Element-wise difference
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return np.subtract(x_tensor, y_tensor)


def multiply(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Multiply two NumPy arrays element-wise.

    Args:
        x: First array
        y: Second array

    Returns:
        Element-wise product
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return np.multiply(x_tensor, y_tensor)


def divide(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Divide two NumPy arrays element-wise.

    Args:
        x: First array
        y: Second array

    Returns:
        Element-wise quotient
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return np.divide(x_tensor, y_tensor)


def dot(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Compute the dot product of two NumPy arrays.

    Args:
        x: First array
        y: Second array

    Returns:
        Dot product
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return np.dot(x_tensor, y_tensor)


def matmul(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Compute the matrix product of two NumPy arrays.

    Args:
        x: First array
        y: Second array

    Returns:
        Matrix product
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return np.matmul(x_tensor, y_tensor)


def mean(x: ArrayLike,
          axis: Optional[Union[int, Sequence[int]]] = None,
          keepdims: bool = False) -> np.ndarray:
    """
    Compute the mean of a NumPy array along specified axes.

    Args:
        x: Input array
        axis: Axis or axes along which to compute the mean
        keepdims: Whether to keep the reduced dimensions

    Returns:
        Mean of the array
    """
    x_tensor = convert_to_tensor(x)
    return np.mean(x_tensor, axis=axis, keepdims=keepdims)


def sum(x: ArrayLike,
         axis: Optional[Union[int, Sequence[int]]] = None,
         keepdims: bool = False) -> np.ndarray:
    """
    Compute the sum of a NumPy array along specified axes.

    Args:
        x: Input array
        axis: Axis or axes along which to compute the sum
        keepdims: Whether to keep the reduced dimensions

    Returns:
        Sum of the array
    """
    x_tensor = convert_to_tensor(x)
    return np.sum(x_tensor, axis=axis, keepdims=keepdims)


def var(x: ArrayLike,
         axis: Optional[Union[int, Sequence[int]]] = None,
         keepdims: bool = False) -> np.ndarray:
    """
    Compute the variance of a NumPy array along specified axes.

    Args:
        x: Input array
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the reduced dimensions

    Returns:
        Variance of the array
    """
    x_tensor = convert_to_tensor(x)
    return np.var(x_tensor, axis=axis, keepdims=keepdims)


def exp(x: ArrayLike) -> np.ndarray:
    """
    Compute the exponential of a NumPy array element-wise.

    Args:
        x: Input array

    Returns:
        Element-wise exponential
    """
    x_tensor = convert_to_tensor(x)
    return np.exp(x_tensor)


def log(x: ArrayLike) -> np.ndarray:
    """
    Compute the natural logarithm of a NumPy array element-wise.

    Args:
        x: Input array

    Returns:
        Element-wise logarithm
    """
    x_tensor = convert_to_tensor(x)
    return np.log(x_tensor)


def log10(x: ArrayLike) -> np.ndarray:
    """
    Compute the base-10 logarithm of a NumPy array element-wise.

    Args:
        x: Input array

    Returns:
        Element-wise base-10 logarithm
    """
    x_tensor = convert_to_tensor(x)
    return np.log10(x_tensor)


def log2(x: ArrayLike) -> np.ndarray:
    """
    Compute the base-2 logarithm of a NumPy array element-wise.

    Args:
        x: Input array

    Returns:
        Element-wise base-2 logarithm
    """
    x_tensor = convert_to_tensor(x)
    return np.log2(x_tensor)


def pow(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Compute x raised to the power of y element-wise.

    Args:
        x: Base array
        y: Exponent array

    Returns:
        Element-wise power
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return np.power(x_tensor, y_tensor)


def sqrt(x: ArrayLike) -> np.ndarray:
    """
    Compute the square root of a NumPy array element-wise.

    Args:
        x: Input array

    Returns:
        Element-wise square root
    """
    x_tensor = convert_to_tensor(x)
    return np.sqrt(x_tensor)


def square(x: ArrayLike) -> np.ndarray:
    """
    Compute the square of a NumPy array element-wise.

    Args:
        x: Input array

    Returns:
        Element-wise square
    """
    x_tensor = convert_to_tensor(x)
    return np.square(x_tensor)

def abs(x: ArrayLike) -> np.ndarray:
    """
    Compute the absolute value of a NumPy array element-wise.

    Args:
        x: Input array

    Returns:
        Element-wise absolute value
    """
    x_tensor = convert_to_tensor(x)
    return np.abs(x_tensor)


def negative(x: ArrayLike) -> np.ndarray:
    """
    Compute the negative of a NumPy array element-wise.

    Args:
        x: Input array

    Returns:
        Element-wise negative
    """
    x_tensor = convert_to_tensor(x)
    return np.negative(x_tensor)


def sign(x: ArrayLike) -> np.ndarray:
    """
    Compute the sign of a NumPy array element-wise.

    Args:
        x: Input array

    Returns:
        Element-wise sign
    """
    x_tensor = convert_to_tensor(x)
    return np.sign(x_tensor)


def sin(x: ArrayLike) -> np.ndarray:
    """
    Compute the sine of a NumPy array element-wise.

    Args:
        x: Input array

    Returns:
        Element-wise sine
    """
    x_tensor = convert_to_tensor(x)
    return np.sin(x_tensor)


def cos(x: ArrayLike) -> np.ndarray:
    """
    Compute the cosine of a NumPy array element-wise.

    Args:
        x: Input array

    Returns:
        Element-wise cosine
    """
    x_tensor = convert_to_tensor(x)
    return np.cos(x_tensor)


def tan(x: ArrayLike) -> np.ndarray:
    """
    Compute the tangent of a NumPy array element-wise.

    Args:
        x: Input array

    Returns:
        Element-wise tangent
    """
    x_tensor = convert_to_tensor(x)
    return np.tan(x_tensor)


def sinh(x: ArrayLike) -> np.ndarray:
    """
    Compute the hyperbolic sine of a NumPy array element-wise.

    Args:
        x: Input array

    Returns:
        Element-wise hyperbolic sine
    """
    x_tensor = convert_to_tensor(x)
    return np.sinh(x_tensor)


def cosh(x: ArrayLike) -> np.ndarray:
    """
    Compute the hyperbolic cosine of a NumPy array element-wise.

    Args:
        x: Input array

    Returns:
        Element-wise hyperbolic cosine
    """
    x_tensor = convert_to_tensor(x)
    return np.cosh(x_tensor)


def tanh(x: ArrayLike) -> np.ndarray:
    """
    Compute the hyperbolic tangent of a NumPy array element-wise.

    Args:
        x: Input array

    Returns:
        Element-wise tanh
    """
    x_tensor = convert_to_tensor(x)
    return np.tanh(x_tensor)


def sigmoid(x: ArrayLike) -> np.ndarray:
    """
    Compute the sigmoid of a NumPy array element-wise.

    Args:
        x: Input array

    Returns:
        Element-wise sigmoid
    """
    x_tensor = convert_to_tensor(x)
    x_safe = np.clip(x_tensor, -88.0, 88.0)  # Prevent overflow
    denominator = np.add(1.0, np.exp(np.negative(x_safe)))
    return np.divide(1.0, denominator)


def softplus(x: ArrayLike) -> np.ndarray:
    """
    Compute the softplus of a NumPy array element-wise.
    
    The softplus function is defined as log(1 + exp(x)).

    Args:
        x: Input array

    Returns:
        Element-wise softplus
    """
    x_tensor = convert_to_tensor(x)
    x_safe = np.clip(x_tensor, -88.0, 88.0)  # Prevent overflow
    return np.log(np.add(1.0, np.exp(x_safe)))


def relu(x: ArrayLike) -> np.ndarray:
    """
    Compute the rectified linear unit of a NumPy array element-wise.

    Args:
        x: Input array

    Returns:
        Element-wise ReLU
    """
    x_tensor = convert_to_tensor(x)
    return np.maximum(0, x_tensor)


def mod(x: ArrayLike, y: ArrayLike) -> np.ndarray:
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
    _, remainder = np.divmod(x_tensor, y_tensor)
    return remainder


def floor_divide(x: ArrayLike, y: ArrayLike) -> np.ndarray:
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
    return np.floor_divide(x_tensor, y_tensor)


def gradient(f: ArrayLike, *varargs, axis: Optional[Union[int, Sequence[int]]] = None,
            edge_order: Literal[1, 2] = 1) -> Union[np.ndarray, List[np.ndarray]]:
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
    f_tensor = convert_to_tensor(f)
    return np.gradient(f_tensor, *varargs, axis=axis, edge_order=edge_order)


def softmax(x: ArrayLike, axis: int = -1) -> np.ndarray:
    """
    Compute the softmax of a NumPy array along a specified axis.

    Args:
        x: Input array
        axis: Axis along which to compute the softmax

    Returns:
        Softmax of the array
    """
    x_tensor = convert_to_tensor(x)
    x_max = np.max(x_tensor, axis=axis, keepdims=True)
    exp_x = np.exp(np.subtract(x_tensor, x_max))
    return np.divide(exp_x, np.sum(exp_x, axis=axis, keepdims=True))


def clip(x: ArrayLike, min_val: Union[float, ArrayLike], max_val: Union[float, ArrayLike]) -> np.ndarray:
    """
    Clip the values of a NumPy array to a specified range.

    Args:
        x: Input array
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clipped array
    """
    x_tensor = convert_to_tensor(x)
    min_val_tensor = convert_to_tensor(min_val)
    max_val_tensor = convert_to_tensor(max_val)
    return np.clip(x_tensor, min_val_tensor, max_val_tensor)


def max(x: ArrayLike,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False) -> np.ndarray:
    """
    Compute the maximum of a NumPy array along specified axes.

    Args:
        x: Input array
        axis: Axis or axes along which to compute the maximum
        keepdims: Whether to keep the reduced dimensions

    Returns:
        Maximum of the array
    """
    x_tensor = convert_to_tensor(x)
    return np.max(x_tensor, axis=axis, keepdims=keepdims)


def min(x: ArrayLike,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False) -> np.ndarray:
    """
    Compute the minimum of a NumPy array along specified axes.

    Args:
        x: Input array
        axis: Axis or axes along which to compute the minimum
        keepdims: Whether to keep the reduced dimensions

    Returns:
        Minimum of the array
    """
    x_tensor = convert_to_tensor(x)
    return np.min(x_tensor, axis=axis, keepdims=keepdims)


def cumsum(x: ArrayLike, axis: Optional[int] = None) -> np.ndarray:
    """
    Compute the cumulative sum of a NumPy array along a specified axis.

    Args:
        x: Input array
        axis: Axis along which to compute the cumulative sum

    Returns:
        Array with cumulative sums
    """
    x_tensor = convert_to_tensor(x)
    return np.cumsum(x_tensor, axis=axis)


def eigh(a: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.

    Args:
        a: Input Hermitian or symmetric matrix

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    a_tensor = convert_to_tensor(a)
    return np.linalg.eigh(a_tensor)


def sort(x: ArrayLike, axis: int = -1) -> np.ndarray:
    """
    Sort a NumPy array along a specified axis.

    Args:
        x: Input array
        axis: Axis along which to sort

    Returns:
        Sorted array
    """
    x_tensor = convert_to_tensor(x)
    return np.sort(x_tensor, axis=axis)


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
    C = np.array(640320.0)
    C3_OVER_24 = np.divide(np.power(C, 3), np.array(24.0))
    DIGITS_PER_TERM = np.array(14.1816474627254776555)  # Approx. digits per iteration

    def binary_split(a, b):
        """Recursive binary split for the Chudnovsky algorithm."""
        a_tensor = convert_to_tensor(a)
        b_tensor = convert_to_tensor(b)
        diff = np.subtract(b_tensor, a_tensor)

        if np.array_equal(diff, np.array(1.0)):
            # Base case
            if np.array_equal(a_tensor, np.array(0.0)):
                Pab = np.array(1.0)
                Qab = np.array(1.0)
            else:
                term1 = np.subtract(np.multiply(np.array(6.0), a_tensor), np.array(5.0))
                term2 = np.subtract(np.multiply(np.array(2.0), a_tensor), np.array(1.0))
                term3 = np.subtract(np.multiply(np.array(6.0), a_tensor), np.array(1.0))
                Pab = np.multiply(np.multiply(term1, term2), term3)
                Qab = np.multiply(np.power(a_tensor, 3), C3_OVER_24)

            base_term = np.array(13591409.0)
            multiplier = np.array(545140134.0)
            term = np.add(base_term, np.multiply(multiplier, a_tensor))
            Tab = np.multiply(Pab, term)

            # Check if a is odd
            remainder = np.remainder(a_tensor, np.array(2.0))
            is_odd = np.equal(remainder, np.array(1.0))

            # If a is odd, negate Tab
            Tab = np.where(is_odd, np.negative(Tab), Tab)

            return Pab, Qab, Tab

        # Recursive case
        m = np.divide(np.add(a_tensor, b_tensor), np.array(2.0))
        m = np.floor(m)  # Ensure m is an integer

        Pam, Qam, Tam = binary_split(a, m)
        Pmb, Qmb, Tmb = binary_split(m, b)

        Pab = np.multiply(Pam, Pmb)
        Qab = np.multiply(Qam, Qmb)
        term1 = np.multiply(Qmb, Tam)
        term2 = np.multiply(Pam, Tmb)
        Tab = np.add(term1, term2)

        return Pab, Qab, Tab

    # Number of terms needed for the desired precision
    precision_tensor = convert_to_tensor(precision_digits)
    terms_float = np.divide(precision_tensor, DIGITS_PER_TERM)
    terms_float = np.add(terms_float, np.array(1.0))
    terms = np.floor(terms_float)  # Convert to integer
    terms_int = terms.astype(np.int32)

    # Compute the binary split
    P, Q, T = binary_split(0, terms_int)

    # Calculate pi
    sqrt_10005 = np.sqrt(np.array(10005.0))
    numerator = np.multiply(Q, np.array(426880.0))
    numerator = np.multiply(numerator, sqrt_10005)
    pi_approx = np.divide(numerator, T)

    # Return as NumPy array with shape (1,)
    return pi_approx.reshape(1)

# Calculate pi with appropriate precision for NumPy (float32)
# Ensure it's a scalar with shape (1,) as per NumPy conventions
PI_CONSTANT = _calculate_pi_value(15)  # Increased precision to match reference value


# Alias for pow
power = pow

class NumpyMathOps:
    """NumPy implementation of math operations."""
    
    pi : np.ndarray = np.array([PI_CONSTANT], dtype=np.float32)

    def add(self, x, y):
        """Add two arrays element-wise."""
        return add(x, y)

    def subtract(self, x, y):
        """Subtract two arrays element-wise."""
        return subtract(x, y)

    def multiply(self, x, y):
        """Multiply two arrays element-wise."""
        return multiply(x, y)

    def divide(self, x, y):
        """Divide two arrays element-wise."""
        return divide(x, y)

    def matmul(self, x, y):
        """Compute the matrix product of two arrays."""
        return matmul(x, y)

    def dot(self, x, y):
        """Compute the dot product of two arrays."""
        return dot(x, y)

    def mean(self, x, axis=None, keepdims=False):
        """Compute the mean of an array along specified axes."""
        return mean(x, axis=axis, keepdims=keepdims)

    def sum(self, x, axis=None, keepdims=False):
        """Compute the sum of an array along specified axes."""
        return sum(x, axis=axis, keepdims=keepdims)

    def max(self, x, axis=None, keepdims=False):
        """Compute the maximum of an array along specified axes."""
        return max(x, axis=axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims=False):
        """Compute the minimum of an array along specified axes."""
        return min(x, axis=axis, keepdims=keepdims)

    def exp(self, x):
        """Compute the exponential of an array element-wise."""
        return exp(x)

    def log(self, x):
        """Compute the natural logarithm of an array element-wise."""
        return log(x)

    def log10(self, x):
        """Compute the base-10 logarithm of an array element-wise."""
        return log10(x)

    def log2(self, x):
        """Compute the base-2 logarithm of an array element-wise."""
        return log2(x)

    def pow(self, x, y):
        """Compute x raised to the power of y element-wise."""
        return pow(x, y)

    def sqrt(self, x):
        """Compute the square root of an array element-wise."""
        return sqrt(x)

    def square(self, x):
        """Compute the square of an array element-wise."""
        return square(x)

    def abs(self, x):
        """Compute the absolute value of an array element-wise."""
        return abs(x)
        
    def negative(self, x):
        """Compute the negative of an array element-wise."""
        return negative(x)

    def sign(self, x):
        """Compute the sign of an array element-wise."""
        return sign(x)

    def sin(self, x):
        """Compute the sine of an array element-wise."""
        return sin(x)

    def cos(self, x):
        """Compute the cosine of an array element-wise."""
        return cos(x)

    def tan(self, x):
        """Compute the tangent of an array element-wise."""
        return tan(x)

    def sinh(self, x):
        """Compute the hyperbolic sine of an array element-wise."""
        return sinh(x)

    def cosh(self, x):
        """Compute the hyperbolic cosine of an array element-wise."""
        return cosh(x)

    def tanh(self, x):
        """Compute the hyperbolic tangent of an array element-wise."""
        return tanh(x)

    def sigmoid(self, x):
        """Compute the sigmoid of an array element-wise."""
        return sigmoid(x)
        
    def softplus(self, x):
        """Compute the softplus of an array element-wise."""
        return softplus(x)

    def relu(self, x):
        """Compute the rectified linear unit of an array element-wise."""
        return relu(x)

    def softmax(self, x, axis=-1):
        """Compute the softmax of an array along a specified axis."""
        return softmax(x, axis=axis)

    def clip(self, x, min_val, max_val):
        """Clip the values of an array to a specified range."""
        return clip(x, min_val, max_val)

    def var(self, x, axis=None, keepdims=False):
        """Compute the variance of an array along specified axes."""
        return var(x, axis=axis, keepdims=keepdims)

    def mod(self, x, y):
        """Compute the remainder of division of x by y element-wise."""
        return mod(x, y)
    
    def floor_divide(self, x, y):
        """Element-wise integer division."""
        return floor_divide(x, y)
    
    def gradient(self, f, *varargs, axis=None, edge_order: Literal[1, 2] = 1):
        """Return the gradient of an N-dimensional array."""
        return gradient(f, *varargs, axis=axis, edge_order=edge_order)

    def cumsum(self, x, axis=None):
        """Compute the cumulative sum of an array along a specified axis."""
        return cumsum(x, axis=axis)

    def eigh(self, a):
        """Compute the eigenvalues and eigenvectors of a Hermitian or symmetric matrix."""
        return eigh(a)

    def sort(self, x, axis=-1):
        """Sort an array along a specified axis."""
        return sort(x, axis=axis)

    # Alias for pow
    power = pow
