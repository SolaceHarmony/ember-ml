"""
Ember math operations for ember_ml.

This module provides Ember implementations of math operations.
"""

from typing import Any, Union, Sequence
from ember_ml.backend.ember.tensor_ops import EmberBackendTensor

# Type aliases
ArrayLike = Union[EmberBackendTensor, float, int, list, tuple]
Shape = Union[int, Sequence[int]]


def convert_to_tensor(x: Any) -> EmberBackendTensor:
    """
    Convert input to an EmberTensor.
    
    Args:
        x: Input data
        
    Returns:
        EmberTensor representation of the input
    """
    from ember_ml.backend.ember.tensor_ops import convert_to_tensor as _convert_to_tensor
    return _convert_to_tensor(x)


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
    C = 640320
    C3_OVER_24 = (C**3) / 24
    DIGITS_PER_TERM = 14.1816474627254776555  # Approx. digits per iteration
    
    # Number of terms needed for the desired precision
    # Use float division to avoid precision loss
    terms = precision_digits / DIGITS_PER_TERM + 1
    # Convert to integer using divmod instead of //
    terms, _ = divmod(precision_digits, DIGITS_PER_TERM)
    terms = terms + 1
    
    # Implementation of the Chudnovsky algorithm
    sum_value = 0.0
    # Use a while loop instead of range to avoid the float issue
    k = 0
    while k < terms:
        # Calculate numerator: (6k)! * (13591409 + 545140134k)
        numerator = 1
        for i in range(1, 6*k + 1):
            numerator *= i
        numerator *= (13591409 + 545140134*k)
        
        # Calculate denominator: (3k)! * (k!)^3 * (-640320)^(3k)
        denominator = 1
        for i in range(1, 3*k + 1):
            denominator *= i
        for i in range(1, k + 1):
            denominator *= i**3
        denominator *= (-C)**(3*k)
        
        # Add term to sum
        sum_value += numerator / denominator
        
        # Increment k
        k += 1
    
    # Calculate pi using the formula
    pi_value = (426880 * (10005**0.5)) / sum_value
    
    return pi_value
    
# Calculate pi with appropriate precision (15 digits, matching MLX)
pi = EmberBackendTensor([_calculate_pi_value(15)])


def add(x: ArrayLike, y: ArrayLike) -> EmberBackendTensor:
    """
    Add two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise sum
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    # Use the __add__ operator method which will call the add method
    return x_tensor.__add__(y_tensor)


def subtract(x: ArrayLike, y: ArrayLike) -> EmberBackendTensor:
    """
    Subtract two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise difference
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    # Use the __sub__ operator method which will call the subtract method
    return x_tensor.__sub__(y_tensor)


def multiply(x: ArrayLike, y: ArrayLike) -> EmberBackendTensor:
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
    # Use the __mul__ operator method which will call the multiply method
    return x_tensor.__mul__(y_tensor)


def divide(x: ArrayLike, y: ArrayLike) -> EmberBackendTensor:
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
    # Use the __truediv__ operator method which will call the divide method
    return x_tensor.__truediv__(y_tensor)


class EmberBackendMathOps:
    """Ember backend implementation of math operations."""
    
    def __init__(self):
        """Initialize Ember math operations."""
        pass
    
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
