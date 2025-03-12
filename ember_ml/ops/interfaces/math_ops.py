"""
Mathematical operations interface.

This module defines the abstract interface for mathematical operations.
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Optional, Any, Tuple

# Type aliases
Shape = Union[int, Sequence[int]]

class MathOps(ABC):
    """Abstract interface for mathematical operations."""
    
    @property
    @abstractmethod
    def pi(self) -> float:
        """
        Return the mathematical constant pi.
        
        Returns:
            The value of pi
        """
        pass
    
    @abstractmethod
    def add(self, x: Any, y: Any) -> Any:
        """
        Add two tensors element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Element-wise sum
        """
        pass
    
    @abstractmethod
    def subtract(self, x: Any, y: Any) -> Any:
        """
        Subtract two tensors element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Element-wise difference
        """
        pass
    
    @abstractmethod
    def multiply(self, x: Any, y: Any) -> Any:
        """
        Multiply two tensors element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Element-wise product
        """
        pass
    
    @abstractmethod
    def divide(self, x: Any, y: Any) -> Any:
        """
        Divide two tensors element-wise.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Element-wise quotient
        """
        pass
    
    @abstractmethod
    def floor_divide(self, x: Any, y: Any) -> Any:
        """
        Element-wise integer division.
        
        If either array is a floating point type then it is equivalent to calling floor() after divide().
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Element-wise integer quotient (a // b)
        """
        pass
    
    @abstractmethod
    def dot(self, x: Any, y: Any) -> Any:
        """
        Compute the dot product of two tensors.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Dot product
        """
        pass
    
    @abstractmethod
    def matmul(self, x: Any, y: Any) -> Any:
        """
        Compute the matrix product of two tensors.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Matrix product
        """
        pass
    
    @abstractmethod
    def mean(self, x: Any, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Any:
        """
        Compute the mean of a tensor along specified axes.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the mean
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Mean of the tensor
        """
        pass
    
    @abstractmethod
    def sum(self, x: Any, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Any:
        """
        Compute the sum of a tensor along specified axes.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the sum
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Sum of the tensor
        """
        pass
    
    @abstractmethod
    def max(self, x: Any, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Any:
        """
        Compute the maximum of a tensor along specified axes.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the maximum
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Maximum of the tensor
        """
        pass
    
    @abstractmethod
    def min(self, x: Any, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Any:
        """
        Compute the minimum of a tensor along specified axes.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the minimum
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Minimum of the tensor
        """
        pass
    
    @abstractmethod
    def exp(self, x: Any) -> Any:
        """
        Compute the exponential of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Element-wise exponential
        """
        pass
    
    @abstractmethod
    def log(self, x: Any) -> Any:
        """
        Compute the natural logarithm of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Element-wise logarithm
        """
        pass
    
    @abstractmethod
    def log10(self, x: Any) -> Any:
        """
        Compute the base-10 logarithm of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Element-wise base-10 logarithm
        """
        pass
    
    @abstractmethod
    def log2(self, x: Any) -> Any:
        """
        Compute the base-2 logarithm of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Element-wise base-2 logarithm
        """
        pass
    
    @abstractmethod
    def pow(self, x: Any, y: Any) -> Any:
        """
        Compute x raised to the power of y element-wise.
        
        Args:
            x: Base tensor
            y: Exponent tensor
            
        Returns:
            Element-wise power
        """
        pass
    
    @abstractmethod
    def sqrt(self, x: Any) -> Any:
        """
        Compute the square root of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Element-wise square root
        """
        pass
    
    @abstractmethod
    def square(self, x: Any) -> Any:
        """
        Compute the square of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Element-wise square
        """
        pass
    
    @abstractmethod
    def abs(self, x: Any) -> Any:
        """
        Compute the absolute value of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Element-wise absolute value
        """
        pass
    
    @abstractmethod
    def negative(self, x: Any) -> Any:
        """
        Compute the negative of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Element-wise negative
        """
        pass
    
    @abstractmethod
    def sign(self, x: Any) -> Any:
        """
        Compute the sign of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Element-wise sign
        """
        pass
    
    @abstractmethod
    def clip(self, x: Any, min_val: Union[float, Any], max_val: Union[float, Any]) -> Any:
        """
        Clip the values of a tensor to a specified range.
        
        Args:
            x: Input tensor
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Clipped tensor
        """
        pass
    
    @abstractmethod
    def sin(self, x: Any) -> Any:
        """
        Compute the sine of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Element-wise sine
        """
        pass
    
    @abstractmethod
    def cos(self, x: Any) -> Any:
        """
        Compute the cosine of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Element-wise cosine
        """
        pass
    
    @abstractmethod
    def tan(self, x: Any) -> Any:
        """
        Compute the tangent of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Element-wise tangent
        """
        pass
    
    @abstractmethod
    def sinh(self, x: Any) -> Any:
        """
        Compute the hyperbolic sine of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Element-wise hyperbolic sine
        """
        pass
    
    @abstractmethod
    def cosh(self, x: Any) -> Any:
        """
        Compute the hyperbolic cosine of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Element-wise hyperbolic cosine
        """
        pass
    
    @abstractmethod
    def tanh(self, x: Any) -> Any:
        """
        Compute the hyperbolic tangent of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Element-wise hyperbolic tangent
        """
        pass
    
    @abstractmethod
    def sigmoid(self, x: Any) -> Any:
        """
        Compute the sigmoid of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Element-wise sigmoid
        """
        pass
    
    @abstractmethod
    def softplus(self, x: Any) -> Any:
        """
        Compute the softplus of a tensor element-wise.
        
        The softplus function is defined as log(1 + exp(x)).
        
        Args:
            x: Input tensor
            
        Returns:
            Element-wise softplus
        """
        pass
    
    @abstractmethod
    def relu(self, x: Any) -> Any:
        """
        Compute the rectified linear unit of a tensor element-wise.
        
        Args:
            x: Input tensor
            
        Returns:
            Element-wise ReLU
        """
        pass
    
    @abstractmethod
    def softmax(self, x: Any, axis: int = -1) -> Any:
        """
        Compute the softmax of a tensor along a specified axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to compute the softmax
            
        Returns:
            Softmax of the tensor
        """
        pass
    
    @abstractmethod
    def gradient(self, f: Any, *varargs, axis: Optional[Union[int, Sequence[int]]] = None,
                edge_order: int = 1) -> Union[Any, Sequence[Any]]:
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
        pass
    
    @abstractmethod
    def cumsum(self, x: Any, axis: Optional[int] = None) -> Any:
        """
        Compute the cumulative sum of a tensor along a specified axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to compute the cumulative sum
            
        Returns:
            Tensor with cumulative sums
        """
        pass
    
    @abstractmethod
    def eigh(self, a: Any) -> Tuple[Any, Any]:
        """
        Compute the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.
        
        Args:
            a: Input Hermitian or symmetric matrix
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        pass

    @abstractmethod
    def mod(self, x: Any, y: Any) -> Any:
        """
        Compute the remainder of division of x by y element-wise.
        
        Args:
            x: Input tensor (dividend)
            y: Input tensor (divisor)
            
        Returns:
            Element-wise remainder
        """
        pass

    @abstractmethod
    def sort(self, x: Any, axis: int = -1) -> Any:
        """
        Sort a tensor along a specified axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to sort
            
        Returns:
            Sorted tensor
        """
        pass
