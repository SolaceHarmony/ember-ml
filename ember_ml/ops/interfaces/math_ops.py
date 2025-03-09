"""
Mathematical operations interface.

This module defines the abstract interface for mathematical operations.
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Optional, Any

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