"""
NumPy implementation of statistical operations.

This module provides NumPy implementations of the statistical operations interface.
"""

import numpy as np
from typing import Union, Sequence, Optional

from ember_ml.backend.numpy.types import TensorLike
from ember_ml.backend.numpy.tensor import NumpyDType

dtype_obj = NumpyDType()

class NumpyStatsOps:
    """NumPy implementation of statistical operations."""
    
    def median(self, x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None, 
              keepdims: bool = False) -> np.ndarray:
        """
        Compute the median along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the median
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Median of the tensor
        """
        from ember_ml.backend.numpy.stats.ops import median as median_func
        return median_func(x, axis=axis, keepdims=keepdims)
    
    def std(self, x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None, 
           keepdims: bool = False, ddof: int = 0) -> np.ndarray:
        """
        Compute the standard deviation along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the standard deviation
            keepdims: Whether to keep the reduced dimensions
            ddof: Delta degrees of freedom
            
        Returns:
            Standard deviation of the tensor
        """
        from ember_ml.backend.numpy.stats.ops import std as std_func
        return std_func(x, axis=axis, keepdims=keepdims, ddof=ddof)
    
    def percentile(self, x: TensorLike, q: Union[float, np.ndarray], 
                  axis: Optional[Union[int, Sequence[int]]] = None, 
                  keepdims: bool = False) -> np.ndarray:
        """
        Compute the q-th percentile along the specified axis.
        
        Args:
            x: Input tensor
            q: Percentile(s) to compute, in range [0, 100]
            axis: Axis or axes along which to compute the percentile
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            q-th percentile of the tensor
        """
        from ember_ml.backend.numpy.stats.ops import percentile as percentile_func
        return percentile_func(x, q, axis=axis, keepdims=keepdims)
    
    def mean(self, x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
            keepdims: bool = False) -> np.ndarray:
        """
        Compute the mean along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the mean
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Mean of the tensor
        """
        from ember_ml.backend.numpy.stats.ops import mean as mean_func
        return mean_func(x, axis=axis, keepdims=keepdims)
    
    def var(self, x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
           keepdims: bool = False, ddof: int = 0) -> np.ndarray:
        """
        Compute the variance along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the variance
            keepdims: Whether to keep the reduced dimensions
            ddof: Delta degrees of freedom
            
        Returns:
            Variance of the tensor
        """
        from ember_ml.backend.numpy.stats.ops import var as var_func
        return var_func(x, axis=axis, keepdims=keepdims, ddof=ddof)
    
    def max(self, x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
           keepdims: bool = False) -> np.ndarray:
        """
        Compute the maximum value along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the maximum
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Maximum value of the tensor
        """
        from ember_ml.backend.numpy.stats.ops import max as max_func
        return max_func(x, axis=axis, keepdims=keepdims)
    
    def min(self, x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
           keepdims: bool = False) -> np.ndarray:
        """
        Compute the minimum value along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the minimum
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Minimum value of the tensor
        """
        from ember_ml.backend.numpy.stats.ops import min as min_func
        return min_func(x, axis=axis, keepdims=keepdims)
    
    def sum(self, x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
           keepdims: bool = False) -> np.ndarray:
        """
        Compute the sum along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the sum
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Sum of the tensor
        """
        from ember_ml.backend.numpy.stats.ops import sum as sum_func
        return sum_func(x, axis=axis, keepdims=keepdims)
    
    def cumsum(self, x: TensorLike, axis: Optional[int] = None) -> np.ndarray:
        """
        Compute the cumulative sum along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to compute the cumulative sum
            
        Returns:
            Cumulative sum of the tensor
        """
        from ember_ml.backend.numpy.stats.ops import cumsum as cumsum_func
        return cumsum_func(x, axis=axis)
    
    def argmax(self, x: TensorLike, axis: Optional[int] = None,
              keepdims: bool = False) -> np.ndarray:
        """
        Returns the indices of the maximum values along an axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to compute the argmax
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Indices of the maximum values
        """
        from ember_ml.backend.numpy.stats.ops import argmax as argmax_func
        return argmax_func(x, axis=axis, keepdims=keepdims)
    
    def sort(self, x: TensorLike, axis: int = -1, descending: bool = False) -> np.ndarray:
        """
        Sort a tensor along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to sort
            descending: Whether to sort in descending order
            
        Returns:
            Sorted tensor
        """
        from ember_ml.backend.numpy.stats.ops import sort as sort_func
        return sort_func(x, axis=axis, descending=descending)
    
    def argsort(self, x: TensorLike, axis: int = -1, descending: bool = False) -> np.ndarray:
        """
        Returns the indices that would sort a tensor along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to sort
            descending: Whether to sort in descending order
            
        Returns:
            Indices that would sort the tensor
        """
        from ember_ml.backend.numpy.stats.ops import argsort as argsort_func
        return argsort_func(x, axis=axis, descending=descending)