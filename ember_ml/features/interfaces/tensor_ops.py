"""
Tensor operations interfaces.

This module defines the abstract interfaces for tensor operations used in feature extraction.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Sequence, List, Tuple


class TensorOpsInterface(ABC):
    """Abstract interface for tensor operations used in feature extraction."""
    
    @abstractmethod
    def one_hot(
        self,
        indices: Any,
        num_classes: int,
        *,
        axis: int = -1,
        dtype: Any = None
    ) -> Any:
        """
        Create a one-hot tensor.
        
        Args:
            indices: A tensor of indices.
            num_classes: The number of classes.
            axis: The axis to place the one-hot dimension.
            dtype: The data type of the output tensor.
            
        Returns:
            A tensor with one-hot encoding.
        """
        pass
    
    @abstractmethod
    def scatter(
        self,
        tensor: Any,
        indices: Any,
        updates: Any,
        *,
        axis: int = 0
    ) -> Any:
        """
        Scatter updates into a tensor according to indices.
        
        Args:
            tensor: The tensor to update.
            indices: The indices where updates will be scattered.
            updates: The values to scatter.
            axis: The axis along which to scatter.
            
        Returns:
            The updated tensor.
        """
        pass


__all__ = [
    'TensorOpsInterface',
]