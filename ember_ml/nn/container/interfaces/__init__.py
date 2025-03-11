"""
Container module interfaces.

This module defines the abstract interfaces for container operations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, List, Tuple, Callable

class DenseInterface(ABC):
    """Abstract interface for Dense (fully connected) layer."""
    
    @abstractmethod
    def forward(
        self,
        x: Any,
    ) -> Any:
        """
        Forward pass through the layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass

class DropoutInterface(ABC):
    """Abstract interface for Dropout layer."""
    
    @abstractmethod
    def forward(
        self,
        x: Any,
        training: bool = False,
    ) -> Any:
        """
        Forward pass through the layer.
        
        Args:
            x: Input tensor
            training: Whether to apply dropout (True) or return the input unchanged (False)
            
        Returns:
            Output tensor with dropout applied (if training is True)
        """
        pass

class BatchNormalizationInterface(ABC):
    """Abstract interface for Batch Normalization layer."""
    
    @abstractmethod
    def forward(
        self,
        x: Any,
        training: bool = False,
    ) -> Any:
        """
        Forward pass through the layer.
        
        Args:
            x: Input tensor
            training: Whether to use the moving statistics (False) or compute new statistics (True)
            
        Returns:
            Normalized output tensor
        """
        pass

class SequentialInterface(ABC):
    """Abstract interface for Sequential container."""
    
    @abstractmethod
    def forward(
        self,
        x: Any,
    ) -> Any:
        """
        Forward pass through the container.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def add(
        self,
        layer: Any,
    ) -> None:
        """
        Add a layer to the container.
        
        Args:
            layer: Layer to add
        """
        pass

__all__ = [
    'DenseInterface',
    'DropoutInterface',
    'BatchNormalizationInterface',
    'SequentialInterface',
]