"""
Sequential module for ember_ml.

This module provides a backend-agnostic implementation of a sequential container
that works with any backend (NumPy, PyTorch, MLX).
"""

from typing import Optional, Any, List, Dict, Union

from ember_ml import ops
from ember_ml.nn.modules.base_module import BaseModule as Module
from ember_ml.nn.container.interfaces import SequentialInterface

class Sequential(Module, SequentialInterface):
    """
    Sequential container.
    
    A sequential container that applies layers in order.
    
    Attributes:
        layers: List of layers to apply in order
    """
    
    def __init__(self, layers: Optional[List[Any]] = None):
        """
        Initialize a sequential container.
        
        Args:
            layers: Optional list of layers to add to the container
        """
        super().__init__()
        self.layers = []
        
        if layers is not None:
            for layer in layers:
                self.add(layer)
    
    def add(self, layer: Any) -> None:
        """
        Add a layer to the container.
        
        Args:
            layer: Layer to add
        """
        self.layers.append(layer)
    
    def forward(self, x: Any, **kwargs) -> Any:
        """
        Forward pass through the container.
        
        Args:
            x: Input tensor
            **kwargs: Additional keyword arguments to pass to the layers
            
        Returns:
            Output tensor
        """
        for layer in self.layers:
            # Check if the layer has a forward method
            if hasattr(layer, 'forward'):
                x = layer.forward(x, **kwargs)
            # Otherwise, assume it's a callable
            else:
                x = layer(x, **kwargs)
        
        return x
    
    def __repr__(self) -> str:
        """Return a string representation of the container."""
        layer_reprs = [f"  ({i}): {layer}" for i, layer in enumerate(self.layers)]
        return "Sequential(\n" + "\n".join(layer_reprs) + "\n)"