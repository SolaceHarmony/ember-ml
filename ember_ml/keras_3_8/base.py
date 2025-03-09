"""
Base classes for Keras-style layer implementations.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Union, List
from abc import ABC, abstractmethod

class Layer(nn.Module, ABC):
    """Base class for all layers."""
    
    def __init__(self):
        """Initialize layer."""
        super().__init__()
        self.built = False
        self.trainable = True
        self._trainable_weights = []
        self._non_trainable_weights = []
        
    @abstractmethod
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Build layer based on input shape.

        Args:
            input_shape: Input tensor shape
        """
        pass
    
    @abstractmethod
    def call(self,
             inputs: torch.Tensor,
             training: bool = False,
             **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through layer.

        Args:
            inputs: Input tensor
            training: Whether in training mode
            **kwargs: Additional arguments

        Returns:
            Output tensor(s)
        """
        pass
    
    def forward(self,
                inputs: torch.Tensor,
                training: bool = False,
                **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through layer.

        Args:
            inputs: Input tensor
            training: Whether in training mode
            **kwargs: Additional arguments

        Returns:
            Output tensor(s)
        """
        # Build layer if needed
        if not self.built:
            self.build(inputs.shape)
            self.built = True
            
        return self.call(inputs, training=training, **kwargs)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration.

        Returns:
            Configuration dictionary
        """
        return {
            'trainable': self.trainable
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Layer':
        """
        Create layer from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Configured layer instance
        """
        return cls(**config)
    
    def get_weights(self) -> List[torch.Tensor]:
        """
        Get layer weights.

        Returns:
            List of weight tensors
        """
        return [p.data for p in self.parameters()]
    
    def set_weights(self, weights: List[torch.Tensor]) -> None:
        """
        Set layer weights.

        Args:
            weights: List of weight tensors
        """
        params = [p for p in self.parameters()]
        if len(params) != len(weights):
            raise ValueError(
                f"Expected {len(params)} weight tensors, got {len(weights)}"
            )
            
        for p, w in zip(params, weights):
            p.data = w
            
    def trainable_weights(self) -> List[torch.Tensor]:
        """
        Get trainable weights.

        Returns:
            List of trainable weight tensors
        """
        return [p.data for p in self.parameters() if p.requires_grad]
    
    def non_trainable_weights(self) -> List[torch.Tensor]:
        """
        Get non-trainable weights.

        Returns:
            List of non-trainable weight tensors
        """
        return [p.data for p in self.parameters() if not p.requires_grad]
    
    def set_trainable(self, trainable: bool) -> None:
        """
        Set layer trainability.

        Args:
            trainable: Whether layer should be trainable
        """
        self.trainable = trainable
        for p in self.parameters():
            p.requires_grad = trainable