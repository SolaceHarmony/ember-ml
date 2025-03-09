"""
PyTorch backend implementations for loss functions.

This module provides PyTorch-specific implementations of common loss functions
used in neural networks.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Any

from ember_ml.nn.backends.torch_backend import TorchModule

class TorchLoss(TorchModule):
    """
    Base class for all PyTorch loss functions.
    """
    
    def __init__(self, reduction='mean'):
        """
        Initialize a loss function.
        
        Args:
            reduction: Specifies the reduction to apply to the output:
                'none': no reduction will be applied
                'mean': the sum of the output will be divided by the number of elements
                'sum': the output will be summed
        """
        super().__init__()
        self.reduction = reduction
    
    def extra_repr(self) -> str:
        """Return a string with extra information."""
        return f"reduction='{self.reduction}'"

class TorchMSELoss(TorchLoss):
    """
    PyTorch implementation of Mean Squared Error loss.
    
    Creates a criterion that measures the mean squared error (squared L2 norm)
    between each element in the input and target.
    """
    
    def forward(self, input, target):
        """
        Forward pass of the MSE loss.
        
        Args:
            input: Predicted values
            target: Target values
            
        Returns:
            Loss value
        """
        # Ensure inputs are tensors
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input, device='cpu')
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, device='cpu', dtype=input.dtype)
        
        return F.mse_loss(input, target, reduction=self.reduction)

class TorchCrossEntropyLoss(TorchLoss):
    """
    PyTorch implementation of Cross Entropy loss.
    
    This criterion computes the cross entropy loss between input logits and target.
    It is useful when training a classification problem with C classes.
    """
    
    def __init__(self, weight=None, reduction='mean', ignore_index=-100, label_smoothing=0.0):
        """
        Initialize a Cross Entropy loss.
        
        Args:
            weight: Manual rescaling weight given to each class
            reduction: Specifies the reduction to apply to the output
            ignore_index: Specifies a target value that is ignored
            label_smoothing: Float in [0.0, 1.0], specifies the amount of smoothing
        """
        super().__init__(reduction)
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def forward(self, input, target):
        """
        Forward pass of the Cross Entropy loss.
        
        Args:
            input: Predicted logits (B, C) where B is batch size and C is number of classes
            target: Target class indices (B) or one-hot vectors (B, C)
            
        Returns:
            Loss value
        """
        # Ensure inputs are tensors
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input, device='cpu')
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, device='cpu')
        
        # Convert weight to tensor if provided
        weight = None
        if self.weight is not None:
            if not isinstance(self.weight, torch.Tensor):
                weight = torch.tensor(self.weight, device=input.device, dtype=input.dtype)
            else:
                weight = self.weight
        
        # Handle one-hot targets
        if target.dim() == input.dim():
            # Target is one-hot encoded, convert to class indices
            target = torch.argmax(target, dim=1)
        
        return F.cross_entropy(
            input, target,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )
    
    def extra_repr(self) -> str:
        """Return a string with extra information."""
        return (f"reduction='{self.reduction}', "
                f"ignore_index={self.ignore_index}, "
                f"label_smoothing={self.label_smoothing}")

class TorchBCELoss(TorchLoss):
    """
    PyTorch implementation of Binary Cross Entropy loss.
    
    Creates a criterion that measures the Binary Cross Entropy between the
    target and the input probabilities.
    """
    
    def __init__(self, weight=None, reduction='mean'):
        """
        Initialize a Binary Cross Entropy loss.
        
        Args:
            weight: Manual rescaling weight given to the loss of each batch element
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__(reduction)
        self.weight = weight
    
    def forward(self, input, target):
        """
        Forward pass of the BCE loss.
        
        Args:
            input: Predicted probabilities (0-1)
            target: Target values (0-1)
            
        Returns:
            Loss value
        """
        # Ensure inputs are tensors
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input, device='cpu')
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, device='cpu', dtype=input.dtype)
        
        # Convert weight to tensor if provided
        weight = None
        if self.weight is not None:
            if not isinstance(self.weight, torch.Tensor):
                weight = torch.tensor(self.weight, device=input.device, dtype=input.dtype)
            else:
                weight = self.weight
        
        return F.binary_cross_entropy(input, target, weight=weight, reduction=self.reduction)

class TorchBCEWithLogitsLoss(TorchLoss):
    """
    PyTorch implementation of Binary Cross Entropy with Logits loss.
    
    This loss combines a Sigmoid layer and the BCELoss in one single class.
    This is more numerically stable than using a plain Sigmoid followed by a BCELoss.
    """
    
    def __init__(self, weight=None, reduction='mean', pos_weight=None):
        """
        Initialize a Binary Cross Entropy with Logits loss.
        
        Args:
            weight: Manual rescaling weight given to the loss of each batch element
            reduction: Specifies the reduction to apply to the output
            pos_weight: Weight of positive examples
        """
        super().__init__(reduction)
        self.weight = weight
        self.pos_weight = pos_weight
    
    def forward(self, input, target):
        """
        Forward pass of the BCE with Logits loss.
        
        Args:
            input: Predicted logits
            target: Target values (0-1)
            
        Returns:
            Loss value
        """
        # Ensure inputs are tensors
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input, device='cpu')
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, device='cpu', dtype=input.dtype)
        
        # Convert weight and pos_weight to tensors if provided
        weight = None
        if self.weight is not None:
            if not isinstance(self.weight, torch.Tensor):
                weight = torch.tensor(self.weight, device=input.device, dtype=input.dtype)
            else:
                weight = self.weight
        
        pos_weight = None
        if self.pos_weight is not None:
            if not isinstance(self.pos_weight, torch.Tensor):
                pos_weight = torch.tensor(self.pos_weight, device=input.device, dtype=input.dtype)
            else:
                pos_weight = self.pos_weight
        
        return F.binary_cross_entropy_with_logits(
            input, target,
            weight=weight,
            reduction=self.reduction,
            pos_weight=pos_weight
        )
    
    def extra_repr(self) -> str:
        """Return a string with extra information."""
        return f"reduction='{self.reduction}'"