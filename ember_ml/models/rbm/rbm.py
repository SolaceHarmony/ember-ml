"""
Restricted Boltzmann Machine (RBM) implementation.

This module provides a Restricted Boltzmann Machine (RBM) implementation for the ember_ml library.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union, Callable

class RestrictedBoltzmannMachine(nn.Module):
    """
    Restricted Boltzmann Machine (RBM) implementation.
    
    RBMs are generative stochastic neural networks that can learn a probability distribution
    over their inputs. They consist of a visible layer and a hidden layer, with no connections
    between units within the same layer.
    """
    
    def __init__(self, visible_size: int, hidden_size: int, 
                 visible_type: str = 'binary', hidden_type: str = 'binary',
                 device: Optional[torch.device] = None):
        """
        Initialize the RBM.
        
        Args:
            visible_size: Number of visible units
            hidden_size: Number of hidden units
            visible_type: Type of visible units ('binary' or 'gaussian')
            hidden_type: Type of hidden units ('binary' or 'gaussian')
            device: Device to use for computation
        """
        super().__init__()
        
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.visible_type = visible_type
        self.hidden_type = hidden_type
        
        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize weights and biases
        self.weights = nn.Parameter(torch.randn(visible_size, hidden_size, device=self.device) * 0.1)
        self.visible_bias = nn.Parameter(torch.zeros(visible_size, device=self.device))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_size, device=self.device))
        
        # Move to device
        self.to(self.device)
    
    def visible_to_hidden(self, visible: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute hidden activations and probabilities given visible units.
        
        Args:
            visible: Visible units tensor of shape (batch_size, visible_size)
            
        Returns:
            Tuple of (hidden_probs, hidden_states)
        """
        # Compute hidden activations
        hidden_activations = F.linear(visible, self.weights.t(), self.hidden_bias)
        
        # Compute hidden probabilities
        if self.hidden_type == 'binary':
            hidden_probs = torch.sigmoid(hidden_activations)
        else:  # gaussian
            hidden_probs = hidden_activations
        
        # Sample hidden states
        if self.hidden_type == 'binary':
            hidden_states = torch.bernoulli(hidden_probs)
        else:  # gaussian
            hidden_states = hidden_probs + torch.randn_like(hidden_probs)
        
        return hidden_probs, hidden_states
    
    def hidden_to_visible(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute visible activations and probabilities given hidden units.
        
        Args:
            hidden: Hidden units tensor of shape (batch_size, hidden_size)
            
        Returns:
            Tuple of (visible_probs, visible_states)
        """
        # Compute visible activations
        visible_activations = F.linear(hidden, self.weights, self.visible_bias)
        
        # Compute visible probabilities
        if self.visible_type == 'binary':
            visible_probs = torch.sigmoid(visible_activations)
        else:  # gaussian
            visible_probs = visible_activations
        
        # Sample visible states
        if self.visible_type == 'binary':
            visible_states = torch.bernoulli(visible_probs)
        else:  # gaussian
            visible_states = visible_probs + torch.randn_like(visible_probs)
        
        return visible_probs, visible_states
    
    def forward(self, visible: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            visible: Visible units tensor of shape (batch_size, visible_size)
            
        Returns:
            Tuple of (hidden_probs, hidden_states, visible_probs, visible_states)
        """
        # Visible to hidden
        hidden_probs, hidden_states = self.visible_to_hidden(visible)
        
        # Hidden to visible
        visible_probs, visible_states = self.hidden_to_visible(hidden_states)
        
        return hidden_probs, hidden_states, visible_probs, visible_states
    
    def free_energy(self, visible: torch.Tensor) -> torch.Tensor:
        """
        Compute the free energy of a visible vector.
        
        Args:
            visible: Visible units tensor of shape (batch_size, visible_size)
            
        Returns:
            Free energy tensor of shape (batch_size,)
        """
        # Compute visible term
        visible_term = -torch.matmul(visible, self.visible_bias)
        
        # Compute hidden term
        hidden_activations = F.linear(visible, self.weights.t(), self.hidden_bias)
        
        if self.hidden_type == 'binary':
            hidden_term = -torch.sum(F.softplus(hidden_activations), dim=1)
        else:  # gaussian
            hidden_term = -0.5 * torch.sum(hidden_activations ** 2, dim=1)
        
        return visible_term + hidden_term
    
    def reconstruct(self, visible: torch.Tensor, num_gibbs_steps: int = 1) -> torch.Tensor:
        """
        Reconstruct visible units.
        
        Args:
            visible: Visible units tensor of shape (batch_size, visible_size)
            num_gibbs_steps: Number of Gibbs sampling steps
            
        Returns:
            Reconstructed visible units tensor of shape (batch_size, visible_size)
        """
        # Initial hidden states
        _, hidden_states = self.visible_to_hidden(visible)
        
        # Gibbs sampling
        for _ in range(num_gibbs_steps):
            visible_probs, visible_states = self.hidden_to_visible(hidden_states)
            _, hidden_states = self.visible_to_hidden(visible_states)
        
        return visible_probs
    
    def sample(self, num_samples: int, num_gibbs_steps: int = 1000) -> torch.Tensor:
        """
        Sample from the RBM.
        
        Args:
            num_samples: Number of samples to generate
            num_gibbs_steps: Number of Gibbs sampling steps
            
        Returns:
            Samples tensor of shape (num_samples, visible_size)
        """
        # Initialize visible states randomly
        visible_states = torch.rand(num_samples, self.visible_size, device=self.device)
        
        # Gibbs sampling
        for _ in range(num_gibbs_steps):
            _, hidden_states = self.visible_to_hidden(visible_states)
            visible_probs, visible_states = self.hidden_to_visible(hidden_states)
        
        return visible_probs

def train_rbm(rbm: RestrictedBoltzmannMachine, 
              data: torch.Tensor, 
              num_epochs: int = 10, 
              batch_size: int = 32, 
              learning_rate: float = 0.01, 
              momentum: float = 0.5, 
              weight_decay: float = 0.0001, 
              num_gibbs_steps: int = 1,
              callback: Optional[Callable[[int, float], None]] = None) -> List[float]:
    """
    Train an RBM using contrastive divergence.
    
    Args:
        rbm: RBM to train
        data: Training data tensor of shape (num_samples, visible_size)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        momentum: Momentum coefficient
        weight_decay: Weight decay coefficient
        num_gibbs_steps: Number of Gibbs sampling steps
        callback: Optional callback function called after each epoch with (epoch, loss)
        
    Returns:
        List of losses for each epoch
    """
    # Move data to device
    data = data.to(rbm.device)
    
    # Create optimizer
    optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
    # Training loop
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch in data_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Positive phase
            positive_visible = batch
            positive_hidden_probs, positive_hidden_states = rbm.visible_to_hidden(positive_visible)
            
            # Negative phase
            negative_hidden_states = positive_hidden_states
            
            for _ in range(num_gibbs_steps):
                negative_visible_probs, negative_visible_states = rbm.hidden_to_visible(negative_hidden_states)
                negative_hidden_probs, negative_hidden_states = rbm.visible_to_hidden(negative_visible_states)
            
            # Compute loss (free energy difference)
            positive_free_energy = rbm.free_energy(positive_visible)
            negative_free_energy = rbm.free_energy(negative_visible_probs)
            
            loss = torch.mean(positive_free_energy - negative_free_energy)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Average loss for the epoch
        epoch_loss /= len(data_loader)
        losses.append(epoch_loss)
        
        # Call callback if provided
        if callback is not None:
            callback(epoch, epoch_loss)
    
    return losses

def reconstruct_with_rbm(rbm: RestrictedBoltzmannMachine, 
                         data: torch.Tensor, 
                         num_gibbs_steps: int = 1) -> torch.Tensor:
    """
    Reconstruct data using an RBM.
    
    Args:
        rbm: Trained RBM
        data: Data tensor of shape (num_samples, visible_size)
        num_gibbs_steps: Number of Gibbs sampling steps
        
    Returns:
        Reconstructed data tensor of shape (num_samples, visible_size)
    """
    # Move data to device
    data = data.to(rbm.device)
    
    # Reconstruct
    with torch.no_grad():
        reconstructed = rbm.reconstruct(data, num_gibbs_steps)
    
    return reconstructed