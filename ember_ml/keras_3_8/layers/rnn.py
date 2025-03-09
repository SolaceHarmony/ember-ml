"""
RNN layer implementations with LTC cell support.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, Dict, Union, List
from ..base import Layer

class LTCCell(nn.Module):
    """Liquid Time Constant cell implementation."""
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 tau: float = 1.0):
        """
        Initialize LTC cell.

        Args:
            input_size: Input dimension
            hidden_size: Hidden state dimension
            tau: Time constant
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = tau
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Hidden projection
        self.hidden_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self,
                inputs: torch.Tensor,
                state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LTC cell.

        Args:
            inputs: Input tensor [batch_size, input_size]
            state: Optional previous state [batch_size, hidden_size]

        Returns:
            Tuple of (output, new_state)
        """
        if state is None:
            state = torch.zeros(
                inputs.size(0),
                self.hidden_size,
                device=inputs.device
            )
            
        # Compute input contribution
        input_signal = self.input_proj(inputs)
        
        # Compute hidden contribution
        hidden_signal = self.hidden_proj(state)
        
        # LTC update
        new_state = state + (1.0 / self.tau) * (
            input_signal + hidden_signal - state
        )
        
        # Compute output
        output = self.output_proj(new_state)
        
        return output, new_state

class RNN(Layer):
    """RNN layer with support for various cell types."""
    
    def __init__(self,
                 cell: nn.Module,
                 return_sequences: bool = False,
                 return_state: bool = False):
        """
        Initialize RNN layer.

        Args:
            cell: RNN cell module
            return_sequences: Whether to return full sequence
            return_state: Whether to return final state
        """
        super().__init__()
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Build layer based on input shape.

        Args:
            input_shape: Input tensor shape
        """
        # Input shape: [batch, time, features]
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input shape [batch, time, features], got {len(input_shape)}D"
            )
            
        feature_dim = input_shape[-1]
        if hasattr(self.cell, 'build'):
            self.cell.build(feature_dim)
            
        self.built = True
        
    def call(self,
             inputs: torch.Tensor,
             initial_state: Optional[torch.Tensor] = None,
             training: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through RNN layer.

        Args:
            inputs: Input tensor [batch, time, features]
            initial_state: Optional initial state
            training: Whether in training mode

        Returns:
            Output tensor [batch, time, hidden] if return_sequences
            or [batch, hidden] if not return_sequences
            and optionally final state if return_state
        """
        # Input dimensions
        batch_size = inputs.size(0)
        time_steps = inputs.size(1)
        
        # Initialize state
        if initial_state is None:
            if hasattr(self.cell, 'hidden_size'):
                state = torch.zeros(
                    batch_size,
                    self.cell.hidden_size,
                    device=inputs.device
                )
            else:
                raise ValueError("Cell must have hidden_size attribute or initial_state must be provided")
        else:
            state = initial_state
            
        # Process sequence
        outputs = []
        for t in range(time_steps):
            output, state = self.cell(inputs[:, t], state)
            outputs.append(output)
            
        # Stack outputs
        if self.return_sequences:
            outputs = torch.stack(outputs, dim=1)  # [batch, time, hidden]
        else:
            outputs = output  # Last output [batch, hidden]
            
        if self.return_state:
            return outputs, state
        return outputs
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'return_sequences': self.return_sequences,
            'return_state': self.return_state
        })
        return config

def create_ltc_rnn(input_size: int,
                  hidden_size: int,
                  tau: float = 1.0,
                  return_sequences: bool = False,
                  return_state: bool = False) -> RNN:
    """
    Create RNN layer with LTC cell.

    Args:
        input_size: Input dimension
        hidden_size: Hidden state dimension
        tau: Time constant
        return_sequences: Whether to return full sequence
        return_state: Whether to return final state

    Returns:
        Configured RNN layer with LTC cell
    """
    cell = LTCCell(
        input_size=input_size,
        hidden_size=hidden_size,
        tau=tau
    )
    return RNN(
        cell=cell,
        return_sequences=return_sequences,
        return_state=return_state
    )