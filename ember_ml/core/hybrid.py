"""
Hybrid neural architectures combining LTC networks with attention mechanisms and LSTM layers.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from .base import BaseNeuron

class HybridNeuron(BaseNeuron):
    """
    Hybrid neuron combining LTC dynamics with attention mechanisms.
    """
    
    def __init__(self,
                 neuron_id: int,
                 tau: float = 1.0,
                 dt: float = 0.01,
                 hidden_size: int = 64,
                 attention_heads: int = 4):
        """
        Initialize hybrid neuron.

        Args:
            neuron_id: Unique identifier for the neuron
            tau: Time constant
            dt: Time step for numerical integration
            hidden_size: Hidden state dimension
            attention_heads: Number of attention heads
        """
        super().__init__(neuron_id, tau, dt)
        self.hidden_size = hidden_size
        
        # Attention mechanism
        self.attention = AttentionLayer(
            hidden_size,
            hidden_size,
            hidden_size,
            hidden_size
        )
        
        # Memory buffer for temporal attention
        self.memory_buffer = []
        self.max_memory_size = 100
        
    def _initialize_state(self) -> torch.Tensor:
        """Initialize neuron state."""
        return torch.zeros(self.hidden_size)
        
    def update(self,
               input_signal: torch.Tensor,
               **kwargs) -> torch.Tensor:
        """
        Update neuron state using hybrid processing.

        Args:
            input_signal: Input tensor [hidden_size]
            **kwargs: Additional parameters

        Returns:
            Updated state tensor [hidden_size]
        """
        # Add current input to memory buffer
        self.memory_buffer.append(input_signal)
        if len(self.memory_buffer) > self.max_memory_size:
            self.memory_buffer.pop(0)
            
        # Create memory tensor
        memory = torch.stack(self.memory_buffer)
        
        # Apply attention over memory
        attended = self.attention(
            self.state.unsqueeze(0),  # [1, hidden_size]
            memory.unsqueeze(0),      # [1, memory_size, hidden_size]
            memory.unsqueeze(0)       # [1, memory_size, hidden_size]
        )
        
        # LTC update with attention-modulated input
        dh = (1.0 / self.tau) * (attended.squeeze(0) - self.state)
        self.state = self.state + self.dt * dh
        
        # Store history
        self.history.append(self.state.clone())
        
        return self.state
        
    def save_state(self) -> Dict[str, Any]:
        """Save neuron state and parameters."""
        state_dict = super().save_state()
        state_dict.update({
            'hidden_size': self.hidden_size,
            'memory_buffer': [m.tolist() for m in self.memory_buffer],
            'max_memory_size': self.max_memory_size
        })
        return state_dict
        
    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load neuron state and parameters."""
        super().load_state(state_dict)
        self.hidden_size = state_dict['hidden_size']
        self.memory_buffer = [torch.tensor(m) for m in state_dict['memory_buffer']]
        self.max_memory_size = state_dict['max_memory_size']

class AttentionLayer(nn.Module):
    """Multi-head attention layer for temporal processing."""
    
    def __init__(self, 
                 query_dim: int,
                 key_dim: int,
                 value_dim: int,
                 hidden_dim: int):
        """
        Initialize attention layer.

        Args:
            query_dim: Query dimension
            key_dim: Key dimension
            value_dim: Value dimension
            hidden_dim: Hidden dimension for attention computation
        """
        super().__init__()
        self.query = nn.Linear(query_dim, hidden_dim)
        self.key = nn.Linear(key_dim, hidden_dim)
        self.value = nn.Linear(value_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-weighted output.

        Args:
            query: Query tensor [batch, query_dim]
            key: Key tensor [batch, seq_len, key_dim]
            value: Value tensor [batch, seq_len, value_dim]

        Returns:
            Attention-weighted output [batch, hidden_dim]
        """
        # Compute attention scores
        scores = torch.matmul(
            self.query(query).unsqueeze(1),
            self.key(key).transpose(-2, -1)
        ) / self.scale
        
        # Apply attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, self.value(value))
        
        return output.squeeze(1)

class HybridLNNModel(nn.Module):
    """
    Hybrid architecture combining LTC networks with LSTM and attention mechanisms.
    Implements the enhanced model from the LNN-CNN analysis.
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 lstm_hidden_size: int,
                 output_size: int,
                 parallel_chains: int,
                 attention_hidden_dim: int):
        """
        Initialize hybrid model.

        Args:
            input_size: Input dimension
            hidden_size: Hidden dimension for LTC
            lstm_hidden_size: Hidden dimension for LSTM
            output_size: Output dimension
            parallel_chains: Number of parallel LTC chains
            attention_hidden_dim: Hidden dimension for attention
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.parallel_chains = parallel_chains
        
        # LTC chains
        self.ltc_cells = nn.ModuleList([
            ImprovedLiquidTimeConstantCell(input_size, hidden_size)
            for _ in range(parallel_chains)
        ])
        
        # LSTM layer
        self.lstm = nn.LSTM(
            lstm_hidden_size,
            lstm_hidden_size,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(
            lstm_hidden_size,
            lstm_hidden_size,
            lstm_hidden_size,
            attention_hidden_dim
        )
        
        # Output layer
        self.output_layer = nn.Linear(lstm_hidden_size, output_size)
        
    def forward(self,
                input_sequence: torch.Tensor,
                times: torch.Tensor) -> torch.Tensor:
        """
        Process input sequence through hybrid architecture.

        Args:
            input_sequence: Input tensor [batch, seq_len, input_size]
            times: Time points tensor [seq_len]

        Returns:
            Output tensor [batch, seq_len-1, output_size]
        """
        batch_size, seq_len, _ = input_sequence.size()
        device = input_sequence.device
        
        # Initialize hidden state
        x0 = torch.zeros(
            batch_size,
            self.hidden_size * self.parallel_chains
        ).to(device)
        
        outputs = []
        
        # Process sequence
        for t in range(seq_len - 1):
            # Time interval for integration
            t_span = torch.tensor(
                [times[t], times[t + 1]]
            ).to(device)
            
            # Current input
            u = input_sequence[:, t, :]
            
            # Split hidden state for parallel chains
            x0_split = torch.chunk(x0, self.parallel_chains, dim=1)
            
            # Process through LTC chains
            x_new = []
            for chain_idx in range(self.parallel_chains):
                chain_cell = self.ltc_cells[chain_idx]
                x_chain = x0_split[chain_idx]
                
                # Integrate ODE
                x = self._integrate_ode(
                    chain_cell,
                    x_chain,
                    t_span,
                    u[:, chain_idx]
                )
                x_new.append(x[-1])
            
            # Combine chain outputs
            x0 = torch.cat(x_new, dim=1)
            
            # Generate output
            y = self.output_layer(x0)
            outputs.append(y)
            
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)
        
        return outputs
    
    def _integrate_ode(self,
                      cell: nn.Module,
                      x0: torch.Tensor,
                      t_span: torch.Tensor,
                      u: torch.Tensor,
                      method: str = 'rk4',
                      options: dict = {'step_size': 0.1}) -> torch.Tensor:
        """
        Integrate ODE for LTC cell.

        Args:
            cell: LTC cell module
            x0: Initial state
            t_span: Time interval
            u: Input
            method: Integration method
            options: Integration options

        Returns:
            Integrated states
        """
        from torchdiffeq import odeint
        return odeint(cell, x0, t_span, method=method, options=options, args=(u,))

class ImprovedLiquidTimeConstantCell(nn.Module):
    """Enhanced LTC cell with nonlinear dynamics."""
    
    def __init__(self, input_size: int, hidden_size: int):
        """
        Initialize LTC cell.

        Args:
            input_size: Input dimension
            hidden_size: Hidden state dimension
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weight matrices
        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(input_size, hidden_size)
        self.b = nn.Parameter(torch.zeros(hidden_size))
        
        # Time constants
        self.tau = nn.Parameter(torch.ones(hidden_size))
        
    def forward(self,
                t: torch.Tensor,
                x: torch.Tensor,
                u: torch.Tensor) -> torch.Tensor:
        """
        Compute state derivative.

        Args:
            t: Time point
            x: Current state
            u: Input

        Returns:
            State derivative
        """
        # Nonlinear term
        nonlinear_term = torch.sqrt(x + 1)
        
        # State derivative
        dxdt = (-x / self.tau) + torch.tanh(
            self.W(x) + self.U(u) + self.b
        )
        
        # Apply nonlinearity
        dxdt = dxdt * nonlinear_term
        
        return dxdt