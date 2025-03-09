"""
Temporal attention mechanisms for sequence processing and time-based patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
from .base import BaseAttention

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal information."""
    
    def __init__(self,
                 hidden_size: int,
                 dropout: float = 0.1,
                 max_len: int = 1000):
        """
        Initialize positional encoding.

        Args:
            hidden_size: Hidden state dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() *
            -(math.log(10000.0) / hidden_size)
        )
        
        # Compute sinusoidal pattern
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self,
                x: torch.Tensor,
                times: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            times: Optional time points [batch, seq_len]

        Returns:
            Encoded tensor [batch, seq_len, hidden_size]
        """
        seq_len = x.size(1)
        if times is not None:
            # Scale positional encoding by time differences
            time_scale = times / times.max()  # Normalize to [0, 1]
            time_scale = time_scale.unsqueeze(-1)
            pe = self.pe[:, :seq_len] * time_scale
        else:
            pe = self.pe[:, :seq_len]
            
        # Expand positional encoding to match batch size
        pe = pe.expand(x.size(0), -1, -1)
        return self.dropout(x + pe)

class TemporalAttention(BaseAttention):
    """
    Attention mechanism specialized for temporal sequence processing
    with time-aware attention computation.
    """
    
    def __init__(self,
                 hidden_size: int,
                 num_heads: int = 1,
                 dropout: float = 0.1,
                 max_len: int = 1000,
                 use_time_embedding: bool = True):
        """
        Initialize temporal attention.

        Args:
            hidden_size: Hidden state dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_len: Maximum sequence length
            use_time_embedding: Whether to use temporal embeddings
        """
        super().__init__()
        assert hidden_size % num_heads == 0, \
            "hidden_size must be divisible by num_heads"
            
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_len = max_len
        self.use_time_embedding = use_time_embedding
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Temporal components
        if use_time_embedding:
            self.time_embedding = PositionalEncoding(
                hidden_size,
                dropout=dropout,
                max_len=max_len
            )
        
        # Time-aware attention components
        self.time_gate = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self._attention_weights = None
        
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                times: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute temporal attention.

        Args:
            query: Query tensor [batch, query_len, hidden_size]
            key: Key tensor [batch, key_len, hidden_size]
            value: Value tensor [batch, key_len, hidden_size]
            times: Optional time points [batch, seq_len]
            mask: Optional attention mask [batch, query_len, key_len]

        Returns:
            Attention output [batch, query_len, hidden_size]
        """
        batch_size, query_len, _ = query.size()
        key_len = key.size(1)
        
        # Add temporal embeddings if enabled
        if self.use_time_embedding and times is not None:
            query = self.time_embedding(query, times[:, :query_len])
            key = self.time_embedding(key, times[:, :key_len])
            value = self.time_embedding(value, times[:, :key_len])
        
        # Project and reshape
        q = self.q_proj(query).view(
            batch_size, query_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(key).view(
            batch_size, key_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(value).view(
            batch_size, key_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply time-based attention if times provided
        if times is not None:
            # Compute time differences [batch, query_len, key_len]
            time_diffs = times.unsqueeze(2) - times.unsqueeze(1)
            
            # Project time differences to match query dimensions
            time_diffs = time_diffs.unsqueeze(-1)  # [batch, query_len, key_len, 1]
            
            # Reshape query for time gating
            query_expanded = query.unsqueeze(2).expand(-1, -1, key_len, -1)
            
            # Concatenate along feature dimension
            time_features = torch.cat([
                query_expanded,
                time_diffs
            ], dim=-1)
            
            # Apply time gating
            time_gates = self.time_gate(time_features)
            
            # Reshape gates to match attention scores
            time_gates = time_gates.mean(dim=-1).unsqueeze(1)
            scores = scores * time_gates
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        # Apply attention weights
        self._attention_weights = F.softmax(scores, dim=-1)
        self._attention_weights = self.dropout(self._attention_weights)
        
        # Compute output
        attn_output = torch.matmul(self._attention_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.hidden_size
        )
        attn_output = self.out_proj(attn_output)
        
        return attn_output
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get last computed attention weights."""
        return self._attention_weights

def create_temporal_attention(hidden_size: int,
                            num_heads: int = 1,
                            dropout: float = 0.1,
                            max_len: int = 1000,
                            use_time_embedding: bool = True) -> TemporalAttention:
    """
    Factory function to create temporal attention mechanism.

    Args:
        hidden_size: Hidden state dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        max_len: Maximum sequence length
        use_time_embedding: Whether to use temporal embeddings

    Returns:
        Configured temporal attention mechanism
    """
    return TemporalAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        dropout=dropout,
        max_len=max_len,
        use_time_embedding=use_time_embedding
    )