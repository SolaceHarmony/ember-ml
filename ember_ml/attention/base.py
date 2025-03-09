"""
Base attention mechanisms and multi-head attention implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod

class BaseAttention(nn.Module, ABC):
    """Abstract base class for attention mechanisms."""
    
    def __init__(self):
        """Initialize base attention."""
        super().__init__()
    
    @abstractmethod
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention mechanism.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask

        Returns:
            Attention output
        """
        pass
    
    @abstractmethod
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get last computed attention weights.

        Returns:
            Optional attention weights tensor
        """
        pass

class AttentionMask:
    """Utility class for creating attention masks."""
    
    @staticmethod
    def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Create padding mask from sequence lengths.

        Args:
            lengths: Sequence lengths [batch_size]
            max_len: Maximum sequence length

        Returns:
            Padding mask [batch_size, max_len]
        """
        batch_size = lengths.size(0)
        mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask
    
    @staticmethod
    def create_causal_mask(seq_len: int) -> torch.Tensor:
        """
        Create causal (triangular) mask.

        Args:
            seq_len: Sequence length

        Returns:
            Causal mask [seq_len, seq_len]
        """
        return torch.tril(torch.ones(seq_len, seq_len))
    
    @staticmethod
    def create_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
        """
        Create sliding window mask.

        Args:
            seq_len: Sequence length
            window_size: Window size

        Returns:
            Window mask [seq_len, seq_len]
        """
        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            mask[i, start:end] = 1
        return mask

class AttentionScore:
    """Utility class for computing attention scores."""
    
    @staticmethod
    def dot_product(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Compute dot product attention scores.

        Args:
            query: Query tensor [..., query_dim]
            key: Key tensor [..., key_dim]

        Returns:
            Attention scores [..., query_len, key_len]
        """
        return torch.matmul(query, key.transpose(-2, -1))
    
    @staticmethod
    def scaled_dot_product(query: torch.Tensor,
                          key: torch.Tensor,
                          scale: float) -> torch.Tensor:
        """
        Compute scaled dot product attention scores.

        Args:
            query: Query tensor [..., query_dim]
            key: Key tensor [..., key_dim]
            scale: Scaling factor

        Returns:
            Scaled attention scores [..., query_len, key_len]
        """
        return torch.matmul(query, key.transpose(-2, -1)) / scale
    
    @staticmethod
    def additive(query: torch.Tensor,
                 key: torch.Tensor,
                 weight: torch.Tensor,
                 bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute additive attention scores.

        Args:
            query: Query tensor [..., query_dim]
            key: Key tensor [..., key_dim]
            weight: Weight matrix [query_dim + key_dim, hidden_dim]
            bias: Optional bias tensor [hidden_dim]

        Returns:
            Attention scores [..., query_len, key_len]
        """
        q_len, k_len = query.size(-2), key.size(-2)
        query = query.unsqueeze(-2).expand(-1, -1, k_len, -1)
        key = key.unsqueeze(-3).expand(-1, q_len, -1, -1)
        
        # Concatenate query and key
        combined = torch.cat([query, key], dim=-1)
        
        # Apply weight and optional bias
        scores = torch.matmul(combined, weight)
        if bias is not None:
            scores = scores + bias
            
        return torch.tanh(scores)

class AttentionLayer(BaseAttention):
    """Basic attention layer implementation."""
    
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
        self._attention_weights = None
        
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention-weighted output.

        Args:
            query: Query tensor [batch, query_len, query_dim]
            key: Key tensor [batch, key_len, key_dim]
            value: Value tensor [batch, key_len, value_dim]
            mask: Optional attention mask [batch, query_len, key_len]

        Returns:
            Attention-weighted output [batch, query_len, hidden_dim]
        """
        # Project inputs
        Q = self.query(query)  # [batch, query_len, hidden_dim]
        K = self.key(key)      # [batch, key_len, hidden_dim]
        V = self.value(value)  # [batch, key_len, hidden_dim]
        
        # Compute attention scores
        scores = AttentionScore.scaled_dot_product(Q, K, self.scale)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply attention weights
        self._attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(self._attention_weights, V)
        
        return output
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get last computed attention weights."""
        return self._attention_weights

class MultiHeadAttention(BaseAttention):
    """Multi-head attention implementation."""
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 bias: bool = True):
        """
        Initialize multi-head attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear projections
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
            
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Store attention weights
        self._attention_weights = None
        
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            query: Query tensor [batch, query_len, embed_dim]
            key: Key tensor [batch, key_len, embed_dim]
            value: Value tensor [batch, key_len, embed_dim]
            mask: Optional attention mask [batch, query_len, key_len]

        Returns:
            Attention output [batch, query_len, embed_dim]
        """
        batch_size, query_len, _ = query.size()
        key_len = key.size(1)
        
        scaling = float(self.head_dim) ** -0.5
        
        # Linear projections and reshape
        q = self.q_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, key_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, key_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, num_heads, query_len, head_dim]
        k = k.transpose(1, 2)  # [batch, num_heads, key_len, head_dim]
        v = v.transpose(1, 2)  # [batch, num_heads, key_len, head_dim]
        
        # Compute attention scores
        scores = AttentionScore.scaled_dot_product(q, k, scaling)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        # Apply attention weights
        self._attention_weights = F.softmax(scores, dim=-1)
        self._attention_weights = self.dropout_layer(self._attention_weights)
        
        # Compute output
        attn_output = torch.matmul(self._attention_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.embed_dim
        )
        attn_output = self.out_proj(attn_output)
        
        return attn_output
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get last computed attention weights."""
        return self._attention_weights

def create_attention_layer(query_dim: int,
                         key_dim: int,
                         value_dim: int,
                         hidden_dim: int) -> AttentionLayer:
    """
    Factory function to create attention layer.

    Args:
        query_dim: Query dimension
        key_dim: Key dimension
        value_dim: Value dimension
        hidden_dim: Hidden dimension

    Returns:
        Configured attention layer
    """
    return AttentionLayer(
        query_dim=query_dim,
        key_dim=key_dim,
        value_dim=value_dim,
        hidden_dim=hidden_dim
    )

def create_multihead_attention(embed_dim: int,
                             num_heads: int,
                             dropout: float = 0.1,
                             bias: bool = True) -> MultiHeadAttention:
    """
    Factory function to create multi-head attention.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias

    Returns:
        Configured multi-head attention
    """
    return MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        bias=bias
    )