"""
Base attention mechanisms and multi-head attention implementations.
"""

from typing import Optional
from abc import ABC, abstractmethod

from ember_ml import ops
from ember_ml.ops.tensor import EmberTensor
from ember_ml.nn.modules import Module
from ember_ml.nn.linear import Linear
from ember_ml.nn.container import Dropout

# Type aliases
Tensor = EmberTensor

# Constants
NINF = ops.convert_to_tensor(-1.0e38)  # Approximation of negative infinity

class BaseAttention(Module, ABC):
    """Abstract base class for attention mechanisms."""
    
    def __init__(self):
        """Initialize base attention."""
        super().__init__()
    
    @abstractmethod
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
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
    def get_attention_weights(self) -> Optional[Tensor]:
        """
        Get last computed attention weights.

        Returns:
            Optional attention weights tensor
        """
        pass

class AttentionMask:
    """Utility class for creating attention masks."""
    
    @staticmethod
    def create_padding_mask(lengths: Tensor, max_len: int) -> Tensor:
        """
        Create padding mask from sequence lengths.

        Args:
            lengths: Sequence lengths [batch_size]
            max_len: Maximum sequence length

        Returns:
            Padding mask [batch_size, max_len]
        """
        batch_size = ops.shape(lengths)[0]
        mask = ops.less(
            ops.expand_dims(ops.arange(max_len), 0),
            ops.expand_dims(lengths, 1)
        )
        return mask
    
    @staticmethod
    def create_causal_mask(seq_len: int) -> Tensor:
        """
        Create causal (triangular) mask.

        Args:
            seq_len: Sequence length

        Returns:
            Causal mask [seq_len, seq_len]
        """
        # Create a matrix where each row i contains [0, 1, 2, ..., seq_len-1]
        row_indices = ops.expand_dims(ops.arange(seq_len), 0)
        # Create a matrix where each column j contains [0, 1, 2, ..., seq_len-1]
        col_indices = ops.expand_dims(ops.arange(seq_len), 1)
        # Create a lower triangular matrix where entry (i,j) is 1 if j <= i, else 0
        return ops.cast(ops.less_equal(col_indices, row_indices), ops.float32)
    
    @staticmethod
    def create_window_mask(seq_len: int, window_size: int) -> Tensor:
        """
        Create sliding window mask.

        Args:
            seq_len: Sequence length
            window_size: Window size

        Returns:
            Window mask [seq_len, seq_len]
        """
        # Create a matrix where each row i contains [0, 1, 2, ..., seq_len-1]
        row_indices = ops.expand_dims(ops.arange(seq_len), 0)
        # Create a matrix where each column j contains [0, 1, 2, ..., seq_len-1]
        col_indices = ops.expand_dims(ops.arange(seq_len), 1)
        # Create a mask where entry (i,j) is 1 if |i-j| <= window_size, else 0
        distance = ops.abs(ops.subtract(row_indices, col_indices))
        return ops.cast(ops.less_equal(distance, window_size), ops.float32)

class AttentionScore:
    """Utility class for computing attention scores."""
    
    @staticmethod
    def dot_product(query: Tensor, key: Tensor) -> Tensor:
        """
        Compute dot product attention scores.

        Args:
            query: Query tensor [..., query_dim]
            key: Key tensor [..., key_dim]

        Returns:
            Attention scores [..., query_len, key_len]
        """
        return ops.matmul(query, ops.transpose(key, axes=(-2, -1)))
    
    @staticmethod
    def scaled_dot_product(query: Tensor,
                          key: Tensor,
                          scale: float) -> Tensor:
        """
        Compute scaled dot product attention scores.

        Args:
            query: Query tensor [..., query_dim]
            key: Key tensor [..., key_dim]
            scale: Scaling factor

        Returns:
            Scaled attention scores [..., query_len, key_len]
        """
        return ops.divide(
            ops.matmul(query, ops.transpose(key, axes=(-2, -1))),
            ops.convert_to_tensor(scale)
        )
    
    @staticmethod
    def additive(query: Tensor,
                 key: Tensor,
                 weight: Tensor,
                 bias: Optional[Tensor] = None) -> Tensor:
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
        q_len = ops.shape(query)[-2]
        k_len = ops.shape(key)[-2]
        
        # Expand dimensions for broadcasting
        query_expanded = ops.expand_dims(query, -2)
        # Repeat query for each key
        query_expanded = ops.tile(query_expanded, [1, 1, k_len, 1])
        
        key_expanded = ops.expand_dims(key, -3)
        # Repeat key for each query
        key_expanded = ops.tile(key_expanded, [1, q_len, 1, 1])
        
        # Concatenate query and key
        combined = ops.concatenate([query_expanded, key_expanded], axis=-1)
        
        # Apply weight and optional bias
        scores = ops.matmul(combined, weight)
        if bias is not None:
            scores = ops.add(scores, bias)
            
        return ops.tanh(scores)

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
        self.query = Linear(query_dim, hidden_dim)
        self.key = Linear(key_dim, hidden_dim)
        self.value = Linear(value_dim, hidden_dim)
        self.scale = ops.sqrt(ops.convert_to_tensor(hidden_dim))
        self._attention_weights = None
        
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
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
            scores = ops.where(ops.equal(mask, 0), ops.full_like(scores, NINF), scores)
        
        # Apply attention weights
        self._attention_weights = ops.softmax(scores, axis=-1)
        output = ops.matmul(self._attention_weights, V)
        
        return output
    
    def get_attention_weights(self) -> Optional[Tensor]:
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
        self.dropout_rate = dropout
        self.head_dim = ops.floor_divide(embed_dim, num_heads)
        assert ops.multiply(self.head_dim, num_heads) == embed_dim, \
            "embed_dim must be divisible by num_heads"
            
        # Linear projections
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.dropout_layer = Dropout(dropout)
        
        # Store attention weights
        self._attention_weights = None
        
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
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
        batch_size = ops.shape(query)[0]
        query_len = ops.shape(query)[1]
        key_len = ops.shape(key)[1]
        
        scaling = ops.sqrt(ops.convert_to_tensor(self.head_dim))
        
        # Linear projections and reshape
        q = self.q_proj(query)
        q = ops.reshape(q, (batch_size, query_len, self.num_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))  # [batch, num_heads, query_len, head_dim]
        
        k = self.k_proj(key)
        k = ops.reshape(k, (batch_size, key_len, self.num_heads, self.head_dim))
        k = ops.transpose(k, (0, 2, 1, 3))  # [batch, num_heads, key_len, head_dim]
        
        v = self.v_proj(value)
        v = ops.reshape(v, (batch_size, key_len, self.num_heads, self.head_dim))
        v = ops.transpose(v, (0, 2, 1, 3))  # [batch, num_heads, key_len, head_dim]
        
        # Compute attention scores
        scores = AttentionScore.scaled_dot_product(q, k, scaling)
        
        # Apply mask if provided
        if mask is not None:
            # Add a dimension for the heads
            mask_expanded = ops.expand_dims(mask, axis=1)
            scores = ops.where(ops.equal(mask_expanded, 0), ops.full_like(scores, NINF), scores)
        
        # Apply attention weights
        self._attention_weights = ops.softmax(scores, axis=-1)
        self._attention_weights = self.dropout_layer(self._attention_weights)
        
        # Compute output
        attn_output = ops.matmul(self._attention_weights, v)
        
        # Reshape and project output
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(attn_output, (batch_size, query_len, self.embed_dim))
        attn_output = self.out_proj(attn_output)
        
        return attn_output
    
    def get_attention_weights(self) -> Optional[Tensor]:
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