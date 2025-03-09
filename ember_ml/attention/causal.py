"""
Causal attention mechanisms incorporating temporal, causal, and novelty factors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from .base import BaseAttention

@dataclass
class AttentionState:
    """State container for causal attention components."""
    
    temporal_weight: float = 0.0    # Recent history importance
    causal_weight: float = 0.0      # Prediction accuracy impact
    novelty_weight: float = 0.0     # Curiosity factor
    
    def compute_total(self) -> float:
        """Compute total attention weight."""
        return (
            self.temporal_weight +
            self.causal_weight +
            self.novelty_weight
        ) / 3.0

class CausalMemory:
    """Memory buffer for causal relationships and predictions."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize causal memory.

        Args:
            max_size: Maximum memory size
        """
        self.max_size = max_size
        self.cause_effect_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.prediction_accuracy: List[float] = []
        
    def add(self, cause: torch.Tensor, effect: torch.Tensor, accuracy: float):
        """
        Add cause-effect pair to memory.

        Args:
            cause: Cause state tensor
            effect: Effect state tensor
            accuracy: Prediction accuracy
        """
        self.cause_effect_pairs.append((cause.clone(), effect.clone()))
        self.prediction_accuracy.append(accuracy)
        
        if len(self.cause_effect_pairs) > self.max_size:
            self.cause_effect_pairs.pop(0)
            self.prediction_accuracy.pop(0)
            
    def get_similar_causes(self,
                          current_state: torch.Tensor,
                          threshold: float = 0.8) -> List[int]:
        """
        Find indices of similar causes in memory.

        Args:
            current_state: Current state tensor
            threshold: Similarity threshold

        Returns:
            List of indices with similar causes
        """
        similar_indices = []
        for i, (cause, _) in enumerate(self.cause_effect_pairs):
            similarity = torch.cosine_similarity(
                current_state.view(1, -1),
                cause.view(1, -1)
            ).item()
            if similarity > threshold:
                similar_indices.append(i)
        return similar_indices
        
    def get_prediction(self,
                      current_state: torch.Tensor,
                      k: int = 5) -> Tuple[torch.Tensor, float]:
        """
        Get predicted effect based on similar causes.

        Args:
            current_state: Current state tensor
            k: Number of nearest neighbors to consider

        Returns:
            Tuple of (predicted effect, confidence)
        """
        similar_indices = self.get_similar_causes(current_state)
        if not similar_indices:
            return None, 0.0
            
        # Get top-k similar causes
        similarities = []
        for idx in similar_indices:
            cause, _ = self.cause_effect_pairs[idx]
            similarity = torch.cosine_similarity(
                current_state.view(1, -1),
                cause.view(1, -1)
            ).item()
            similarities.append((idx, similarity))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        # Weighted average of effects
        total_weight = sum(sim for _, sim in top_k)
        predicted_effect = torch.zeros_like(
            self.cause_effect_pairs[0][1]
        )
        
        for idx, sim in top_k:
            weight = sim / total_weight
            _, effect = self.cause_effect_pairs[idx]
            predicted_effect += weight * effect
            
        # Compute confidence
        confidence = sum(
            sim * self.prediction_accuracy[idx]
            for idx, sim in top_k
        ) / len(top_k)
        
        return predicted_effect, confidence
        
    def clear(self):
        """Clear memory buffer."""
        self.cause_effect_pairs.clear()
        self.prediction_accuracy.clear()

class PredictionAttention(BaseAttention):
    """Attention mechanism based on prediction accuracy and causal relationships."""
    
    def __init__(self,
                 hidden_size: int,
                 num_heads: int = 1,
                 dropout: float = 0.1,
                 memory_size: int = 1000):
        """
        Initialize prediction attention.

        Args:
            hidden_size: Hidden state dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            memory_size: Size of causal memory
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, \
            "hidden_size must be divisible by num_heads"
            
        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Prediction components
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Memory
        self.memory = CausalMemory(max_size=memory_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights
        self._attention_weights = None
        
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute prediction-based attention.

        Args:
            query: Query tensor [batch, query_len, hidden_size]
            key: Key tensor [batch, key_len, hidden_size]
            value: Value tensor [batch, key_len, hidden_size]
            mask: Optional attention mask [batch, query_len, key_len]

        Returns:
            Attention output [batch, query_len, hidden_size]
        """
        batch_size, query_len, _ = query.size()
        key_len = key.size(1)
        
        # Project inputs
        q = self.q_proj(query).view(
            batch_size, query_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(key).view(
            batch_size, key_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(value).view(
            batch_size, key_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # Make predictions
        predicted_values = self.predictor(key)
        prediction_error = torch.norm(
            predicted_values - value,
            dim=-1,
            keepdim=True
        )
        prediction_weights = F.softmax(-prediction_error, dim=1)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply prediction weights
        scores = scores * prediction_weights.unsqueeze(1)
        
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
        
        # Update memory with predictions
        for i in range(batch_size):
            for j in range(key_len - 1):
                cause = key[i, j]
                effect = value[i, j + 1]
                accuracy = 1.0 - prediction_error[i, j].item()
                self.memory.add(cause, effect, accuracy)
        
        return attn_output
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get last computed attention weights."""
        return self._attention_weights

class CausalAttention(BaseAttention):
    """
    Attention mechanism incorporating causality, temporal dynamics,
    and novelty detection.
    """
    
    def __init__(self,
                 hidden_size: int,
                 decay_rate: float = 0.1,
                 novelty_threshold: float = 0.3,
                 memory_length: int = 100):
        """
        Initialize causal attention.

        Args:
            hidden_size: Hidden state dimension
            decay_rate: Temporal decay rate
            novelty_threshold: Threshold for novelty detection
            memory_length: Length of attention history
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.decay_rate = decay_rate
        self.novelty_threshold = novelty_threshold
        self.memory_length = memory_length
        
        # State tracking
        self.states: Dict[int, AttentionState] = {}
        self.history: List[Tuple[int, float]] = []
        
        # Learnable components
        self.temporal_proj = nn.Linear(hidden_size, hidden_size)
        self.causal_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.novelty_proj = nn.Linear(hidden_size, 1)
        
        # Store attention weights
        self._attention_weights = None
        
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute causal attention.

        Args:
            query: Query tensor [batch_size, hidden_size]
            key: Key tensor [batch_size, hidden_size]
            value: Value tensor [batch_size, hidden_size]
            mask: Optional attention mask

        Returns:
            Attention output [batch_size, hidden_size]
        """
        batch_size = query.size(0)
        attention_states = []
        
        # Process each item in batch
        attention_weights = []
        for i in range(batch_size):
            state = self.update(
                i,  # Use batch index as neuron_id
                query[i],
                key[i]
            )
            attention_states.append(state)
            attention_weights.append(state.compute_total())
            
        # Convert to tensor and apply sigmoid for normalization
        self._attention_weights = torch.sigmoid(torch.tensor(
            attention_weights,
            device=query.device
        )).view(batch_size, 1, 1)  # Changed from view(batch_size) to view(batch_size, 1, 1)
        
        # Apply attention weights
        output = value * self._attention_weights.squeeze(-1).squeeze(-1).unsqueeze(-1)
        
        return output
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get last computed attention weights."""
        return self._attention_weights
    
    def update(self,
               neuron_id: int,
               current_state: torch.Tensor,
               target_state: torch.Tensor) -> AttentionState:
        """
        Update attention state for a single neuron.

        Args:
            neuron_id: Neuron identifier
            current_state: Current hidden state
            target_state: Target hidden state

        Returns:
            Updated attention state
        """
        # Get or create attention state
        state = self.states.get(neuron_id, AttentionState())
        
        # Update temporal weight
        temporal_decay = torch.exp(
            torch.tensor(-self.decay_rate * len(self.history))
        ).item()
        temporal_features = self.temporal_proj(current_state)
        state.temporal_weight = torch.mean(temporal_features).item() * temporal_decay
        
        # Update causal weight
        prediction_error = target_state - current_state
        causal_input = torch.cat([current_state, prediction_error])
        causal_features = self.causal_proj(causal_input)
        prediction_accuracy = 1.0 - min(
            torch.norm(prediction_error).item(),
            1.0
        )
        state.causal_weight = prediction_accuracy * torch.mean(
            causal_features
        ).item()
        
        # Update novelty weight
        novelty = self.novelty_proj(
            target_state - current_state
        ).item()
        if abs(novelty) > self.novelty_threshold:
            state.novelty_weight = abs(novelty)
        else:
            state.novelty_weight *= (1 - self.decay_rate)
        
        # Store updated state
        self.states[neuron_id] = state
        
        # Update history
        total_attention = state.compute_total()
        self.history.append((neuron_id, total_attention))
        if len(self.history) > self.memory_length:
            self.history.pop(0)
            
        return state
    
    def reset(self) -> None:
        """Reset attention states and history."""
        self.states.clear()
        self.history.clear()
        
    def save_state(self) -> Dict[str, Any]:
        """Save attention mechanism state."""
        return {
            'hidden_size': self.hidden_size,
            'decay_rate': self.decay_rate,
            'novelty_threshold': self.novelty_threshold,
            'memory_length': self.memory_length,
            'states': {
                k: {
                    'temporal_weight': v.temporal_weight,
                    'causal_weight': v.causal_weight,
                    'novelty_weight': v.novelty_weight
                }
                for k, v in self.states.items()
            },
            'history': self.history,
            'temporal_proj': self.temporal_proj.state_dict(),
            'causal_proj': self.causal_proj.state_dict(),
            'novelty_proj': self.novelty_proj.state_dict()
        }
    
    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load attention mechanism state."""
        self.hidden_size = state_dict['hidden_size']
        self.decay_rate = state_dict['decay_rate']
        self.novelty_threshold = state_dict['novelty_threshold']
        self.memory_length = state_dict['memory_length']
        
        self.states = {
            k: AttentionState(
                temporal_weight=v['temporal_weight'],
                causal_weight=v['causal_weight'],
                novelty_weight=v['novelty_weight']
            )
            for k, v in state_dict['states'].items()
        }
        
        self.history = state_dict['history']
        self.temporal_proj.load_state_dict(state_dict['temporal_proj'])
        self.causal_proj.load_state_dict(state_dict['causal_proj'])
        self.novelty_proj.load_state_dict(state_dict['novelty_proj'])

def create_causal_attention(hidden_size: int,
                          decay_rate: float = 0.1,
                          novelty_threshold: float = 0.3,
                          memory_length: int = 100) -> CausalAttention:
    """
    Factory function to create causal attention mechanism.

    Args:
        hidden_size: Hidden state dimension
        decay_rate: Temporal decay rate
        novelty_threshold: Threshold for novelty detection
        memory_length: Length of attention history

    Returns:
        Configured causal attention mechanism
    """
    return CausalAttention(
        hidden_size=hidden_size,
        decay_rate=decay_rate,
        novelty_threshold=novelty_threshold,
        memory_length=memory_length
    )