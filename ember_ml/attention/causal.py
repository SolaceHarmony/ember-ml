"""
Causal attention mechanisms incorporating temporal, causal, and novelty factors.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

from ember_ml import ops
from ember_ml.ops.tensor import EmberTensor
from ember_ml.nn.linear import Linear
from ember_ml.nn.container import Sequential, Dropout
from ember_ml.nn.activations import Tanh
from ember_ml.attention.base import BaseAttention

# Type aliases
Tensor = EmberTensor

# Constants
NINF = ops.convert_to_tensor(-1.0e38)  # Approximation of negative infinity

# Helper functions
def get_scalar_value(x: Tensor) -> float:
    """Get scalar value from single-element tensor."""
    return ops.cast(x.data, ops.float32)

def normalize(x: Tensor, axis: int = -1) -> Tensor:
    """Apply softmax normalization."""
    exp_x = ops.exp(x)
    return ops.divide(exp_x, ops.sum(exp_x, axis=axis, keepdims=True))

def zero_masking(x: Tensor, mask: Tensor) -> Tensor:
    """Apply zero masking to tensor."""
    return ops.where(mask, x, ops.zeros_like(x))

def masked_fill(x: Tensor, mask: Tensor, value: float) -> Tensor:
    """Fill masked positions with value."""
    value_tensor = ops.full_like(x, value)
    return ops.where(mask, value_tensor, x)

@dataclass
class AttentionState:
    """State container for causal attention components."""
    
    temporal_weight: float = 0.0    # Recent history importance
    causal_weight: float = 0.0      # Prediction accuracy impact
    novelty_weight: float = 0.0     # Curiosity factor
    
    def compute_total(self) -> float:
        """Compute total attention weight."""
        return ops.divide(
            ops.add(
                ops.add(self.temporal_weight, self.causal_weight),
                self.novelty_weight
            ),
            ops.convert_to_tensor(3.0)
        )

class CausalMemory:
    """Memory buffer for causal relationships and predictions."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize causal memory.

        Args:
            max_size: Maximum memory size
        """
        self.max_size = max_size
        self.cause_effect_pairs: List[Tuple[Tensor, Tensor]] = []
        self.prediction_accuracy: List[float] = []
        
    def add(self, cause: Tensor, effect: Tensor, accuracy: float):
        """
        Add cause-effect pair to memory.

        Args:
            cause: Cause state tensor
            effect: Effect state tensor
            accuracy: Prediction accuracy
        """
        self.cause_effect_pairs.append((ops.copy(cause), ops.copy(effect)))
        self.prediction_accuracy.append(accuracy)
        
        if len(self.cause_effect_pairs) > self.max_size:
            self.cause_effect_pairs.pop(0)
            self.prediction_accuracy.pop(0)
            
    def _compute_cosine_similarity(self, a: Tensor, b: Tensor) -> float:
        """
        Compute cosine similarity between two tensors using ops functions.
        
        Args:
            a: First tensor
            b: Second tensor
            
        Returns:
            Cosine similarity value
        """
        # Reshape inputs to 1D
        a_flat = ops.reshape(a, (-1,))
        b_flat = ops.reshape(b, (-1,))
        
        # Compute dot product
        dot_product = ops.sum(ops.multiply(a_flat, b_flat))
        
        # Compute norms
        norm_a = ops.sqrt(ops.sum(ops.multiply(a_flat, a_flat)))
        norm_b = ops.sqrt(ops.sum(ops.multiply(b_flat, b_flat)))
        
        # Compute similarity
        similarity = ops.divide(dot_product, ops.multiply(norm_a, norm_b))
        
        return ops.cast(similarity.data, ops.float32)  # Safe conversion to float

    def get_similar_causes(self,
                          current_state: Tensor,
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
            similarity = self._compute_cosine_similarity(current_state, cause)
            if similarity > threshold:
                similar_indices.append(i)
        return similar_indices
        
    def get_prediction(self,
                      current_state: Tensor,
                      k: int = 5) -> Tuple[Tensor, float]:
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
            # Return zero tensor with same shape as current_state and zero confidence
            if self.cause_effect_pairs:
                empty_shape = ops.shape(self.cause_effect_pairs[0][1])
                return ops.zeros(empty_shape), 0.0
            else:
                return ops.zeros_like(current_state), 0.0
            
        # Get top-k similar causes
        similarities = []
        for idx in similar_indices:
            cause, _ = self.cause_effect_pairs[idx]
            similarity = self._compute_cosine_similarity(current_state, cause)
            similarities.append((idx, similarity))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        # Weighted average of effects
        total_weight = sum(sim for _, sim in top_k)
        predicted_effect = ops.zeros_like(self.cause_effect_pairs[0][1])
        
        for idx, sim in top_k:
            weight = ops.divide(sim, total_weight)
            _, effect = self.cause_effect_pairs[idx]
            predicted_effect = ops.add(predicted_effect, ops.multiply(effect, weight))
            
        # Compute confidence
        confidence = ops.divide(
            sum(ops.multiply(sim, self.prediction_accuracy[idx]) for idx, sim in top_k),
            ops.convert_to_tensor(len(top_k))
        )
        
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
        self.head_dim = ops.floor_divide(hidden_size, num_heads)
        assert ops.multiply(self.head_dim, num_heads) == hidden_size, \
            "hidden_size must be divisible by num_heads"
            
        # Create projections using Module
        self.q_proj = Linear(hidden_size, hidden_size)
        self.k_proj = Linear(hidden_size, hidden_size)
        self.v_proj = Linear(hidden_size, hidden_size)
        self.out_proj = Linear(hidden_size, hidden_size)
        
        # Prediction components
        self.predictor = Sequential([
            Linear(hidden_size, hidden_size),
            Tanh(),
            Linear(hidden_size, hidden_size)
        ])
        
        # Memory
        self.memory = CausalMemory(max_size=memory_size)
        
        # Dropout
        self.dropout = Dropout(dropout)
        
        # Store attention weights
        self._attention_weights = None
        
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
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
        batch_size = ops.shape(query)[0]
        query_len = ops.shape(query)[1]
        key_len = ops.shape(key)[1]
        
        # Project inputs
        q = self.q_proj(query)
        q = ops.reshape(q, (batch_size, query_len, self.num_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))
        
        k = self.k_proj(key)
        k = ops.reshape(k, (batch_size, key_len, self.num_heads, self.head_dim))
        k = ops.transpose(k, (0, 2, 1, 3))
        
        v = self.v_proj(value)
        v = ops.reshape(v, (batch_size, key_len, self.num_heads, self.head_dim))
        v = ops.transpose(v, (0, 2, 1, 3))
        
        # Make predictions
        predicted_values = self.predictor(key)
        prediction_error = ops.norm(
            ops.subtract(predicted_values, value),
            axis=-1,
            keepdims=True
        )
        neg_one = ops.convert_to_tensor(-1.0)
        prediction_weights = normalize(ops.multiply(prediction_error, neg_one), axis=1)
        
        # Compute attention scores
        k_t = ops.transpose(k, (0, 1, 3, 2))  # Transpose for matmul
        scores = ops.divide(ops.matmul(q, k_t), ops.sqrt(ops.convert_to_tensor(self.head_dim)))
        
        # Apply prediction weights
        prediction_weights_expanded = ops.expand_dims(prediction_weights, axis=1)
        scores = ops.multiply(scores, prediction_weights_expanded)
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = ops.expand_dims(mask, axis=1)
            mask_value = NINF  # Use the predefined constant
            scores = ops.where(ops.equal(mask_expanded, 0), ops.full_like(scores, mask_value), scores)
        
        # Apply attention weights
        attention_weights = normalize(scores, axis=-1)
        self._attention_weights = self.dropout(attention_weights)
        
        # Compute output
        attn_output = ops.matmul(self._attention_weights, v)
        
        # Reshape and project output
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(attn_output, (batch_size, query_len, self.hidden_size))
        attn_output = self.out_proj(attn_output)
        
        # Update memory with predictions
        for i in range(batch_size):
            key_len_minus_one = ops.subtract(key_len, ops.convert_to_tensor(1))
            for j in range(ops.cast(key_len_minus_one, ops.int32)):
                # Use slice instead of direct indexing
                cause = ops.slice(key, [i, j, 0], [1, 1, -1])
                cause = ops.reshape(cause, ops.shape(cause)[2:])  # Remove batch and seq dims
                
                j_plus_one = ops.add(j, ops.convert_to_tensor(1))
                effect = ops.slice(value, [i, j_plus_one, 0], [1, 1, -1])
                effect = ops.reshape(effect, ops.shape(effect)[2:])  # Remove batch and seq dims
                
                error_val = ops.slice(prediction_error, [i, j], [1, 1])
                one = ops.convert_to_tensor(1.0)
                accuracy = ops.subtract(one, ops.cast(ops.item(error_val), ops.float32))
                self.memory.add(cause, effect, accuracy)
        
        return attn_output
    
    def get_attention_weights(self) -> Optional[Tensor]:
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
        self.temporal_proj = Linear(hidden_size, hidden_size)
        self.causal_proj = Linear(ops.multiply(hidden_size, ops.convert_to_tensor(2)), hidden_size)
        self.novelty_proj = Linear(hidden_size, 1)
        
        # Store attention weights
        self._attention_weights = None
        
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
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
        batch_size = ops.shape(query)[0]
        attention_states = []
        
        # Process each item in batch
        attention_weights = []
        for i in range(batch_size):
            # Use slice instead of direct indexing
            query_i = ops.slice(query, [i, 0], [1, -1])
            query_i = ops.reshape(query_i, ops.shape(query_i)[1:])  # Remove batch dim
            
            key_i = ops.slice(key, [i, 0], [1, -1])
            key_i = ops.reshape(key_i, ops.shape(key_i)[1:])  # Remove batch dim
            
            state = self.update(
                i,  # Use batch index as neuron_id
                query_i,
                key_i
            )
            attention_states.append(state)
            attention_weights.append(state.compute_total())
            
        # Convert to tensor and apply sigmoid for normalization
        self._attention_weights = ops.sigmoid(ops.convert_to_tensor(
            attention_weights
        ))
        self._attention_weights = ops.reshape(self._attention_weights, (batch_size, 1, 1))
        
        # Apply attention weights
        output = ops.multiply(value, ops.reshape(self._attention_weights, (batch_size, 1)))
        
        return output
    
    def get_attention_weights(self) -> Optional[Tensor]:
        """Get last computed attention weights."""
        return self._attention_weights
    
    def update(self,
               neuron_id: int,
               current_state: Tensor,
               target_state: Tensor) -> AttentionState:
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
        neg_decay_rate = ops.multiply(ops.convert_to_tensor(-self.decay_rate), len(self.history))
        temporal_decay = ops.exp(neg_decay_rate)
        temporal_features = self.temporal_proj(current_state)
        state.temporal_weight = ops.multiply(ops.mean(temporal_features), temporal_decay)
        
        # Update causal weight
        prediction_error = ops.subtract(target_state, current_state)
        causal_input = ops.concatenate([current_state, prediction_error])
        causal_features = self.causal_proj(causal_input)
        one = ops.convert_to_tensor(1.0)
        prediction_accuracy = ops.subtract(one, ops.min(
            ops.norm(prediction_error),
            one
        ))
        state.causal_weight = ops.multiply(prediction_accuracy, ops.mean(
            causal_features
        ))
        
        # Update novelty weight
        novelty = self.novelty_proj(
            ops.subtract(target_state, current_state)
        )
        if ops.greater(ops.abs(novelty), self.novelty_threshold):
            state.novelty_weight = ops.abs(novelty)
        else:
            one_minus_decay = ops.subtract(one, self.decay_rate)
            state.novelty_weight = ops.multiply(state.novelty_weight, one_minus_decay)
        
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