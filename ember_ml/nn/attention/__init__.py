"""
Specialized neurons module.

This module provides implementations of specialized neurons,
including attention neurons and base neurons.
"""

# Import classes now located within this package
from .base import BaseAttention, AttentionLayer, MultiHeadAttention, AttentionMask, AttentionScore
from .causal import CausalAttention, PredictionAttention, AttentionState # CausalAttention moved here
from .temporal import TemporalAttention, PositionalEncoding
from .mechanisms import CausalAttention as MechanismCausalAttention # Import mechanism if different
from .attention import LTCNeuronWithAttention # Keep this if it's still relevant
# Removed WeightedLTCNeuron and SpecializedNeuron as they were deleted or moved elsewhere

__all__ = [
    # Core Attention Classes
    "BaseAttention",
    "AttentionLayer",
    "MultiHeadAttention",
    # Specific Implementations
    "CausalAttention",
    "PredictionAttention",
    "TemporalAttention",
    # Utilities / Supporting classes
    "AttentionMask",
    "AttentionScore",
    "AttentionState",
    "PositionalEncoding",
    # Specialized Neuron (If still relevant)
    "LTCNeuronWithAttention",
    # Note: MechanismCausalAttention might be redundant if same as models.attention.causal.CausalAttention
]
