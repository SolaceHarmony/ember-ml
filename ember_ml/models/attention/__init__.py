"""
Attention mechanisms module.

This module provides implementations of attention mechanisms,
including temporal and causal attention.
"""

# Import from the shared attention mechanisms package
from ember_ml.nn.attention.mechanisms import (
    CausalAttention,
    AttentionState,
)

__all__ = [
    "CausalAttention",
    "AttentionState"
    ]
