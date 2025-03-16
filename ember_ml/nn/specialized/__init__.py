"""
Specialized neurons module.

This module provides implementations of specialized neurons,
including attention neurons and base neurons.
"""

from ember_ml.nn.specialized.base import WeightedLTCNeuron
from ember_ml.nn.specialized.specialized import SpecializedNeuron
from ember_ml.nn.specialized.attention import LTCNeuronWithAttention

__all__ = [
    "LTCNeuronWithAttention",
    "WeightedLTCNeuron",
    "SpecializedNeuron"
]
