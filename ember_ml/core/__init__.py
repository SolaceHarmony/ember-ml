"""
Core Neural Network Implementations
=================================

This module provides the fundamental neural network implementations,
including LTC (Liquid Time Constant) neurons and their geometric variants.

Classes
-------
BaseNeuron
    Base class for all neural implementations
LTCNeuron
    Base implementation of Liquid Time Constant neuron
GeometricNeuron
    Base class for geometry-aware neural processing
SphericalLTCNeuron
    LTC neuron with spherical geometry processing
HybridNeuron
    Combined architecture supporting multiple processing modes
BlockyRoadNeuron
    Discrete-step neural implementation with quantized activation

Example Usage
------------
>>> from ember_ml.core import LTCNeuron, SphericalLTCNeuron

# Create basic LTC neuron
>>> neuron = LTCNeuron(neuron_id=1, tau=1.0, dt=0.01)
>>> output = neuron.update(input_signal)

# Use spherical variant
>>> spherical = SphericalLTCNeuron(neuron_id=1, dim=3)
>>> output = spherical.update(input_vector)

# Use hybrid neuron with attention
>>> from ember_ml.core import HybridNeuron
>>> hybrid = HybridNeuron(neuron_id=1, hidden_size=64)
>>> output = hybrid.update(input_tensor)

# Use blocky road neuron
>>> from ember_ml.core import BlockyRoadNeuron
>>> blocky = BlockyRoadNeuron(neuron_id=1)
>>> output = blocky.update(input_signal)
"""

from .base import BaseNeuron
from .ltc import LTCNeuron
from .geometric import GeometricNeuron
from .spherical_ltc import SphericalLTCNeuron, SphericalLTCConfig, SphericalLTCChain
from .hybrid import HybridNeuron
from .blocky import BlockyRoadNeuron

# List of public classes exposed by this module
__all__ = [
    'BaseNeuron',
    'LTCNeuron',
    'GeometricNeuron',
    'SphericalLTCNeuron', 'SphericalLTCConfig', 'SphericalLTCChain',
    'HybridNeuron',
    'BlockyRoadNeuron',
]

# Version of the core module
__version__ = '0.1.0'

# Module level docstring
__doc__ = """
Neural Network Core Module
=========================

Provides core implementations of neural architectures:

- LTC (Liquid Time Constant) neurons
- Geometric and spherical variants
- Hybrid architectures with attention
- Blocky road implementations with quantized activation

The module focuses on biologically-inspired neural computations
with an emphasis on temporal dynamics and geometric processing.

Key Features:
- Temporal integration with variable time constants
- Geometric and spherical coordinate processing
- Hybrid architectures combining multiple approaches
- Quantized implementations for specific use cases
"""