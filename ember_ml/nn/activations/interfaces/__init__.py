"""Activation function interfaces for neural network components.

This module defines the abstract interfaces for activation functions used in 
neural networks. All implementations maintain strict backend independence through
the ops abstraction layer.

Key Components:
    ActivationInterface: Base interface for all activation functions
    - Defines common activation function properties
    - Ensures backend-agnostic implementations
    - Provides consistent error handling
    - Maintains type safety across backends
"""

from ember_ml.nn.activations.interfaces.activation import ActivationInterface

__all__ = [
    'ActivationInterface',
]