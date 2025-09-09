"""Neural network primitives and core components.

This module provides the foundational building blocks for constructing neural
networks with backend-agnostic implementations.
        
All components maintain strict backend independence through the ops abstraction.
"""

import ember_ml.nn.modules
import ember_ml.nn.layers
import ember_ml.nn.initializers
import ember_ml.features as features



__all__ = [
    'modules',
    'layers',
    'initializers',
    'features',
]
