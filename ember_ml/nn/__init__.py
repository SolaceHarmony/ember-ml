"""Neural network primitives and core components.

This module provides the foundational building blocks for constructing neural
networks with backend-agnostic implementations.
        
All components maintain strict backend independence through the ops abstraction.
"""

import ember_ml.nn.modules
import ember_ml.nn.layers
import ember_ml.nn.initializers

try:
    import ember_ml.features as features
except Exception as _features_exc:  # pragma: no cover - optional dependency
    class _FeaturesUnavailable:
        def __getattr__(self, item):
            raise ImportError(
                "ember_ml.features requires optional pandas dependencies that are "
                "not available in this environment."
            ) from _features_exc

    features = _FeaturesUnavailable()



__all__ = [
    'modules',
    'layers',
    'initializers',
    'features',
]
