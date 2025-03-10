"""
Common feature extraction and transformation implementations.

This module provides backend-agnostic implementations of feature extraction and
transformation operations using the ops abstraction layer.
"""

from ember_ml.features.common.pca_features import PCA
from ember_ml.features.common.standardize_features import Standardize
from ember_ml.features.common.normalize_features import Normalize

__all__ = [
    'PCA',
    'Standardize',
    'Normalize',
]