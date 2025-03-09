"""
Feature extraction module.

This module provides implementations of feature extraction,
including generic feature extraction, BigQuery feature extraction,
and column-based feature extraction.
"""

from ember_ml.features.generic_feature_extraction import *
from ember_ml.features.column_feature_extraction import *

try:
    from ember_ml.features.bigquery_feature_extraction import *
    __has_bigquery__ = True
except ImportError:
    __has_bigquery__ = False

__all__ = [
    'generic_feature_extraction',
    'bigquery_feature_extraction',
    'column_feature_extraction',
]
