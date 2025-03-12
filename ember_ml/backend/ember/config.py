"""
Ember backend configuration for ember_ml.

This module provides configuration settings for the Ember backend.
"""

# Default device for Ember operations
DEFAULT_DEVICE = 'cpu'

# Default data type for Ember operations
# Using string representation since we don't have a specific dtype module for Ember
DEFAULT_DTYPE = 'float32'

# Current random seed
_current_seed = None