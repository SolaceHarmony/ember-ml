"""Neural network parameter initialization module.

This module provides backend-agnostic weight initialization schemes for 
neural network parameters.

Components:
    Standard Initializations:
        - glorot_uniform: Glorot/Xavier uniform initialization
        - glorot_normal: Glorot/Xavier normal initialization
        - orthogonal: Orthogonal matrix initialization
        
    Specialized Initializations:
        - BinomialInitializer: Discrete binary initialization
        - binomial: Helper function for binomial initialization

All initializers maintain numerical stability and proper scaling
while preserving backend independence.
"""

from ember_ml.initializers.glorot import glorot_uniform, glorot_normal, orthogonal
from ember_ml.initializers.binomial import BinomialInitializer, binomial

__all__ = [
    'glorot_uniform',
    'glorot_normal',
    'orthogonal',
    'BinomialInitializer',
    'binomial',
]
