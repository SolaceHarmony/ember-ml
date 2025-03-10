"""
Keras-style neural network implementations.
"""

from .base import Layer
from .layers.rnn import (
    RNN,
    LTCCell,
    create_ltc_rnn
)

# List of public classes exposed by this module
__all__ = [
    'Layer',
    'RNN',
    'LTCCell',
    'create_ltc_rnn'
]

# Version of the keras module
__version__ = '0.1.0'

# Module level docstring
__doc__ += """

Components
----------
1. Base Classes
   - Layer: Base class for all layers

2. RNN Layers
   - RNN: Recurrent layer implementation
   - LTCCell: Liquid Time Constant cell
   - create_ltc_rnn: Factory function for LTC-RNN

Usage Examples
-------------
>>> from ember_ml.keras_3_8 import create_ltc_rnn

# Create LTC-RNN layer
>>> rnn = create_ltc_rnn(
...     input_size=10,
...     hidden_size=32,
...     tau=1.0,
...     return_sequences=True
... )

# Process sequence
>>> output = rnn(input_sequence)
"""