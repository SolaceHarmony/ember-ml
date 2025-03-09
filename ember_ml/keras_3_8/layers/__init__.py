"""
Neural network layer implementations.
"""

from .rnn import (
    RNN,
    LTCCell,
    create_ltc_rnn
)

# List of public classes exposed by this module
__all__ = [
    'RNN',
    'LTCCell',
    'create_ltc_rnn'
]

# Module level docstring
__doc__ += """

Available Layers
---------------
1. RNN Layers
   - RNN: Base recurrent layer
   - LTCCell: Liquid Time Constant cell
   - create_ltc_rnn: Factory for LTC-RNN creation

Usage Examples
-------------
>>> from emberharmony.keras_3_8.layers import create_ltc_rnn

# Create and use LTC-RNN
>>> rnn = create_ltc_rnn(
...     input_size=10,
...     hidden_size=32
... )
>>> output = rnn(input_sequence)
"""