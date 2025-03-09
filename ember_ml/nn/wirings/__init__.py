"""
Wiring module for neural circuit policies.

This module provides wiring configurations for neural circuit policies,
which define the connectivity patterns between neurons.
"""

from ember_ml.nn.wirings.wiring import Wiring
from ember_ml.nn.wirings.full_wiring import FullyConnectedWiring
from ember_ml.nn.wirings.random_wiring import RandomWiring
from ember_ml.nn.wirings.ncp_wiring import NCPWiring
from ember_ml.nn.wirings.auto_ncp import AutoNCP
from ember_ml.nn.wirings.ncp import NCP

__all__ = [
    'Wiring',
    'FullyConnectedWiring',
    'RandomWiring',
    'NCPWiring',
    'AutoNCP',
    'NCP',
]