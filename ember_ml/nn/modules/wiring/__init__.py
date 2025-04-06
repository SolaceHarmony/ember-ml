# ember_ml/nn/modules/wiring/__init__.py
"""
NeuronMap implementations defining neural connectivity structures.
"""

from .neuron_map import NeuronMap
from .fully_connected_map import FullyConnectedMap
from .ncp_map import NCPMap
from .random_map import RandomMap
# Note: AutoNCPMap is not here, it's a layer convenience class in modules/auto_ncp.py

__all__ = [
    "NeuronMap",
    "FullyConnectedMap",
    "NCPMap",
    "RandomMap",
]