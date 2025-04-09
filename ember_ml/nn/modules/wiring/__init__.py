# ember_ml/nn/modules/wiring/__init__.py
"""
NeuronMap implementations defining neural connectivity structures.
"""

from ember_ml.nn.modules.wiring.neuron_map import NeuronMap
from ember_ml.nn.modules.wiring.fully_connected_map import FullyConnectedMap
from ember_ml.nn.modules.wiring.ncp_map import NCPMap
from ember_ml.nn.modules.wiring.random_map import RandomMap
from ember_ml.nn.modules.wiring.fully_connected_ncp_map import FullyConnectedNCPMap
from ember_ml.nn.modules.wiring.language_wiring import LanguageWiring
from ember_ml.nn.modules.wiring.robotics_wiring import RoboticsWiring
from ember_ml.nn.modules.wiring.signal_wiring import SignalWiring
from ember_ml.nn.modules.wiring.frequency_wiring import FrequencyWiring
from ember_ml.nn.modules.wiring.vision_wiring import VisionWiring

__all__ = [
    "NeuronMap",
    "FullyConnectedMap",
    "NCPMap",
    "RandomMap",
    "FullyConnectedNCPMap",
    "LanguageWiring",
    "RoboticsWiring",
    "SignalWiring",
    "FrequencyWiring",
    "VisionWiring",
]