"""
Wave-based neural processing module.

This module provides implementations of wave-based neural processing,
including binary wave neurons, harmonic embeddings, and wave memory analysis.
"""

# Import the classes directly from the harmonic.py file
from ember_ml.wave.harmonic import HarmonicProcessor, FrequencyAnalyzer, WaveSynthesizer

# Import the rest of the modules
from ember_ml.wave.binary import *
from ember_ml.wave.memory import *
from ember_ml.wave.audio import *
from ember_ml.wave.limb import *

# Import the harmonic module itself
import ember_ml.wave.harmonic

__all__ = [
    'binary',
    'harmonic',
    'memory',
    'audio',
    'limb',
    'HarmonicProcessor',
    'FrequencyAnalyzer',
    'WaveSynthesizer',
]
